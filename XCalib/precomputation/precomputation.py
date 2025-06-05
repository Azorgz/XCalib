from pathlib import Path
import numpy as np
import torch
import yaml
from einops import rearrange
from torch import nn, Tensor
from tensordict import TensorDict
from torch.nn.functional import interpolate

from ..dataset import Batch
from ..backbone import BackboneCfg, get_backbone


class Precomputation:

    def __init__(self, cache_path: Path, dataset_name: str, sampler_name: str, cameras_name: list[str],
                 backbone_depth: BackboneCfg,
                 device: torch.device,
                 image_shape: list[int, int]):
        super().__init__()
        self.bar = None
        self.image_shape = image_shape[0], image_shape[1]
        self.depthCfg = backbone_depth
        self.device = device
        self.cache_path = str(cache_path)
        self.dataset_name = dataset_name
        self.sampler_name = sampler_name
        self.cameras_name = cameras_name
        self.backbone_depth = get_backbone(backbone_depth)

        # Create the cache directory and subdirectories if they don't already exist
        depth_engine = self.backbone_depth.cfg.name
        self.path_depth = Path(self.cache_path + f'/depths/{depth_engine}/{dataset_name}/')
        self.path_depth.parent.mkdir(exist_ok=True, parents=True)

        # Create a dictionary to store pre-computed depths
        self.computed_depths = {f'{cam}': [] for cam in self.cameras_name}

        # Load pre-computed features indexes from cache if available
        try:
            with open(f'{self.cache_path}/indexes.yaml', 'r') as files_indexes:
                dict_idx = yaml.safe_load(files_indexes)
                if dict_idx is not None:
                    try:
                        self.computed_depths.update(dict_idx[f'depths_{depth_engine}'])
                    except (KeyError or TypeError):
                        pass
        except FileNotFoundError:
            pass

    def to(self, device):
        self.backbone_depth = self.backbone_depth.to(device)

    @torch.no_grad()
    def __call__(self, batch: dict):
        b, f, *_ = batch['videos'].shape
        device = batch['videos'].device
        self.to(device)
        batch['videos'] = rearrange(interpolate(rearrange(batch['videos'], 'b f c h w -> (b f) c h w'),
                                                self.image_shape), '(b f) c h w -> b f c h w', b=b, f=f)

        depths = self.process_depth(batch)

        with open(f'{self.cache_path}/indexes.yaml', 'w') as files_indexes:
            dict_indexes = {'depths': self.computed_depths}
            yaml.dump(dict_indexes, files_indexes)
        batch.update({'depths': depths})
        self.to(torch.device('cpu'))
        return Batch(**batch)

    def process_depth(self, batch: dict) -> Tensor:

        indices_to_be_computed = list(
            [i for i, idx in enumerate(batch['indices']) if idx not in self.computed_depths[f'{cam}']] for cam in
            self.cameras_name)

        indices_to_be_loaded = list(
            [idx for i, idx in enumerate(batch['indices']) if i not in j] for j in indices_to_be_computed)

        depths = [[] for _ in self.cameras_name]
        indices_batch = batch['indices'].cpu().numpy()
        videos = batch['videos']
        indices_computed = [[] for _ in self.cameras_name]
        for i, indices in enumerate(indices_to_be_computed):
            if indices:
                images = videos[i, indices]
                while images.ndim < 5:
                    images = images[None]
                out = self.backbone_depth(images)
                out['metric_depth'] = interpolate(out['metric_depth'], self.image_shape)
                depths[i].extend(out['metric_depth'].split(1, 1))
                for depth, j in zip(depths[i], indices_batch[indices]):
                    indices_computed[i].append(int(j))
                    path = Path(f'{self.path_depth}/{self.cameras_name[i]}/{j}')
                    path.parent.mkdir(exist_ok=True, parents=True)
                    TensorDict({'metric_depth': depth}).save(str(path), return_early=False)
        for i, indices in enumerate(indices_to_be_loaded):
            if indices:
                for j in indices:
                    tensor_dict = TensorDict.load(f'{self.path_depth}/{self.cameras_name[i]}/{j}')
                    depth = interpolate(tensor_dict['metric_depth'].data.to(self.device), self.image_shape)
                    depths[i].append(depth)

        for i in range(len(self.cameras_name)):
            indices_to_be_loaded[i].extend(indices_computed[i])
            indices_sorted = np.argsort(indices_to_be_loaded[i])
            depths[i] = torch.cat(depths[i], dim=1)[:, indices_sorted]

        depths = torch.cat(depths, dim=0)
        for cam, indices in zip(self.cameras_name, indices_computed):
            self.computed_depths[cam].extend(indices)
        return depths
