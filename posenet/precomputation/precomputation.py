from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml
from einops import rearrange
from torch import nn, Tensor
from tensordict import TensorDict
from torch.nn.functional import interpolate

from ..dataset import Batch
from posenet.flow import FlowPredictorCfg, get_flow_predictor, Flows
from ..backbone import BackboneCfg, get_backbone
from posenet.tracking import TrackPredictorCfg, get_track_predictor, compute_tracks_with_predictor, Tracks
from ..feature import get_features_predictor, FeatureCfg
from ..feature.yolo import YOLO_Objects
from ..keypoints.keypoints import KeypointsCfg, KeypointsPredictor, Keypoints


class Precomputation:

    def __init__(self, cache_path: Path, dataset_name: str, sampler_name: str, cameras_name: list[str],
                 backbone_depth: BackboneCfg,
                 backbone_flow: FlowPredictorCfg,
                 backbone_keypoints: KeypointsCfg,
                 backbone_tracks: TrackPredictorCfg,
                 backbone_features: FeatureCfg,
                 device: torch.device,
                 image_shape: list[int, int],
                 sequential: bool):
        super().__init__()
        self.bar = None
        self.image_shape = image_shape[0], image_shape[1]
        self.sequential = sequential
        if self.sequential:
            self.flowCfg = backbone_flow
            self.trackCfg = backbone_tracks
        self.depthCfg = backbone_depth
        self.device = device
        self.cache_path = str(cache_path)
        self.dataset_name = dataset_name
        self.sampler_name = sampler_name
        self.cameras_name = cameras_name
        self.backbone_depth = get_backbone(backbone_depth)
        self.backbone_features, self.feature_object = get_features_predictor(backbone_features)
        if self.sequential:
            self.backbone_flow = get_flow_predictor(backbone_flow)
            self.backbone_tracks = get_track_predictor(backbone_tracks)
            self.backbone_keypoints = KeypointsPredictor(backbone_keypoints)

        # Create the cache directory and subdirectories if they don't already exist
        depth_engine = self.backbone_depth.cfg.name
        self.path_depth = Path(self.cache_path + f'/depths/{depth_engine}/{dataset_name}/')
        self.path_depth.parent.mkdir(exist_ok=True, parents=True)
        # self.path_features = Path(self.cache_path + f'/objects/{dataset_name}/')
        # self.path_features.parent.mkdir(exist_ok=True, parents=True)
        if self.sequential:
            self.path_flows = Path(self.cache_path + f'/{sampler_name}/flows/{dataset_name}/')
            self.path_flows.parent.mkdir(exist_ok=True, parents=True)
            self.path_tracks = Path(self.cache_path + f'/{sampler_name}/tracks/{dataset_name}/')
            self.path_tracks.parent.mkdir(exist_ok=True, parents=True)
            self.path_keypoints = Path(self.cache_path + f'/{sampler_name}/keypoints/{dataset_name}/')
            self.path_keypoints.parent.mkdir(exist_ok=True, parents=True)

        # Create a dictionary to store pre-computed features
        self.computed_depths = {f'{cam}': [] for cam in self.cameras_name}
        self.computed_features = {f'{cam}': [] for cam in self.cameras_name}
        self.computed_flows = {f'{cam}': [] for cam in self.cameras_name}
        self.computed_tracks = {f'{cam}': [] for cam in self.cameras_name}
        self.computed_keypoints = {f'{cam}': [] for cam in self.cameras_name}

        # Load pre-computed features indexes from cache if available
        try:
            with open(f'{self.cache_path}/indexes.yaml', 'r') as files_indexes:
                dict_idx = yaml.safe_load(files_indexes)
                if dict_idx is not None:
                    try:
                        self.computed_depths.update(dict_idx[f'depths_{depth_engine}'])
                    except (KeyError or TypeError):
                        pass
                    try:
                        self.computed_features.update(dict_idx['objects'])
                    except (KeyError or TypeError):
                        pass
                    if self.sequential:
                        try:
                            self.computed_flows.update(dict_idx['flows'][self.sampler_name])
                        except (KeyError or TypeError):
                            pass
                        try:
                            self.computed_keypoints.update(dict_idx['keypoints'][self.sampler_name])
                        except (KeyError or TypeError):
                            pass
                        try:
                            self.computed_tracks.update(dict_idx['tracks'][self.sampler_name])
                        except (KeyError or TypeError):
                            pass
        except FileNotFoundError:
            pass

    def to(self, device):
        self.backbone_depth = self.backbone_depth.to(device)
        if self.sequential:
            self.backbone_flow = self.backbone_flow.to(device)
            self.backbone_tracks = self.backbone_tracks.to(device)
            self.backbone_keypoints.to(device)

    @torch.no_grad()
    def __call__(self, batch: dict, tracks=False, depths=True, flows=False, objects=False, keypoints=False):
        b, f, *_ = batch['videos'].shape
        device = batch['videos'].device
        self.to(device)
        batch['videos'] = rearrange(interpolate(rearrange(batch['videos'], 'b f c h w -> (b f) c h w'),
                                                self.image_shape), '(b f) c h w -> b f c h w', b=b, f=f)

        if depths:
            depths = self.process_depth(batch)
        else:
            depths = None
            # features = None
        if objects:
            objects = self.process_objects(batch)
        else:
            objects = None

        if self.sequential:
            tracks = self.process_tracks(batch) if tracks else None
            flows = self.process_flow(batch) if flows else None
            keypoints = self.process_keypoints(batch) if keypoints else None
        else:
            tracks = None
            flows = None
            keypoints = None
        # self.bar.close()
        with open(f'{self.cache_path}/indexes.yaml', 'w') as files_indexes:
            dict_indexes = {'depths': self.computed_depths,
                            'objects': self.computed_features,
                            'flows': {self.sampler_name: self.computed_flows},
                            'tracks': {self.sampler_name: self.computed_tracks},
                            'keypoints': {self.sampler_name: self.computed_keypoints}}
            yaml.dump(dict_indexes, files_indexes)
        batch.update({'depths': depths,
                      'flows': flows,
                      'tracks': tracks,
                      # 'features': features,
                      'objects': objects,
                      'keypoints': keypoints})
        self.to(torch.device('cpu'))
        return Batch(**batch)

    def process_tracks(self, batch: dict) -> list[Tracks]:

        segments_computed = [[np.arange(int(x.split('_To_')[0]), int(x.split('_To_')[1]) + 1, 1)
                              for x in self.computed_tracks[f'{cam}']] for cam in self.cameras_name]
        segments_cams = [[] for _ in self.cameras_name]
        for i, segments in enumerate(segments_computed):
            for segment in segments:
                segments_cams[i].extend(segment.tolist())
        indices_to_be_computed = list(
            [i for i, idx in enumerate(batch['indices']) if idx not in np.array(segments).flatten().tolist()]
            for segments in segments_cams)

        indices_to_be_loaded = list([f'{seg[0]}_To_{seg[-1]}' for seg in segments if
                                     (seg[0] not in idx_t and
                                      batch['indices'][-1] >= seg[0] >= batch['indices'][0] and
                                      batch['indices'][-1] >= seg[-1] >= batch['indices'][0])] for segments, idx_t in
                                    zip(segments_computed, indices_to_be_computed))

        segments = [[] for _ in self.cameras_name]
        indices_computed = [[] for _ in self.cameras_name]
        for i, indices in enumerate(indices_to_be_computed):
            if indices:
                # self.bar.set_description(desc='Compute Tracks camera 1')
                images = batch['videos'][i, indices]
                while images.ndim < 5:
                    images = images[None]
                try:
                    out = compute_tracks_with_predictor(self.backbone_tracks, images, self.trackCfg.interval,
                                                        self.trackCfg.radius, batch['indices'])
                except torch.OutOfMemoryError:
                    try:
                        b, f, c, h, w = images.shape
                        images = rearrange(interpolate(rearrange(images, 'b f c h w -> (b f) c h w'), scale_factor=0.5),
                                           '(b f) c h w -> b f c h w', b=b, f=f)
                        out = compute_tracks_with_predictor(self.backbone_tracks, images, self.trackCfg.interval,
                                                            self.trackCfg.radius, batch['indices'])
                    except torch.OutOfMemoryError:
                        out = []
                # self.bar.update(len(indices_to_compute[0]))
                for tracks_segment in out:
                    start, end = tracks_segment.start_frame, tracks_segment.start_frame + tracks_segment.xy.shape[1] - 1
                    indices_computed[i].append(f'{batch["indices"][start]}_To_{batch["indices"][end]}')
                    path = Path(
                        f'{self.path_tracks}/{self.cameras_name[i]}/{batch["indices"][start]}_To_{batch["indices"][end]}')
                    path.parent.mkdir(exist_ok=True, parents=True)
                    TensorDict({'xy': tracks_segment.xy.cpu(),
                                'visibility': tracks_segment.visibility.cpu(),
                                'start_frame': start}).save(str(path), return_early=False)
                segments[i].extend(out)
        for i, indices in enumerate(indices_to_be_loaded):
            for j in indices:
                # self.bar.set_description(desc=f'Load Tracks camera {self.cameras_name[i]}')
                tensor_dict = TensorDict.load(f'{self.path_tracks}/{self.cameras_name[i]}/{j}')
                segments[i].append(Tracks(**tensor_dict))
                # self.bar.update()
        try:
            assert all([len(segments[i]) == len(segments[i + 1]) for i in range(len(segments) - 1)])
            for i in range(len(segments) - 1):
                assert all(
                    [segments[i][j].start_frame == segments[i + 1][j].start_frame for j in range(len(segments[i]))])

            tracks_seg = [Tracks(xy=torch.cat([seg.xy.to(self.device) for seg in segment]),
                                 visibility=torch.cat([seg.visibility.to(self.device) for seg in segment]),
                                 start_frame=segment[0].start_frame) for segment in zip(*segments)]
            for i, indices in enumerate(indices_computed):
                self.computed_tracks[self.cameras_name[i]].extend(indices)
        except AssertionError:
            tracks_seg = []
        return tracks_seg

    def process_flow(self, batch: dict) -> Flows:
        indices_to_be_computed = list(
            [(i, i + 1) for i, (idx_i, idx_j) in enumerate(zip(batch['indices'][:-1], batch['indices'][1:])) if
             f'{idx_i}_To_{idx_j}' not in self.computed_flows[f'{cam}']] for cam in self.cameras_name)

        indices_to_be_loaded = [[f'{i}_To_{j}' for idx, (i, j) in enumerate(zip(batch['indices'][:-1],
                                                                                batch['indices'][1:])) if
                                 (idx, idx + 1) not in k] for k in indices_to_be_computed]
        indices_in_batch_to_be_loaded = [[(i, i + 1) for i in range(len(batch['indices'][:-1])) if (i, i + 1) not in j]
                                         for j in indices_to_be_computed]

        backward = [[] for _ in self.cameras_name]
        backward_mask = [[] for _ in self.cameras_name]
        forward = [[] for _ in self.cameras_name]
        forward_mask = [[] for _ in self.cameras_name]
        for idx, indices in enumerate(indices_to_be_computed):
            if indices:
                # self.bar.set_description(desc='Compute Flows camera 1')
                images = batch['videos'][idx, indices]
                while images.ndim < 5:
                    images = images[None]
                f, _, c, h, w = images.shape
                images = rearrange(images, 'f p c h w -> (f p) c h w')
                images = rearrange(interpolate(images, scale_factor=self.flowCfg.scale_multiplier),
                                   '(f p) c h w -> f p c h w', f=f, p=2)
                out = self.backbone_flow.compute_bidirectional_flow(images, self.image_shape)
                backward[idx].extend(out.backward.split(1, 0))
                backward_mask[idx].extend(out.backward_mask.split(1, 0))
                forward[idx].extend(out.forward.split(1, 0))
                forward_mask[idx].extend(out.forward_mask.split(1, 0))
                for b, b_m, f, f_m, (i, j) in zip(backward[idx], backward_mask[idx], forward[idx], forward_mask[idx],
                                                  indices):
                    path = Path(
                        f'{self.path_flows}/{self.cameras_name[idx]}/{batch["indices"][i]}_To_{batch["indices"][j]}')
                    path.parent.mkdir(exist_ok=True, parents=True)
                    TensorDict({'backward': b.cpu(),
                                'backward_mask': b_m.cpu(),
                                'forward': f.cpu(),
                                'forward_mask': f_m.cpu()}).save(str(path), return_early=False)
                    # self.bar.update()
        for idx, indices in enumerate(indices_to_be_loaded):
            for i in indices:
                # self.bar.set_description(desc='Load Flows camera 1')
                tensor_dict = TensorDict.load(f'{self.path_flows}/{self.cameras_name[idx]}/{i}')
                b, f, h, w, c = tensor_dict['backward'].data.shape
                backward[idx].append(
                    rearrange(interpolate(rearrange(tensor_dict['backward'].data, 'b f h w c -> (b f) c h w'),
                                          self.image_shape), '(b f) c h w -> b f h w c', b=b, f=f).to(self.device))
                backward_mask[idx].append(
                    interpolate(tensor_dict['backward_mask'].data, self.image_shape).to(self.device))
                forward[idx].append(
                    rearrange(interpolate(rearrange(tensor_dict['forward'].data, 'b f h w c -> (b f) c h w'),
                                          self.image_shape), '(b f) c h w -> b f h w c', b=b, f=f).to(self.device))
                forward_mask[idx].append(
                    interpolate(tensor_dict['forward_mask'].data, self.image_shape).to(self.device))
                # self.bar.update()

        for idx, (indices, indices_in_batch) in enumerate(zip(indices_to_be_computed, indices_in_batch_to_be_loaded)):
            indices.extend(indices_in_batch)
            indices_sorted = np.argsort([i for i, j in indices])
            backward[idx] = torch.cat(backward[idx], dim=1)[:, indices_sorted]
            backward_mask[idx] = torch.cat(backward_mask[idx], dim=1)[:, indices_sorted]
            forward[idx] = torch.cat(forward[idx], dim=1)[:, indices_sorted]
            forward_mask[idx] = torch.cat(forward_mask[idx], dim=1)[:, indices_sorted]

        backward = torch.cat(backward, dim=0)
        backward_mask = torch.cat(backward_mask, dim=0)
        forward = torch.cat(forward, dim=0)
        forward_mask = torch.cat(forward_mask, dim=0)

        indices_computed = [[f'{idx_i}_To_{idx_j}' for idx_i, idx_j in zip(batch['indices'][:-1], batch['indices'][1:])
                             if f'{idx_i}_To_{idx_j}' not in self.computed_flows[cam]] for cam in self.cameras_name]
        for cam, indices in zip(self.cameras_name, indices_computed):
            self.computed_flows[cam].extend(indices)

        return Flows(backward=backward, backward_mask=backward_mask, forward=forward, forward_mask=forward_mask)

    def process_keypoints(self, batch: dict) -> Keypoints:
        indices_to_be_computed = list(
            [(i, i + 1) for i, (idx_i, idx_j) in enumerate(zip(batch['indices'][:-1], batch['indices'][1:])) if
             f'{idx_i}_To_{idx_j}' not in self.computed_keypoints[f'{cam}']] for cam in self.cameras_name)

        indices_to_be_loaded = [[f'{i}_To_{j}' for idx, (i, j) in enumerate(zip(batch['indices'][:-1],
                                                                                batch['indices'][1:])) if
                                 (idx, idx + 1) not in k] for k in indices_to_be_computed]
        indices_in_batch_to_be_loaded = [[(i, i + 1) for i in range(len(batch['indices'][:-1])) if (i, i + 1) not in j]
                                         for j in indices_to_be_computed]

        keypoints0 = [[] for _ in self.cameras_name]
        keypoints1 = [[] for _ in self.cameras_name]
        for idx, indices in enumerate(indices_to_be_computed):
            if indices:
                # self.bar.set_description(desc='Compute Keypoints camera 1')
                images = batch['videos'][idx, indices]
                while images.ndim < 5:
                    images = images[None]
                keypoints0_, keypoints1_ = self.backbone_keypoints(images)
                keypoints0[idx].extend(keypoints0_.split(1, 0))
                keypoints1[idx].extend(keypoints1_.split(1, 0))
                for kp0, kp1, (i, j) in zip(keypoints0[idx],
                                            keypoints1[idx],
                                            indices):
                    path = Path(
                        f'{self.path_keypoints}/{self.cameras_name[idx]}/{batch["indices"][i]}_To_{batch["indices"][j]}')
                    path.parent.mkdir(exist_ok=True, parents=True)
                    TensorDict({'keypoints0': kp0.cpu(),
                                'keypoints1': kp1.cpu()}).save(str(path), return_early=False)
                    # self.bar.update()
        for idx, indices in enumerate(indices_to_be_loaded):
            for i in indices:
                # self.bar.set_description(desc='Load Flows camera 1')
                tensor_dict = TensorDict.load(f'{self.path_keypoints}/{self.cameras_name[idx]}/{i}')
                keypoints0[idx].append(tensor_dict['keypoints0'].data.to(self.device))
                keypoints1[idx].append(tensor_dict['keypoints1'].data.to(self.device))
                # self.bar.update()

        for idx, (indices, indices_in_batch) in enumerate(zip(indices_to_be_computed, indices_in_batch_to_be_loaded)):
            indices.extend(indices_in_batch)
            indices_sorted = np.argsort([i for i, j in indices])
            keypoints0[idx] = torch.cat(keypoints0[idx], dim=0)[indices_sorted]
            keypoints1[idx] = torch.cat(keypoints1[idx], dim=0)[indices_sorted]
        l = [k.shape[1] for k in keypoints0]
        l.extend([k.shape[1] for k in keypoints1])
        nb_kp = min(l)
        keypoints0 = torch.stack([k[:, :nb_kp] for k in keypoints0], dim=0)
        keypoints1 = torch.stack([k[:, :nb_kp] for k in keypoints1], dim=0)

        indices_computed = [[f'{idx_i}_To_{idx_j}' for idx_i, idx_j in zip(batch['indices'][:-1], batch['indices'][1:])
                             if f'{idx_i}_To_{idx_j}' not in self.computed_keypoints[cam]] for cam in self.cameras_name]
        for cam, indices in zip(self.cameras_name, indices_computed):
            self.computed_keypoints[cam].extend(indices)

        return Keypoints(keypoints0=keypoints0, keypoints1=keypoints1)

    def process_depth(self, batch: dict) -> Tensor:

        indices_to_be_computed = list(
            [i for i, idx in enumerate(batch['indices']) if idx not in self.computed_depths[f'{cam}']] for cam in
            self.cameras_name)

        indices_to_be_loaded = list(
            [idx for i, idx in enumerate(batch['indices']) if i not in j] for j in indices_to_be_computed)

        depths = [[] for _ in self.cameras_name]
        # features = [[] for _ in self.cameras_name]
        indices_batch = batch['indices'].cpu().numpy()
        videos = batch['videos']
        indices_computed = [[] for _ in self.cameras_name]
        for i, indices in enumerate(indices_to_be_computed):
            if indices:
                # self.bar.set_description(desc='Compute Depths camera 1')
                images = videos[i, indices]
                while images.ndim < 5:
                    images = images[None]
                out = self.backbone_depth(images)
                out['metric_depth'] = interpolate(out['metric_depth'], self.image_shape)
                b, f, *_ = out['features'].shape
                # out['features'] = rearrange(
                #     interpolate(rearrange(out['features'], 'b f c h w -> (b f) c h w'), self.image_shape),
                #     '(b f) c h w -> b f c h w', b=b, f=f)
                depths[i].extend(out['metric_depth'].split(1, 1))
                # features[i].extend(out['features'].split(1, 1))
                for depth, j in zip(depths[i], indices_batch[indices]):
                    indices_computed[i].append(int(j))
                    path = Path(f'{self.path_depth}/{self.cameras_name[i]}/{j}')
                    path.parent.mkdir(exist_ok=True, parents=True)
                    TensorDict({'metric_depth': depth}).save(str(path), return_early=False)
                    # self.bar.update()
        for i, indices in enumerate(indices_to_be_loaded):
            if indices:
                for j in indices:
                    # self.bar.set_description(desc='Load Depths camera 1')
                    tensor_dict = TensorDict.load(f'{self.path_depth}/{self.cameras_name[i]}/{j}')
                    depth = interpolate(tensor_dict['metric_depth'].data.to(self.device), self.image_shape)
                    # feature = tensor_dict['features'].data.to(self.device)
                    # b, f, *_ = feature.shape
                    # feature = rearrange(interpolate(rearrange(feature, 'b f c h w -> (b f) c h w'), self.image_shape),
                    #                     '(b f) c h w -> b f c h w', b=b, f=f)
                    depths[i].append(depth)
                    # features[i].append(feature)
                    # self.bar.update()

        for i in range(len(self.cameras_name)):
            indices_to_be_loaded[i].extend(indices_computed[i])
            indices_sorted = np.argsort(indices_to_be_loaded[i])
            depths[i] = torch.cat(depths[i], dim=1)[:, indices_sorted]
            # features[i] = torch.cat(features[i], dim=1)[:, indices_sorted]

        depths = torch.cat(depths, dim=0)
        # features = torch.cat(features, dim=0)
        for cam, indices in zip(self.cameras_name, indices_computed):
            self.computed_depths[cam].extend(indices)
        return depths

    def process_objects(self, batch: dict) -> list[list[YOLO_Objects]]:

        indices_to_be_computed = list(
            [i for i, idx in enumerate(batch['indices']) if idx not in self.computed_features[f'{cam}']] for cam in
            self.cameras_name)

        indices_to_be_loaded = list(
            [idx for i, idx in enumerate(batch['indices']) if i not in j] for j in indices_to_be_computed)

        objects = [[] for _ in self.cameras_name]
        indices_batch = batch['indices'].cpu().numpy()
        videos = batch['videos']
        indices_computed = [[] for _ in self.cameras_name]
        for i, indices in enumerate(indices_to_be_computed):
            if indices:
                images = videos[i, indices]
                while images.ndim < 4:
                    images = images[None]
                classes_number, out = self.backbone_features(images)
                objects[i].extend(out)
                for k, (obj, j) in enumerate(zip(objects[i], indices_batch[indices])):
                    path = Path(f'{self.path_features}/{self.cameras_name[i]}/{j}')
                    path.parent.mkdir(exist_ok=True, parents=True)
                    if obj is not None:
                        TensorDict({cls: torch.stack(obj.__dict__[cls]) if cls != 'classes_number'
                        else obj.__dict__[cls] for cls in obj.__dict__.keys()}).save(str(path), return_early=False)
                    else:
                        TensorDict({'classes_number': classes_number}).save(str(path), return_early=False)
                    indices_to_be_loaded[i].append(int(j))
                objects[i] = []
                # self.bar.update()
        for i, indices in enumerate(indices_to_be_loaded):
            if indices:
                for j in indices:
                    # self.bar.set_description(desc='Load Depths camera 1')
                    tensor_dict = TensorDict.load(f'{self.path_features}/{self.cameras_name[i]}/{j}')
                    data = {k: torch.tensor(v) for k, v in tensor_dict.items()}
                    classes = list(data.keys())
                    classes.remove('classes_number')
                    obj = self.feature_object(**data).to(self.device) if classes else None
                    objects[i].append(obj)

        for i in range(len(self.cameras_name)):
            indices_to_be_loaded[i].extend(indices_computed[i])
            indices_sorted = np.argsort(indices_to_be_loaded[i])
            objects[i] = np.array(objects[i])[indices_sorted].tolist()

        for cam, indices in zip(self.cameras_name, indices_computed):
            self.computed_features[cam].extend(indices)
        return objects
