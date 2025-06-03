from pathlib import Path

import torch
import torch.utils.data as data
from lightning.pytorch import LightningDataModule
from torch import Generator
from torch.utils.data import DataLoader

from posenet.precomputation.precomputation import Precomputation
from . import get_dataset
from .types import collate_batch_fn
from ..config.posenet import PoseNetCfg


class SequentialDataLoader(DataLoader):
    def __init__(
            self,
            dataset: data.Dataset,
            batch_size: int,
            rep_batch: int,
            precomputation_module: Precomputation,
            device: torch.device,
            shuffle: bool,
            **kwargs):
        super().__init__(dataset, batch_size, shuffle=shuffle, **kwargs)
        self.rep_batch = rep_batch
        self.precomputation = precomputation_module
        self.device = device

    def __len__(self):
        return self.rep_batch * super().__len__()

    def __iter__(self):
        for batch in super().__iter__():
            self.batch = self.precomputation(batch).to(self.device)
            for _ in range(self.rep_batch):
                yield self.batch


class RandomDataLoader(DataLoader):
    def __init__(
            self,
            dataset: data.Dataset,
            precomputation_module: Precomputation,
            device: torch.device,
            **kwargs):
        super().__init__(dataset, 1, shuffle=True, **kwargs)
        self.precomputation = precomputation_module
        self.device = device

    def __len__(self):
        return 1

    def __iter__(self):
        batch = next(super().__iter__())
        batch = self.precomputation(batch).to(self.device)
        yield batch


class DataModule(LightningDataModule):
    def __init__(
            self,
            device: torch.device,
            cfg: PoseNetCfg,
            dataset,
            global_rank: int = 0,
            sequential: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.cfg = cfg
        self.dataset_cfgs = cfg.dataset
        self.data_module_cfg = cfg.datamodule
        self.frame_sampler_cfg = cfg.frame_sampler
        self.global_rank = global_rank
        self.sequential = sequential
        cache_path = Path(dataset.path).parent.joinpath(Path(self.data_module_cfg.cache_path))
        self.dataset = dataset
        self.last_batch = None
        cfg.depth.stage = dataset.stage
        self.precomputation_module = Precomputation(cache_path,
                                                    dataset.cfg.name,
                                                    dataset.frame_sampler.name,
                                                    [dataset.cameras[i].name for i in range(len(dataset.cameras))],
                                                    cfg.depth,
                                                    cfg.flow,
                                                    cfg.keypoints,
                                                    cfg.tracking,
                                                    cfg.feature,
                                                    self.device,
                                                    cfg.model.image_shape,
                                                    self.sequential)

    def get_generator(self, loader_cfg) -> torch.Generator | None:
        if loader_cfg.seed is None:
            return None
        generator = Generator()
        generator.manual_seed(loader_cfg.seed + self.global_rank)
        return generator

    def train_dataloader(self):
        return SequentialDataLoader(
            self.dataset,
            1,
            self.data_module_cfg.train.rep_batch,
            self.precomputation_module,
            self.device,
            self.data_module_cfg.shuffle,
            num_workers=0,
            generator=self.get_generator(self.data_module_cfg.train),
            collate_fn=collate_batch_fn)

    # def test_dataloader(self):
    #     return DataLoader(
    #         self.test,
    #         self.data_module_cfg.train.batch_size,
    #         num_workers=self.data_module_cfg.train.num_workers,
    #         generator=self.get_generator(self.data_module_cfg.train),
    #         collate_fn=collate_batch_fn,
    #         # worker_init_fn=worker_init_fn,
    #         # persistent_workers=self.get_persistent(self.data_module_cfg.train),
    #     )
    #
    def val_dataloader(self):
        return RandomDataLoader(
            self.dataset,
            self.precomputation_module,
            self.device,
            num_workers=0,
            generator=self.get_generator(self.data_module_cfg.val),
            collate_fn=collate_batch_fn)


def get_dataModule(device: torch.device, cfg, sequential: bool = False) -> DataModule | list:
    for i in range(len(cfg.dataset)):
        cfg.dataset[i].normalize_visible = cfg.datamodule.normalize_visible
        cfg.dataset[i].normalize_infrared = cfg.datamodule.normalize_infrared
        cfg.dataset[i].equalize_visible = cfg.datamodule.equalize_visible
        cfg.dataset[i].equalize_infrared = cfg.datamodule.equalize_infrared
        cfg.dataset[i].nb_cam = cfg.datamodule.nb_cam
    dataset = get_dataset(cfg.dataset, cfg.frame_sampler)
    if isinstance(dataset, list):
        return [DataModule(device, cfg, dataset[i], sequential=sequential) for i in range(len(dataset))]
    else:
        return DataModule(device, cfg, dataset, sequential=sequential)
