from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn
from utils.misc import get_gpu_memory

T = TypeVar("T")

class Backbone(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(
        self,
        cfg: T,
    ) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(self, batch):
        pass

    def infer_depth_memory_save(self,
                                videos: Float[Tensor, "batch channel height width"],
                                ressource: float = 1.) -> (
            Float)[Tensor, "batch frame height width"]:
        max_batch = max(int(np.array([m // ressource for m in get_gpu_memory()]).sum()), 1)
        videos_splited = videos.split(max_batch, dim=0)
        depths = []
        for video in videos_splited:
            depths.append(self.model(video))
        return {f'{k}': torch.cat([v[f'{k}'] for v in depths]) for k in depths[0].keys() if k != 'focallength_px'}
