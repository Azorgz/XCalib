from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch
from jaxtyping import Float
from torch import Tensor, nn

from ..relative_pose.relative_pose import RelativePoseOutput
from ...dataset.types import Batch

T = TypeVar("T")


class Extrinsics(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T, frames: int = 2) -> None:
        super().__init__()
        self.cfg = cfg
        self.cfg.frames = frames
        self._trajectory = {}

    @property
    def trajectory(self):
        return self._trajectory

    @abstractmethod
    def forward(
            self,
            batch: Batch,
            weights: Tensor,
            surfaces: Float[Tensor, "batch frame height width 3"],
            initial_pose: Float[Tensor, '1 4 4'],
            new_batch: bool
    ) -> Float[Tensor, "batch frame 4 4"]:
        pass
