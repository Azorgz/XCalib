from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from beartype.typing import Tuple
from jaxtyping import Float
from torch import Tensor, nn

from ..relative_pose.relative_pose import RelativePoseOutput
from ...dataset.types import Batch

T = TypeVar("T")


@dataclass
class IntrinsicsOutput:
    focal_lengths: Float[Tuple, '...']


class Intrinsics(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T, num_cams: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_cams = num_cams
        base_focal_length = torch.tensor([1], dtype=torch.float32)
        self.last_value = base_focal_length[None].repeat(self.num_cams, 1)

    @abstractmethod
    def forward(
        self,
        batch: Batch,
        backbone_output: Tensor,
        relative_pose: RelativePoseOutput,
        global_step: int,
        initial_pose: Float[Tensor, 'batch 4 4']
    ) -> Tensor:
        pass
