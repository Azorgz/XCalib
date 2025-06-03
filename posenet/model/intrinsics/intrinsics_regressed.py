from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor, nn

from ..relative_pose.relative_pose import RelativePoseOutput
from ...dataset.types import Batch
from .intrinsics import Intrinsics


@dataclass
class IntrinsicsRegressedCfg:
    name: Literal["regressed"]
    initial_focal_length: float | list[float]
    random_init: float
    image_shape: list[int] | None = None
    freeze_center: bool = True
    freeze_shear: bool = True


class IntrinsicsRegressed(Intrinsics[IntrinsicsRegressedCfg]):
    def __init__(self, cfg: IntrinsicsRegressedCfg, num_cams: int) -> None:
        super().__init__(cfg, num_cams)
        # focal_length = torch.full(tuple(), cfg.initial_focal_length, dtype=torch.float32)
        if isinstance(cfg.initial_focal_length, list):
            assert len(cfg.initial_focal_length) == num_cams
            focal_length = torch.tensor(cfg.initial_focal_length, dtype=torch.float32)[:, None]
        else:
            focal_length = torch.tensor([cfg.initial_focal_length]*num_cams, dtype=torch.float32)[:, None]
        focal_length += (torch.rand(num_cams)[:, None]-0.5) * cfg.random_init
        self.focal_length = nn.Parameter(focal_length)

    def forward(
        self,
        batch: Batch,
        backbone_output: Tensor,
        relative_pose: RelativePoseOutput,
        global_step: int,
        initial_pose: Float[Tensor, 'batch 4 4']
    ) -> Float[Tensor, " *batch"]:
        self.last_value = self.focal_length
        return self.last_value
