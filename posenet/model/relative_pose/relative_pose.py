from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from jaxtyping import Float
from kornia.geometry import Rt_to_matrix4x4, quaternion_to_rotation_matrix
from torch import Tensor, nn

from ...dataset.types import Batch

T = TypeVar("T")


@dataclass
class RelativePoseOutput:
    translation: Float[Tensor, "b 3 1"]
    rotation: Float[Tensor, "b 4 1"]
    focal_length: Float[Tensor, "b 1"]
    center: Float[Tensor, "b 2"]
    shear: Float[Tensor, "b 2"]
    Rt: Float[Tensor, "b 4 4"]


class RelativePose(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T, num_cams: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_cams = num_cams

    @abstractmethod
    def forward(
            self,
            batch: Batch,
            global_step: int,
            params=None
    ) -> RelativePoseOutput:
        pass

    def explicit_pose(self, params=None):
        rot_delta = params[:, 1:5].mean(dim=0, keepdim=True) if params is not None else 0
        rotation = (self.rotation if self.rotation is not None else torch.tensor([1., 0, 0, 0])[None]
                    .repeat(self.num_cams-1, 1)) + rot_delta
        trans_delta = (params[:, 5:8].mean(dim=0, keepdim=True) * (10 ** (params[:, -1]))
                       .mean(dim=0)) if params is not None else 0
        translation = (self.translation if self.translation is not None else torch.tensor([0, 0, 0])[None]
                       .repeat(self.num_cams-1, 1)) + trans_delta
        return Rt_to_matrix4x4(quaternion_to_rotation_matrix(rotation), translation[..., None])

    def export(self, params=None):
        focal_length = (self.focal_length if self.focal_length is not None else torch.tensor([0.85]).unsqueeze(0))
        center = self.center
        shear = self.shear
        rotation = (self.rotation if self.rotation is not None else torch.tensor([1, 0, 0, 0]))
        translation = (self.translation if self.translation is not None else torch.tensor([0, 0, 0]).unsqueeze(0))
        return RelativePoseOutput(translation, rotation, focal_length, center, shear, self.explicit_pose(params))
