from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float
from kornia.geometry import Rt_to_matrix4x4, quaternion_to_rotation_matrix
from torch import Tensor, nn

from .extrinsics import Extrinsics
from ..relative_pose.relative_pose import RelativePoseOutput
from ...dataset.types import Batch


@dataclass
class ExtrinsicsRegressedCfg:
    name: Literal["regressed"]
    frames: int = 2


class ExtrinsicsRegressed(Extrinsics[ExtrinsicsRegressedCfg]):
    initialized: bool = False

    def __init__(self, cfg):
        super().__init__(cfg)
        self.rotation = nn.Parameter(torch.tensor([1., 0., 0., 0.])[None].repeat(cfg.frames, 1))
        self.translation = nn.Parameter(torch.zeros(1, 3).repeat(cfg.frames, 1))

    def forward(
        self,
        batch: Batch,
        weights: Tensor,
        surfaces: Float[Tensor, "batch frame height width 3"],
        initial_pose: Float[Tensor, '1 4 4'],
        new_batch: bool
    ) -> Float[Tensor, "batch frame 4 4"]:
        if not self.initialized:
            self.initialize(self.rotation, self.translation)
        trajectory = self.explicit_pose()
        if not new_batch:
            for i, idx in enumerate(batch.indices):
                id = str(int(idx.cpu()))
                self.trajectory[id] = trajectory[i]
        return self.export(trajectory, initial_pose)

    def initialize(self, rotation, translation):
        self.initialized = True
        self.rotation = nn.Parameter(rotation.to(rotation.device))
        t = torch.zeros(translation.shape[0], 3, 3)
        t[:, :, -1] = 0.1
        t[:, :, 0] = translation
        self.translation = nn.Parameter(t.to(rotation.device))

    def explicit_pose(self):
        translation = self.translation[..., 0] + self.translation[..., 1] / torch.abs(self.translation[..., 2])
        return Rt_to_matrix4x4(quaternion_to_rotation_matrix(self.rotation), translation[..., None])

    def export(self, trajectory, poses):
        extrinsics = torch.stack([trajectory @ p.inverse() for p in poses])
        return extrinsics, trajectory
