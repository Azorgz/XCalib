from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float
from kornia.geometry import rotation_matrix_to_quaternion
from torch import Tensor

from . import ExtrinsicsRegressed, ExtrinsicsRegressedCfg
from ..relative_pose.relative_pose import RelativePoseOutput
from ...dataset.types import Batch
from ..projection import align_surfaces_joint, align_surfaces_joint_keypoints
from .extrinsics import Extrinsics
from ...misc.cropping import resize_flow


@dataclass
class ExtrinsicsProcrustesCfg:
    name: Literal["procrustes"]
    num_points: int | None
    randomize_points: bool
    use_flow: bool = False
    regressed: bool = True
    frames: int = 2


class ExtrinsicsProcrustes(Extrinsics[ExtrinsicsProcrustesCfg]):

    def __init__(self, cfg, frames: int):
        super().__init__(cfg, frames)
        if self.cfg.use_flow:
            self.align_fn = align_surfaces_joint
        else:
            self.align_fn = align_surfaces_joint_keypoints
        if self.cfg.regressed:
            regressedCfg = ExtrinsicsRegressedCfg(name='regressed', frames=cfg.frames)
            self.regressed = ExtrinsicsRegressed(regressedCfg)

    @property
    def trajectory(self):
        if self.cfg.regressed:
            return self.regressed._trajectory
        else:
            return self._trajectory

    def forward(
        self,
        batch: Batch,
        weights: Tensor,
        surfaces: Float[Tensor, "batch frame height width 3"],
        initial_pose: Float[Tensor, '1 4 4'],
        new_batch: bool
    ) -> Float[tuple[Tensor, Tensor], "batch frame 4 4"]:
        if (not self.regressed.initialized) or new_batch:
            device = surfaces.device
            b, f, h, w, _ = surfaces.shape
            if self.cfg.use_flow:
                flows = resize_flow(batch.flows, (h, w))
                # Select the subset of points used for the alignment.
                if self.cfg.num_points is None:
                    indices = torch.arange(h * w, dtype=torch.int64, device=device)
                elif self.cfg.randomize_points:
                    indices = torch.randint(
                        0,
                        h * w,
                        (self.cfg.num_points,),
                        dtype=torch.int64,
                        device=device,
                        )
                else:
                    indices = torch.linspace(
                        0,
                        h * w - 1,
                        self.cfg.num_points,
                        dtype=torch.int64,
                        device=device,
                        )
            else:
                flows = batch.keypoints.crop(self.cfg.num_points)
                indices = None
            # Align the depth maps using a Procrustes fit.
            if initial_pose is None:
                initial_pose = torch.eye(4, dtype=torch.float32, device=device)
            initial_pose = initial_pose.expand((b, 4, 4)).contiguous()
            key = str(int(batch.indices[0].cpu()))
            if key in self.trajectory.keys():
                first_pose = self.trajectory[key]
            else:
                first_pose = torch.eye(4, dtype=torch.float32, device=device)
            trajectory = self.align_fn(
                surfaces,
                flows,
                weights,
                initial_pose,
                first_pose,
                indices=indices,).squeeze(0)  # RIG 2 World
            rotation = rotation_matrix_to_quaternion(trajectory[..., :3, :3])
            translation = trajectory[..., :3, 3]
            self.regressed.initialize(rotation, translation)
        return self.regressed(batch,
                              weights,
                              surfaces,
                              initial_pose,
                              new_batch)

