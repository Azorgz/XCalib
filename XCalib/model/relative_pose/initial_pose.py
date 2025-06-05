from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange
from jaxtyping import Float
from kornia.geometry import rotation_matrix_to_quaternion
from torch import Tensor, nn


# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def quaternion_to_matrix(
        quaternions: Float[Tensor, "*batch 4"],
        eps: float = 1e-8,
) -> Float[Tensor, "*batch 3 3"]:
    # Order changed to match scipy format!
    r, i, j, k = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)


@dataclass
class InitialPoseCfg:
    freeze_initial_pose: bool
    freeze_axis: dict


class InitialPose(nn.Module):
    def __init__(self, cfg: InitialPoseCfg,
                 initial_pose_ref: Float[Tensor, 'batch 4 4'] | None = None,
                 *args, **kwargs) -> None:
        # Initialize identity translations and rotations.
        super().__init__(*args, **kwargs)
        self.freeze_axis = cfg.freeze_axis
        self.freeze_initial_pose = cfg.freeze_initial_pose
        if initial_pose_ref is not None:
            rotations = rotation_matrix_to_quaternion(initial_pose_ref[:, :3, :3])
            self.translations = nn.Parameter(initial_pose_ref[:, :3, -1], requires_grad=not self.freeze_initial_pose)
        else:
            self.translations = nn.Parameter(torch.zeros((1, 3), dtype=torch.float32), requires_grad=not self.freeze_initial_pose)
            rotations = torch.zeros((1, 4), dtype=torch.float32)
            rotations[0, 0] = 1
        self.rotations = nn.Parameter(rotations, requires_grad=not self.freeze_initial_pose)

    def forward(
            self,
            relative_pose: Float[Tensor, "batch 4  4"],
    ) -> Float[Tensor, "batch 4 4"]:
        device = relative_pose.device
        if not self.freeze_initial_pose:
            rotations = torch.zeros_like(self.rotations)
            rotations[:, 0] = float(not self.freeze_axis['rx']) * self.rotations[:, 0]
            rotations[:, 1] = float(not self.freeze_axis['ry']) * self.rotations[:, 1]
            rotations[:, 2] = float(not self.freeze_axis['rz']) * self.rotations[:, 2]

            translations = torch.zeros_like(self.translations)
            translations[:, 0] = float(not self.freeze_axis['x']) * self.translations[:, 0]
            translations[:, 1] = float(not self.freeze_axis['y']) * self.translations[:, 1]
            translations[:, 2] = float(not self.freeze_axis['z']) * self.translations[:, 2]
        else:
            rotations = self.rotations
            translations = self.translations

        ref_pose_in_world = torch.eye(4, dtype=torch.float32, device=device)[None]
        ref_pose_in_world[:, :3, :3] = quaternion_to_matrix(rotations)
        ref_pose_in_world[:, :3, 3] = translations
        target_pose_in_world = relative_pose @ ref_pose_in_world

        return torch.cat([ref_pose_in_world, target_pose_in_world], dim=0)
