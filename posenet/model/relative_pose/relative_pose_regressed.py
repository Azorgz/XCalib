from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float
from kornia.geometry import Rt_to_matrix4x4, quaternion_to_rotation_matrix
from torch import Tensor, nn

from . import RelativePose
from .relative_pose import RelativePoseOutput
from ...dataset.dataset_cameras import CameraBundle
from ...dataset.types import Batch


@dataclass
class RelativePoseRegressedCfg:
    name: Literal["regressed"]
    parameter_limitation: dict = None
    random_init: float = 0.01
    rotation: Float[list, "b 4"] | None = None
    translation: Float[list, "b 3 3"] | None = None
    focal_length: list | None = None
    freeze_center: bool = True
    freeze_shear: bool = True


class RelativePoseRegressed(RelativePose[RelativePoseRegressedCfg]):
    def __init__(self, cfg: RelativePoseRegressedCfg, num_cams: int, cameras: CameraBundle) -> None:
        super().__init__(cfg, num_cams)
        if self.cfg.parameter_limitation is None:
            self.cfg.parameter_limitation = {'x': None, 'y': None, 'z': None, 'rx': None, 'ry': None, 'rz': None}
        if cfg.rotation is not None:
            assert len(cfg.rotation) == num_cams - 1
            self.rotation = nn.Parameter(torch.tensor(cfg.rotation, dtype=torch.float32))
        else:
            t = torch.tensor([1., 0, 0, 0])[None].repeat(self.num_cams - 1, 1) + (
                        torch.rand([self.num_cams - 1, 4]) - 0.5) * cfg.random_init
            self.rotation = nn.Parameter(t)
        if cfg.translation is not None:
            self.translation = nn.Parameter(torch.tensor(cfg.translation, dtype=torch.float32))
        else:
            t = torch.zeros(1, 3, 3).repeat(self.num_cams - 1, 1, 1) + (
                        torch.rand([self.num_cams - 1, 3, 3]) - 0.5) * cfg.random_init
            t[:, :, 2] = torch.ones(1, 3).repeat(self.num_cams - 1, 1) * 0.1
            self.translation = nn.Parameter(t)
        if cfg.focal_length is not None:
            focal_length = torch.tensor(cfg.focal_length, dtype=torch.float32)[:, None]
        else:
            ref_ratio = cameras[0].height / cameras[0].width
            focal_length = torch.tensor([0.95 * ref_ratio / (c.height / c.width) for c in cameras], dtype=torch.float32)[:, None]
        focal_length += (torch.rand(num_cams)[:, None] - 0.5) * cfg.random_init
        self.focal_length = nn.Parameter(focal_length)
        self.center = nn.Parameter(torch.tensor([[0.5, 0.5]] * num_cams, dtype=torch.float32)) if not self.cfg.freeze_center else None
        self.shear = nn.Parameter(nn.Parameter(torch.tensor([0] * num_cams, dtype=torch.float32)[:, None])) if not self.cfg.freeze_shear else None
        self.rotation_limitation = torch.tensor([1,
                                                 0 if cfg.parameter_limitation['rx'] is not None else 1,
                                                 0 if cfg.parameter_limitation['ry'] is not None else 1,
                                                 0 if cfg.parameter_limitation['rz'] is not None else 1])
        self.init = False

    def initialization(self,
                       rotation: Float[Tensor, "batch-1 4"],
                       translation: Float[Tensor, "batch-1 3"],
                       scale: Float[Tensor, "batch"]):
        self.rotation = nn.Parameter(rotation)
        self.translation = nn.Parameter(translation)
        self.focal_length = nn.Parameter(scale)
        self.init = True

    # def freeze_translation(self):
    #     self.translation.requires_grad = False
    #
    # def unfreeze_translation(self):
    #     self.translation.requires_grad = True
    #
    # def freeze_scale(self):
    #     self.scale.requires_grad = False
    #
    # def unfreeze_scale(self):
    #     self.scale.requires_grad = True
    #
    # def freeze_rotation(self):
    #     self.rotation.requires_grad = False
    #
    # def unfreeze_rotation(self):
    #     self.rotation.requires_grad = True

    def forward(
            self,
            batch: Batch,
            global_step: int,
            params=None
    ) -> RelativePoseOutput:
        # if not self.init:
        #     image_size = [batch.image_sizes[cam.name] for cam in batch.cameras]
        #     ratio = torch.tensor([im_s[1]/im_s[0]*480/640 for im_s in image_size]).to(self.focal_length.device)
        #     self.initialization(self.rotation, self.translation, self.focal_length * ratio[:, None])
        return self.export(params)

    def export(self, params=None):
        focal_length = self.limit(self.focal_length, self.cfg.parameter_limitation['f'])
        center = self.center
        shear = self.shear
        rotation = self.rotation / torch.linalg.vector_norm(self.rotation) * self.rotation_limitation.to(self.rotation.device)
        # rotation_x = self.limit(rotation[:, 0], self.cfg.parameter_limitation['rx'])
        # rotation_y = self.limit(rotation[:, 1], self.cfg.parameter_limitation['ry'])
        # rotation_z = self.limit(rotation[:, 2], self.cfg.parameter_limitation['rz'])
        # rotation = axis_angle_to_quaternion(torch.stack([rotation_x, rotation_y, rotation_z], dim=1))
        translation = self.translation[:, :, 0] + self.translation[:, :, 1] / (torch.abs(self.translation[:, :, 2]) + 1e-6)
        translation_x = self.limit(translation[:, 0], self.cfg.parameter_limitation['x'])
        translation_y = self.limit(translation[:, 1], self.cfg.parameter_limitation['y'])
        translation_z = self.limit(translation[:, 2], self.cfg.parameter_limitation['z'])
        translation = torch.stack([translation_x, translation_y, translation_z], dim=1)
        return RelativePoseOutput(translation, rotation, focal_length, center, shear, self.explicit_pose_(rotation, translation))

    @staticmethod
    def explicit_pose_(rotation, translation):
        return Rt_to_matrix4x4(quaternion_to_rotation_matrix(rotation), translation[..., None])

    @staticmethod
    def limit(param, cfg):
        if cfg is None:
            return param
        elif isinstance(cfg, float) or isinstance(cfg, int):
            return param*0 + cfg
            # return torch.ones_like(param, requires_grad=True, device=param.device)*cfg
        elif isinstance(cfg, list):
            return torch.clamp(param, cfg[0], cfg[1])
        else:
            raise ValueError("cfg should be either float or list")
