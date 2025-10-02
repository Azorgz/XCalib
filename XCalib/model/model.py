from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor, nn

from .relative_pose import get_relative_pose, RelativePoseCfg
from .relative_pose.initial_pose import InitialPose, InitialPoseCfg
from .relative_pose.relative_pose import RelativePoseOutput
from XCalib2.model.spatial_transformer import SpatialTransformerCfg, SpatialTransformerOutput, SpatialTransformer
from ..dataset.dataset_cameras import CameraBundle
from ..dataset.types import Batch


@dataclass
class ModelCfg:
    relative_pose: RelativePoseCfg
    initial_pose: InitialPoseCfg
    spatial_transformer: SpatialTransformerCfg
    num_frames: int = 0
    num_cams: int = 0
    image_shape: list[int, int] | None = None


@dataclass
class ModelOutput:
    depths: Float[Tensor, "batch frame height width"]
    relative_pose: RelativePoseOutput
    projection: SpatialTransformerOutput
    poses: Float[Tensor, "batch 4 4"]


@dataclass
class ModelExports:
    depths: Float[Tensor, "batch frame 3 3"]
    colors: Float[Tensor, "batch frame 3 height width"]
    relative_pose: RelativePoseOutput
    poses: Float[Tensor, "batch 4 4"]


class Model(nn.Module):
    def __init__(
            self,
            cfg: ModelCfg,
            cameras: CameraBundle) -> None:
        super().__init__()
        self.cfg = cfg
        self.relative_pose = get_relative_pose(cfg.relative_pose, cfg.num_cams, cameras)
        self.spatial_transformer = SpatialTransformer(cfg.spatial_transformer, self.cfg.image_shape)
        self.initial_pose = InitialPose(cfg.initial_pose)

    def to(self, device: torch.device):
        super().to(device)
        self.relative_pose.to(device)
        self.initial_pose.to(device)

    def forward(
            self,
            batch: Batch,
            global_step: int) -> ModelOutput:

        # Compute the relative pose.
        relative_pose = self.relative_pose(batch, global_step)
        poses = self.initial_pose(relative_pose.Rt)
        projection = self.spatial_transformer(batch,
                                              relative_pose,
                                              poses,
                                              global_step)

        return ModelOutput(
            batch.depths,
            relative_pose,
            projection,
            poses)

    @torch.no_grad()
    def export(
            self,
            batch: Batch,
            global_step: int) -> ModelExports:
        output = self.forward(batch, global_step)

        return ModelExports(
            output.depths,
            batch.videos,
            output.relative_pose,
            output.poses)
