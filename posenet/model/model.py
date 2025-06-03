from dataclasses import dataclass

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn.functional import interpolate

from .intrinsics.common import focal_lengths_to_intrinsics, focal_lengths_to_scaled_intrinsics
from .relative_pose import get_relative_pose, RelativePoseCfg
from .relative_pose.initial_pose import InitialPose, InitialPoseCfg
from .relative_pose.relative_pose import RelativePoseOutput
from .spatial_transformer.spatial_transformer import SpatialTransformerCfg, SpatialTransformerOutput, SpatialTransformer
from .weights import WeightsNetCfg, get_weightsNet
from ..dataset.dataset_cameras import CameraBundle
from ..dataset.types import Batch
from .extrinsics import ExtrinsicsCfg, get_extrinsics
from .intrinsics import IntrinsicsCfg, get_intrinsics
from .projection import sample_image_grid, unproject, depth_to_3d


@dataclass
class ModelCfg:
    weights: WeightsNetCfg
    extrinsics: ExtrinsicsCfg
    relative_pose: RelativePoseCfg
    initial_pose: InitialPoseCfg
    spatial_transformer: SpatialTransformerCfg
    num_frames: int = 0
    num_cams: int = 0
    image_shape: list[int, int] | None = None


@dataclass
class ModelOutput:
    depths: Float[Tensor, "batch frame height width"]
    surfaces: Float[Tensor, "batch frame height width xyz=3"]
    intrinsics: Float[Tensor, "batch frame 3 3"]
    extrinsics: Float[Tensor, "batch frame 4 4"]
    backward_correspondence_weights: Float[Tensor, "batch frame-1 height width"]
    relative_pose: RelativePoseOutput
    projection: SpatialTransformerOutput
    poses: Float[Tensor, "batch 4 4"]
    trajectory: Float[Tensor, " frame xyz"]
    full_trajectory: dict
    # images: Float[Tensor, "batch frame channel height width"]


@dataclass
class ModelExports:
    extrinsics: Float[Tensor, "batch frame 4 4"]
    intrinsics: Float[Tensor, "batch frame 3 3"]
    depths: Float[Tensor, "batch frame 3 3"]
    colors: Float[Tensor, "batch frame 3 height width"]
    surfaces: Float[Tensor, "batch frame height width xyz=3"]
    relative_pose: RelativePoseOutput
    poses: Float[Tensor, "batch 4 4"]
    trajectory: Float[Tensor, " frame xyz"]
    full_trajectory: dict


class Model(nn.Module):
    def __init__(
            self,
            cfg: ModelCfg,
            cameras: CameraBundle,
            sequential: bool = True
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.sequential = sequential
        cfg.weights.image_shape = self.cfg.image_shape
        if sequential:
            # self.backbone = get_weightsNet(cfg.weights, cfg.num_frames, cfg.num_cams)
            self.extrinsics = get_extrinsics(cfg.extrinsics, cfg.num_frames)
        # else:
        #     self.backbone = HourglassRegresser(3, 128, act_fn=nn.ReLU())
        self.relative_pose = get_relative_pose(cfg.relative_pose, cfg.num_cams, cameras)
        self.spatial_transformer = SpatialTransformer(cfg.spatial_transformer, self.cfg.image_shape)
        self.initial_pose = InitialPose(cfg.initial_pose)

    def to(self, device: torch.device):
        super().to(device)
        if self.sequential:
            # self.backbone.to(device)
            self.extrinsics.to(device)
        self.relative_pose.to(device)
        self.initial_pose.to(device)

    def forward(
            self,
            batch: Batch,
            global_step: int,
            new_batch: bool) -> ModelOutput:
        b, f, c, h, w = batch.videos.shape
        # Run the backbone, which provides depths and correspondence weights.

        # Compute the relative pose.
        relative_pose = self.relative_pose(batch, global_step)
        poses = self.initial_pose(relative_pose.Rt)
        if self.sequential:
            depths = interpolate(batch.depths, (h, w))
            weights = 0.5 / (torch.abs(depths[:, 1:] - depths[:, :-1]) + 0.5)
            depths = depths.clamp(0, 100)
            intrinsics = focal_lengths_to_scaled_intrinsics(relative_pose.focal_length,
                                                            (h, w),
                                                            center=relative_pose.center,
                                                            shear=relative_pose.shear).squeeze()[:, None].repeat(1, f, 1, 1)
            surfaces = depth_to_3d(depths, intrinsics)
            # Finally, compute the extrinsics.
            extrinsics, trajectory = self.extrinsics.forward(batch, weights, surfaces, poses, new_batch)
            full_trajectory = self.extrinsics.trajectory
        else:
            extrinsics, trajectory, surfaces, intrinsics, weights, full_trajectory =None, None, None, None, None, {}

        projection = self.spatial_transformer(batch,
                                              relative_pose,
                                              poses,
                                              global_step,
                                              extrinsics,
                                              surfaces,
                                              intrinsics)

        return ModelOutput(
            batch.depths,
            surfaces,
            intrinsics,
            extrinsics,
            weights,
            relative_pose,
            projection,
            poses,
            trajectory,
            full_trajectory)

    @torch.no_grad()
    def export(
            self,
            batch: Batch,
            global_step: int) -> ModelExports:
        output = self.forward(batch, global_step, False)

        return ModelExports(
            output.extrinsics,
            output.intrinsics,
            output.depths,
            batch.videos,
            output.surfaces,
            output.relative_pose,
            output.poses,
            output.trajectory,
            output.full_trajectory)
