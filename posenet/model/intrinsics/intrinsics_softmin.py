from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from jaxtyping import Float
from torch import Tensor

from ThirdParty.ImagesCameras import ImageTensor
from .common import focal_lengths_to_intrinsics, focal_lengths_to_scaled_intrinsics
from .intrinsics import Intrinsics
from .intrinsics_regressed import IntrinsicsRegressed, IntrinsicsRegressedCfg
from ..projection import (
    align_surfaces,
    compute_backward_flow,
    sample_image_grid,
    unproject,
)
from ..relative_pose.relative_pose import RelativePoseOutput
from ...dataset.types import Batch
from ...misc.cropping import resize_flow


@dataclass
class RegressionCfg:
    after_step: int
    window: int


@dataclass
class IntrinsicsSoftminCfg:
    name: Literal["softmin"]
    num_procrustes_points: int
    image_loss_weight: float
    flow_loss_weight: float
    min_focal_length: float
    max_focal_length: float
    num_candidates: int
    regression: RegressionCfg | None
    image_shape: list[int] | None = None


class IntrinsicsSoftmin(Intrinsics[IntrinsicsSoftminCfg]):
    focal_length_candidates: Float[Tensor, " candidate"]

    def __init__(self, cfg: IntrinsicsSoftminCfg, num_cams: int) -> None:
        super().__init__(cfg, num_cams)
        self.loss_image = cfg.image_loss_weight / (cfg.image_loss_weight + cfg.flow_loss_weight)
        self.loss_flow = cfg.flow_loss_weight / (cfg.image_loss_weight + cfg.flow_loss_weight)
        # Define the set of candidate focal lengths.
        focal_length_candidates = torch.linspace(
            cfg.min_focal_length,
            cfg.max_focal_length,
            cfg.num_candidates,
        )
        self.register_buffer(
            "focal_length_candidates",
            focal_length_candidates.clone(),
            persistent=False,
        )

        if cfg.regression is not None:
            self.window = []
            intrinsics_regressed_cfg = IntrinsicsRegressedCfg("regressed", 0.0)
            self.intrinsics_regressed = IntrinsicsRegressed(intrinsics_regressed_cfg, num_cams)

    def forward(
            self,
            batch: Batch,
            weights: Tensor,
            relative_pose: RelativePoseOutput,
            global_step: int,
            initial_pose: Float[Tensor, 'batch 4 4']
    ) -> Tensor:
        b, f, h, w = batch.depths.shape
        h, w = self.cfg.image_shape or (h, w)
        flows = resize_flow(batch.flows, (h, w))
        depths = F.interpolate(batch.depths, (h, w))
        n = self.cfg.num_candidates
        device = batch.videos.device

        # Handle the second stage (in which the intrinsics are regressed).
        if (
                self.cfg.regression is not None
                and global_step >= self.cfg.regression.after_step
        ):
            if global_step == self.cfg.regression.after_step:
                initial_value = torch.stack(self.window).mean(dim=0)
                self.intrinsics_regressed.focal_length.data = initial_value
            res = self.intrinsics_regressed(batch, weights, relative_pose, global_step, initial_pose)
            self.last_value = self.intrinsics_regressed.last_value
            return res

        # Convert the candidate focal lengths into 3x3 intrinsics matrices.
        candidate_intrinsics = focal_lengths_to_intrinsics(self.focal_length_candidates, (h, w)).unsqueeze(0)
        # if self.dual_channel:
        candidate_intrinsics2 = focal_lengths_to_intrinsics(
            self.focal_length_candidates / relative_pose.scale.squeeze(),
            (h, w)).unsqueeze(0)
        candidate_intrinsics = torch.cat([candidate_intrinsics, candidate_intrinsics2], dim=0)
        # Align the first two frames with all possible intrinsics.
        indices = torch.randperm(h * w, device=device)[: self.cfg.num_procrustes_points]
        xy, _ = sample_image_grid((h, w), device=device)
        f_ = torch.randint(0, f - 1, (1, 1), device=device).squeeze()
        surfaces = unproject(
            xy,
            repeat(depths[:, f_:f_ + 2], "b f h w -> (b n) f h w", n=n),
            repeat(candidate_intrinsics, "b n i j -> (b n) f () () i j", f=2),
        )
        # surfaces = depth_to_3d(repeat(depths[:, f_:f_+2], "b f h w -> b n f h w", n=n),
        #                               repeat(candidate_intrinsics, "b n i j -> b n f i j", f=2))
        # extrinsics = rearrange(align_surfaces(
        #     surfaces,
        #     repeat(flows.backward[:, f_:f_+1], "b f h w xy -> (b n) f h w xy", n=n),
        #     repeat(weights[:, f_:f_+1], "b f h w -> (b n) f h w", n=n),
        #     indices,
        #     initial_pose=initial_pose[:, None]), "b n f h w -> (b n) f h w")
        extrinsics = align_surfaces(
            rearrange(surfaces, "(b n) f h w xyz -> b n f h w xyz", b=b, n=n),
            repeat(flows.backward[:, f_:f_ + 1], "b f h w xy -> (b n) f h w xy", n=n),
            repeat(weights[:, f_:f_ + 1], "b f h w -> (b n) f h w", n=n),
            indices,
            initial_pose=initial_pose[:, None])

        # Compute pose-induced backward flow.
        xy_flowed_backward = compute_backward_flow(
            rearrange(surfaces, "bn f h w xyz -> bn f (h w) xyz")[:, :, indices],
            rearrange(extrinsics, "b n f h w -> (b n) f h w"),
            repeat(candidate_intrinsics, "b n i j -> (b n) f i j", f=2),
        )
        # xy_flowed_backward = compute_backward_flow(
        #     rearrange(surfaces, "b n f h w xyz -> (b n) f (h w) xyz")[:, :, indices],
        #     extrinsics,
        #     repeat(candidate_intrinsics, "b n i j -> (b n) f i j", f=2),
        #     (h, w)
        # )
        # xy_flowed_backward = rearrange(
        #     xy_flowed_backward, "(b n) p xy -> b n p xy", b=b, n=n)
        xy_flowed_backward = rearrange(
            xy_flowed_backward, "(b n) () p xy -> b n p xy", b=b, n=n)
        xy, _ = sample_image_grid((h, w), device)
        xy = rearrange(xy, "h w xy -> (h w) xy")[indices]
        flow = xy_flowed_backward - xy

        # Weights definition
        weight_select = rearrange(weights[:, f_:f_ + 1], "b f h w -> b f (h w) ()")
        weight_select = weight_select[:, :, indices]

        # Compute a loss based on image alignment
        scaled_candidate_intrinsics1 = focal_lengths_to_scaled_intrinsics(self.focal_length_candidates,
                                                                          (480, 640)).unsqueeze(0)
        scaled_candidate_intrinsics2 = focal_lengths_to_scaled_intrinsics(
            self.focal_length_candidates / relative_pose.scale.squeeze(), (480, 640)).unsqueeze(0)
        scaled_candidate_intrinsics = torch.cat([scaled_candidate_intrinsics1, scaled_candidate_intrinsics2], dim=0)
        gt1 = ImageTensor(batch.videos[0][f_])
        gt2 = ImageTensor(batch.videos[1][f_])
        images_1, images_2 = projection_frame_to_frame_sequential(batch,
                                                         (extrinsics[:, :, 1].inverse() @ extrinsics[:, :, 0]),
                                                         scaled_candidate_intrinsics,
                                                         idx=f_,
                                                         size=(480, 640))
        loss_image = torch.abs((rearrange(images_1, "b c h w -> b c (h w)")[:, :, indices] -
                                rearrange(gt1, "b c h w -> b c (h w)")[:, :, indices]) * weight_select[0, 0, :,
                                                                                         0]).mean(-1).mean(-1)
        loss_image += torch.abs((rearrange(images_2, "b c h w -> b c (h w)")[:, :, indices] -
                                 rearrange(gt2, "b c h w -> b c (h w)")[:, :, indices]) * weight_select[1, 0, :,
                                                                                          0]).mean(-1).mean(-1)
        flow_gt = rearrange(flows.backward[:, f_:f_ + 1], "b f h w xy -> b f (h w) xy")
        flow_gt = flow_gt[:, :, indices]

        # Compute flow error for each of the candidate intrinsics.
        error_flow = ((flow - flow_gt) * weight_select).abs()
        error_flow = reduce(error_flow, "b n p xy ->() n", "sum")

        # Compute a softmin-weighted sum of candidates.
        weights_flow = (error_flow - error_flow.min(dim=1, keepdim=True).values)
        loss_image = loss_image[None]
        weights_images = F.softmin((loss_image - loss_image.min(dim=1, keepdim=True).values) * 10, dim=1) * self.loss_image
        weights_flow = F.softmin(weights_flow, dim=1) * self.loss_flow
        focal_lengths = (self.focal_length_candidates * weights_flow +
                         self.focal_length_candidates * weights_images).sum(dim=1)
        # focal_lengths_weights = (self.focal_length_candidates * weights_images).sum(dim=1)

        # intrinsics = (candidate_intrinsics * weights_flow[:, :, None, None]
        #               .repeat(candidate_intrinsics.shape[0], 1, 1, 1)).sum(dim=1)
        self.last_value = focal_lengths, focal_lengths / relative_pose.scale.squeeze()

        # Handle the window that's used to initialize the intrinsics.

        if self.cfg.regression is not None:
            start = self.cfg.regression.after_step - self.cfg.regression.window
            if global_step >= start and self.training:
                self.window.append(focal_lengths)
        return torch.stack(self.last_value)

    def unnormalized_focal_lengths(
            self,
            image_shape: tuple[int, int],
    ) -> Float[Tensor, " candidate"]:
        focal_lengths = self.focal_length_candidates
        # Unnormalize the focal lengths based on the geometric mean of the image's side
        # lengths. This makes the candidate focal lengths invariant to the image's
        # aspect ratio.
        h, w = image_shape
        focal_lengths = focal_lengths * (h * w) ** 0.5

        return focal_lengths
