import itertools
from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange
from jaxtyping import Float
from kornia import create_meshgrid
from torch import Tensor
from torch_similarity.modules import GradientCorrelationLoss2d

from ImagesCameras import ImageTensor
from ImagesCameras.Metrics import NEC
from .loss import Loss, LossCfgCommon
from ..dataset.types import Batch
from ..model.model import ModelOutput


class GC(GradientCorrelationLoss2d):

    def __init__(self):
        super().__init__(return_map=True)

    def forward(self, x, y, mask=None, weights=None, return_coeff=False):
        _, gc_map = super().forward(x, y)
        b, c, h, w = x.shape

        if weights is not None:
            gc_map_ = gc_map * weights
        else:
            gc_map_ = gc_map
        if mask is not None:
            gc_map_ *= mask
        if return_coeff:
            return torch.abs(gc_map_.flatten(1, -1).sum(-1)), torch.abs(gc_map.flatten(1, -1).sum(-1))
        else:
            return torch.abs(gc_map_.flatten(1, -1).sum(-1))


@dataclass
class LossPoseCfg(LossCfgCommon):
    name: Literal["pose"]
    bidirectional: bool
    depth_max: int = 100
    cross_loss: int = 0
    ponderate: bool = False
    coeff: tuple[float, float, float] = (0.9, 0.1, 0.)
    from_file: bool = True


class LossPose(Loss[LossPoseCfg]):
    def __init__(self, cfg: LossPoseCfg) -> None:
        super().__init__(cfg)
        coeff = cfg.coeff
        sum = abs(coeff[0]) + abs(coeff[1])
        self.coeff_nec = torch.tensor(abs(coeff[0]) / sum)
        self.coeff_ssim = torch.tensor(abs(coeff[1]) / sum)
        self.coeff_depth = torch.tensor(coeff[2]).clamp(0, 1)

    def compute_unweighted_loss(
            self,
            batch: Batch,
            model_output: ModelOutput,
            global_step: int,
    ) -> Float[Tensor, ""]:
        b, f, _, h, w = batch.videos.shape
        losses = 0
        tot = 0
        if self.cfg.cross_loss and self.cfg.cross_loss <= global_step:
            list_idx = itertools.combinations([i for i in range(b)], 2)
        else:
            list_idx = [(0, i + 1) for i in range(b - 1)]
        for i, idx in enumerate(list_idx):
            to_, from_ = idx
            loss, coeff = self.projection(batch, model_output, to_, from_, i)
            if self.cfg.ponderate and self.coeff_nec:
                coeff = coeff.mean() if 0 in [to_, from_] else coeff.mean() / 2
                tot += coeff
                losses += (loss * coeff).mean()
            else:
                coeff = 1 if 0 in [to_, from_] else 1 / 2
                tot += coeff
                losses += loss.mean() * coeff
        return losses / tot

    def projection(self, batch, model_output, to_: int, from_: int, i: int):
        # The grid search algorithm tends to focus on x translation as it's meant for rig mounted on car
        device = batch.videos.device
        b, f, *_ = batch.videos.shape
        if self.coeff_nec > 0:
            nec = NEC(device=device)
        if self.coeff_ssim > 0 or self.coeff_depth > 0:
            ssim = GC().to(device)

        # INTRINSICS batch for scaling
        # size = batch.videos.shape[-2:]
        # if self.cfg.from_file:
        # intrinsics1 = focal_lengths_to_scaled_intrinsics(model_output.relative_pose.focal_length[to_],
        #                                                  batch.image_sizes[batch.cameras[to_].name]).squeeze()
        # intrinsics2 = focal_lengths_to_scaled_intrinsics(model_output.relative_pose.focal_length[from_],
        #                                                  batch.image_sizes[batch.cameras[from_].name]).squeeze()
        # # else:
        # #     intrinsics1 = focal_lengths_to_scaled_intrinsics(model_output.focal_lengths[to_], size).squeeze()
        # #     intrinsics2 = focal_lengths_to_scaled_intrinsics(model_output.focal_lengths[from_], size).squeeze()
        # intrinsics = torch.cat([intrinsics1[None, None], intrinsics2[None, None]], dim=0)
        # # if to_ != 0:
        # relative_pose = (model_output.initial_pose[to_].inverse() @ model_output.initial_pose[from_])[None]
        # # else:
        # #     relative_pose = model_output.relative_pose.Rt[from_-1][None]
        # images, idx = projection_frame_to_frame(batch,
        #                                         relative_pose,
        #                                         intrinsics,
        #                                         return_depths=False,
        #                                         bidirectional=False,
        #                                         idx='all',
        #                                         from_=from_, to_=to_, from_file=self.cfg.from_file)
        # Projected images preparation
        if self.cfg.bidirectional:
            images = model_output.projection.images[i * 2: (i + 1) * 2]
            if self.coeff_depth > 0:
                depths = 1 / (rearrange(model_output.projection.depths[i * 2: (i + 1) * 2],
                                        'b f () h w -> (b f) () h w') + 10)
                depths = ImageTensor(depths)
        else:
            images = model_output.projection.images[i:i + 1]
            if self.coeff_depth > 0:
                depths = 1 / (rearrange(model_output.projection.depths[i: i + 1],
                                        'b f () h w -> (b f) () h w') + 10)
                depths = ImageTensor(depths)
        if images.shape[2] == 1:
            images = images.repeat(1, 1, 3, 1, 1)
        images = ImageTensor(images, batched=images.shape[0] * images.shape[1] > 1)

        # Ref images preparation
        if self.cfg.from_file:
            gt = ImageTensor.batch(*ImageTensor(batch.frame_paths[to_])).to(images.device)  # 1 c h w
            if self.cfg.bidirectional:
                gt0 = ImageTensor.batch(*ImageTensor(batch.frame_paths[from_])).to(images.device)  # 1 c h w
                gt = gt0.batch(gt)
        else:
            gt = ImageTensor(batch.videos[to_], batched=batch.videos.shape[1]).resize(images.shape[-2:])  # 1 c h w
            if self.cfg.bidirectional:
                gt0 = ImageTensor(batch.videos[from_], batched=batch.videos.shape[1]).resize(
                    images.shape[-2:])  # 1 c h w
                gt = gt0.batch(gt)
        if self.coeff_depth > 0:
            gt_depth = ImageTensor(batch.depths[to_], batched=batch.depths.shape[1], normalize=False).resize(
                depths.shape[-2:])  # 1 c h w
            if self.cfg.bidirectional:
                gt0_depth = ImageTensor(batch.depths[from_], batched=batch.depths.shape[1], normalize=False).resize(
                    depths.shape[-2:])  # 1 c h w
                gt_depth = ImageTensor(1 / (gt0_depth.batch(gt_depth) + 10), batched=True)
            else:
                gt_depth = ImageTensor(1 / (gt_depth + 10), batched=gt_depth.shape[0])
        xy = create_meshgrid(*images.image_size, device=device)
        weights = (xy[:, :, :, 0] ** 2 + xy[:, :, :, 1] ** 2 + 1) / 2  # self.cfg.depth_max - depths

        if self.coeff_depth < 1:
            loss_nec, coeff = nec(images, gt, mask=images * gt > 0, weights=weights, return_coeff=True) if self.coeff_nec > 0 else (0, 1)
            loss_ssim = ssim(images, gt, mask=images * gt > 0, weights=weights) if self.coeff_ssim > 0 else 0
            loss_image = (1 - loss_nec) * self.coeff_nec + (1 - loss_ssim) * self.coeff_ssim
        else:
            loss_image = 0

        if self.coeff_depth > 0:
            loss_depth_nec = nec(depths.normalize(), gt_depth.normalize(), mask=depths * gt_depth > 0, weights=weights) if self.coeff_nec > 0 else 0
            loss_depth_ssim = ssim(depths.normalize(), gt_depth.normalize(), mask=depths * gt_depth > 0, weights=weights) if self.coeff_depth > 0 else 0
            loss_depth = (1 - loss_depth_nec) * self.coeff_nec + (1 - loss_depth_ssim) * self.coeff_ssim
        else:
            loss_depth = 0

        loss = loss_image * (1 - self.coeff_depth) + self.coeff_depth * loss_depth
        return loss, coeff
