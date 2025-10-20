from dataclasses import dataclass
from typing import Literal

import torch
from ImagesCameras import ImageTensor
from ImagesCameras.Metrics import NEC, SSIM, MSE, RMSE, NCC, PSNR, GradientCorrelation
from jaxtyping import Float
from kornia import create_meshgrid
from torch import Tensor
from torch_similarity.modules import GradientCorrelationLoss2d

from misc.Mytypes import Batch
from .loss import Loss, LossCfgCommon


@dataclass
class LossImageCfg(LossCfgCommon):
    name: Literal["image"]
    losses: dict


losses_dict = {'NEC': NEC, 'GC': GradientCorrelation, 'SSIM': SSIM,
               'RMSE': RMSE, 'NCC': NCC, 'PSNR': PSNR, 'MSE': MSE}


class LossImage(Loss[LossImageCfg]):
    def __init__(self, cfg: LossImageCfg, targets: int, *args) -> None:
        super().__init__(cfg, targets)
        self.losses = [losses_dict[k] for k in cfg.losses]
        self.weights = [cfg.losses[k]['weight'] for k in cfg.losses]

    def compute_unweighted_loss(
            self,
            batch: Batch,
            global_step: int,
            cameras,
    ) -> Float[Tensor, ""]:
        losses = 0
        list_idx = [(self.targets, i) for i in range(len(batch.images)) if i != self.targets]
        for idx in list_idx:
            to_, from_ = idx
            target, reg = batch.projections[to_], batch.projections[from_]
            loss = self.compute_loss(target, reg)
            losses += loss.mean()
        return losses

    def compute_loss(self, target: Tensor, reg: Tensor):
        device = target.device
        loss_tot = 0.
        for weight, loss in zip(self.weights, self.losses):
            if weight > 0:
                metric = loss(device=device)
                xy = create_meshgrid(*target.shape[-2:], device=device)
                weights = (xy[:, :, :, 0] ** 2 + xy[:, :, :, 1] ** 2 + 1) / 2
                if isinstance(metric, NEC):
                    loss, coeff = metric(target, reg, mask=reg * target > 0, return_coeff=True, weights=weights)
                    coeff = coeff / coeff.mean()
                else:
                    loss = metric(target, reg, mask=reg * target > 0, weights=weights)
                    coeff = 1.
                if metric.higher_is_better:
                    if metric.range_max == 1:
                        loss = (1 - loss) * weight * coeff
                    else:
                        loss = weight * coeff / (loss + 1e-8)
                else:
                    loss = loss * weight * coeff
                loss_tot += loss
        return loss_tot
