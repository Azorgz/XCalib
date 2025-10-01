from dataclasses import dataclass
from typing import Literal

import torch
from ImagesCameras.Metrics import NEC
from jaxtyping import Float
from kornia.filters import gaussian_blur2d
from torch import Tensor
from torch_similarity.modules import GradientCorrelationLoss2d

from .loss import Loss, LossCfgCommon
from misc.Mytypes import Batch


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
    losses: dict


losses_dict = {'NEC': NEC}


class LossPose(Loss[LossPoseCfg]):
    def __init__(self, cfg: LossPoseCfg, targets: int, *args) -> None:
        super().__init__(cfg, targets)
        self.losses = [losses_dict[k] for k in cfg.losses]
        self.weights = [cfg.losses[k]['weight'] for k in cfg.losses]

    def compute_unweighted_loss(
            self,
            batch: Batch,
            global_step: int,
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
                mask = gaussian_blur2d(((reg * target) > 0) * 1., 3, (1.6, 1.6)) == 1
                loss = metric(reg, target, mask=mask)
                loss_tot += (1 - loss if metric.higher_is_better else loss) * weight
        return loss_tot
