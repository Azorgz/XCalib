from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor
from misc.Mytypes import Batch
from .loss import Loss, LossCfgCommon


@dataclass
class LossRegCfg(LossCfgCommon):
    name: Literal["reg"]
    weight: float


class LossReg(Loss[LossRegCfg]):
    def __init__(self, cfg: LossRegCfg, targets: int, *args) -> None:
        super().__init__(cfg, targets)
        self.weights = cfg.weight
        self.criterion_translation = torch.nn.HuberLoss(delta=0.1, reduction='mean')
        self.criterion_rotation = torch.nn.HuberLoss(delta=0.05, reduction='mean')

    def compute_unweighted_loss(
            self,
            batch: Batch,
            global_step: int,
            cameras,
    ) -> Float[Tensor, ""]:
        relative_translation_loss = [
            self.criterion_translation(cameras.cameras[0].translation_vector, cameras.cameras[i].translation_vector)
            for i in range(1, len(cameras.cameras))]
        relative_rotation_loss = [
            self.criterion_rotation(cameras.cameras[0].rotation_quaternion, cameras.cameras[i].rotation_quaternion)
            for i in range(1, len(cameras.cameras))]
        losses = sum(relative_translation_loss) + sum(relative_rotation_loss)*2
        return losses