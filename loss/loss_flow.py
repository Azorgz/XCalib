from dataclasses import dataclass
from typing import Literal
import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn.functional import interpolate

from .loss import Loss, LossCfgCommon
from misc.Mytypes import Batch
from flow.CrossModalFlow.get_model import get_model_raft


@dataclass
class LossFlowCfg(LossCfgCommon):
    name: Literal["flow"]


class LossFlow(Loss[LossFlowCfg]):
    def __init__(self, cfg: LossFlowCfg, targets: int, *args) -> None:
        super().__init__(cfg, targets)
        self.FlowModel = get_model_raft()

    def compute_unweighted_loss(
            self,
            batch: Batch,
            global_step: int,
    ) -> Float[Tensor, ""]:
        loss = 0
        list_idx = [(self.targets, i) for i in range(len(batch.cameras)) if i != self.targets]
        for i, idx in enumerate(list_idx):
            to_, from_ = idx
            same_modality = batch.modality[to_] == batch.modality[from_]
            img1, img2 = (interpolate(batch.projections[to_], (360, 360)),
                          interpolate(batch.projections[from_], (360, 360)))
            flow = self.FlowModel(img1, img2)['flow'] if same_modality \
                else (self.FlowModel(img1, img2)['flow'] if batch.modality[to_] == 'IR' else self.FlowModel(img2, img1)['flow'])
            loss += torch.sqrt((flow[:, 0]**2 + flow[:, 1]**2).sum() + 1e-6)
        return loss
