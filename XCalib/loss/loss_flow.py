from dataclasses import dataclass
from typing import Literal
import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn.functional import interpolate

from ..dataset.types import Batch
from ..model.model import ModelOutput
from .loss import Loss, LossCfgCommon
from ..third_party.CrossModalFlow.get_model import get_model


@dataclass
class LossFlowCfg(LossCfgCommon):
    name: Literal["flow"]


class LossFlow(Loss[LossFlowCfg]):
    def __init__(self, cfg: LossFlowCfg) -> None:
        super().__init__(cfg)
        self.crossraft = get_model()

    def compute_unweighted_loss(
            self,
            batch: Batch,
            model_output: ModelOutput,
            global_step: int,
    ) -> Float[Tensor, ""]:
        b, f, _, h, w = batch.videos.shape
        device = batch.videos.device
        self.crossraft = self.crossraft.to(device)
        loss = 0
        list_idx = [(0, i + 1) for i in range(b - 1)]
        for i, idx in enumerate(list_idx):
            to_, from_ = idx
            same_modality = batch.modality[to_] == batch.modality[from_]
            from_ = from_-1 if model_output.projection.images.shape[0] <= b else (from_-1) * 2
            img1, img2 = (interpolate(batch.videos[to_], scale_factor=0.5),
                          interpolate(model_output.projection.images[from_], scale_factor=0.5))
            flow = self.crossraft(img1, img2)['flow'] if same_modality \
                else (self.crossraft(img1, img2)['flow'] if batch.modality[to_] == 'IR' else self.crossraft(img2, img1)['flow'])
            loss += torch.sqrt((flow[:, 0]**2 + flow[:, 1]**2).sum() + 1e-6)
        return loss
