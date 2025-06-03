from dataclasses import dataclass
from typing import Literal

from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from ThirdParty.ImagesCameras import ImageTensor
from ThirdParty.ImagesCameras.Metrics import Metric_nec_tensor, Metric_ssim_tensor
from ..dataset.types import Batch
from ..model.model import ModelOutput
from .loss import Loss, LossCfgCommon
from .mapping import MappingCfg, get_mapping


@dataclass
class LossFlowCfg(LossCfgCommon):
    name: Literal["flow"]
    mapping: MappingCfg
    coeff: list


class LossFlow(Loss[LossFlowCfg]):
    def __init__(self, cfg: LossFlowCfg) -> None:
        super().__init__(cfg)
        self.mapping = get_mapping(cfg.mapping)
        self.coeff_nec = abs(cfg.coeff[0]) / (abs(cfg.coeff[0]) + abs(cfg.coeff[1]))
        self.coeff_ssim = abs(cfg.coeff[1]) / (abs(cfg.coeff[0]) + abs(cfg.coeff[1]))

    def compute_unweighted_loss(
            self,
            batch: Batch,
            model_output: ModelOutput,
            global_step: int,
    ) -> Float[Tensor, ""]:
        b, f, _, h, w = batch.videos.shape
        loss, coeff = self.projection(batch, model_output)
        tot = coeff.sum()
        losses = loss * coeff
        return losses.sum() / tot

    def projection(self, batch, model_output) -> tuple[Tensor, Tensor]:
        # The grid search algorithm tends to focus on x translation as it's meant for rig mounted on car
        device = batch.videos.device
        b, f, *_ = batch.videos.shape
        if self.coeff_nec > 0:
            nec = Metric_nec_tensor(device=device)
        if self.coeff_ssim > 0:
            ssim = Metric_ssim_tensor(device=device)

        flows = rearrange(model_output.projection.flows, 'b f c h w -> (b f) c h w')
        flows = ImageTensor(flows, batched=flows.shape[0] > 1)
        images = rearrange(batch.videos[:, 1:], 'b f c h w -> (b f) c h w')
        images = ImageTensor(images, batched=images.shape[0] > 1)
        weights = rearrange(model_output.backward_correspondence_weights, 'b f h w -> (b f) h w')  # self.cfg.depth_max - depths
        loss_nec, coeff = nec(images, flows, mask=images * flows > 0, weights=weights,
                              return_coeff=True) if self.coeff_nec > 0 else (0, 1)
        loss_ssim = ssim(images, flows, mask=images * flows > 0, weights=weights) if self.coeff_ssim > 0 else 0
        loss = (1 - loss_nec) * self.cfg.coeff[0] + (1 - loss_ssim) * self.cfg.coeff[1]
        return loss, coeff
