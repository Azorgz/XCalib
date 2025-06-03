from dataclasses import dataclass
from typing import Literal

from jaxtyping import Float
from torch import Tensor

from ThirdParty.ImagesCameras import ImageTensor
from ThirdParty.ImagesCameras.Metrics import Metric_rmse_tensor, Metric_ssim_tensor
from .loss import Loss, LossCfgCommon
from ..dataset.types import Batch
from ..model.model import ModelOutput


@dataclass
class LossImageCfg(LossCfgCommon):
    name: Literal["image"]


class LossImage(Loss[LossImageCfg]):
    def __init__(self, cfg: LossImageCfg) -> None:
        super().__init__(cfg)

    def compute_unweighted_loss(
            self,
            batch: Batch,
            model_output: ModelOutput,
            global_step: int,
    ) -> Float[Tensor, ""]:
        device = batch.videos.device
        b, f, _, h, w = batch.videos.shape
        images_ref = ImageTensor(batch.videos[:2])
        image_new = ImageTensor(model_output.images)
        rmse = Metric_rmse_tensor(device)
        ssim = Metric_ssim_tensor(device)
        loss = rmse(images_ref, image_new) + (1 - ssim(images_ref, image_new))
        return loss.mean()
