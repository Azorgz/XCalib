from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor

from .loss import Loss, LossCfgCommon
from ..dataset.types import Batch
from ..model.model import ModelOutput


@dataclass
class LossFeatureCfg(LossCfgCommon):
    name: Literal["feature"]
    person_avg_height: float  # mean, std for men and women
    car_avg_height: float
    list_classes = ["car", "person"]


class LossFeature(Loss[LossFeatureCfg]):
    def __init__(self, cfg: LossFeatureCfg) -> None:
        super().__init__(cfg)
        self.classes_size = {'car': cfg.car_avg_height, 'person': cfg.person_avg_height}

    def compute_unweighted_loss(
            self,
            batch: Batch,
            model_output: ModelOutput,
            global_step: int,
    ) -> Float[Tensor, ""]:
        b, f, _, h, w = batch.videos.shape
        losses = 0
        if batch.objects is not None:
            losses += self.feature_loss(batch, model_output)
        return losses

    def feature_loss(self, batch, model_output):
        device = batch.videos.device
        b, f, *_ = batch.videos.shape
        # INTRINSICS batch for scaling
        hpix = batch.videos.shape[-2]
        # intrinsics = focal_lengths_to_scaled_intrinsics(model_output.focal_lengths, size)
        estimated_AFOV = []
        tot = 0
        for obj, depths, f in zip(batch.objects,
                                  batch.depths,
                                  model_output.relative_pose.focal_length):  # Iter through cameras
            estimated_afov = []
            weights = []
            for o, depth in zip(obj, depths):  # Iter through frames
                if o is not None:
                    for cls, size_class in self.classes_size.items():
                        features = o[cls]
                        if features is not None:
                            afov, depth_feature = self._compute_angles(features, depth, size_class, hpix)
                            estimated_afov.extend(afov)
                            weights.extend(depth_feature)
            if len(estimated_afov) > 0:
                estimated_afov = torch.stack(estimated_afov)
                weights = torch.stack(weights)
                estimated_AFOV.append(torch.sum(estimated_afov * weights)/torch.sum(weights))
                tot += 1
            else:
                estimated_AFOV.append(torch.tensor(f, device=device).squeeze())
        if tot > 0:
            return torch.sum(torch.abs(torch.stack(estimated_AFOV) - model_output.relative_pose.focal_length))/tot
        else:
            return 0

    @staticmethod
    def _compute_angles(features, depth, size_class, hpix):
        features_tensor = features.to(depth.device)
        depths_feature = torch.stack([torch.median(depth[f[1]:f[3], f[0]:f[2]]) for f in features_tensor.to(torch.int64)])
        relative_size_feature = torch.abs(features_tensor[:, 1] - features_tensor[:, 3]) / hpix
        FOV = size_class / relative_size_feature
        ref_angle = torch.deg2rad(torch.tensor([33.75], device=depth.device))
        relative_aFOV = 2 * torch.arctan(FOV/(2*depths_feature)) / ref_angle
        return [*relative_aFOV], 1/depths_feature
