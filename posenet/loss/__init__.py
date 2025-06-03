from .loss import Loss, LossCfgCommon
from .loss_flow import LossFlow, LossFlowCfg
from .loss_feature import LossFeature, LossFeatureCfg
from .loss_pose import LossPoseCfg, LossPose
from .loss_tracking import LossTracking, LossTrackingCfg
from .loss_image import LossImage, LossImageCfg

LOSSES = {
    "flow": LossFlow,
    "feature": LossFeature,
    "tracking": LossTracking,
    "pose": LossPose,
    "image": LossImage}

LossCfg = LossImageCfg | LossFeatureCfg | LossFlowCfg | LossTrackingCfg | LossPoseCfg | LossCfgCommon


def get_losses(cfgs: list[LossCfg]) -> list[Loss]:
    return [LOSSES[cfg.name](cfg) for cfg in cfgs]
