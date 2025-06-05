from .loss import Loss, LossCfgCommon
from .loss_pose import LossPoseCfg, LossPose
from .loss_flow import LossFlow, LossFlowCfg

LOSSES = {
    "flow": LossFlow,
    "pose": LossPose}

LossCfg = LossFlowCfg | LossPoseCfg | LossCfgCommon


def get_losses(cfgs: list[LossCfg]) -> list[Loss]:
    return [LOSSES[cfg.name](cfg) for cfg in cfgs]
