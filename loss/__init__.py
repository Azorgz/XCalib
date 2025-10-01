from .loss import Loss, LossCfgCommon
from .loss_flow import LossFlow, LossFlowCfg
from .loss_pose import LossPoseCfg, LossPose

LOSSES = {
    "flow": LossFlow,
    "pose": LossPose}

LossCfg = LossFlowCfg | LossPoseCfg | LossCfgCommon


def get_losses(cfgs: tuple[list[LossCfg]], targets: int):
    losses = [LOSSES[cfg.name](cfg, targets) for cfg in cfgs]

    class ModelLoss:
        def __init__(self, metrics: list):
            self.metrics = metrics

        def __call__(self, batch, global_step: int):
            loss_tot = 0.
            for metric in self.metrics:
                loss_tot += metric(batch, global_step) * metric.cfg.weight
            return loss_tot

    return ModelLoss(losses)

