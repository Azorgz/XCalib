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
            self.losses = {metric.cfg.name: [] for metric in metrics}

        def __call__(self, batch, global_step: int):
            loss_tot = 0.
            for metric in self.metrics:
                l = metric(batch, global_step)
                self.losses[metric.cfg.name].append(l.detach().cpu().numpy())
                loss_tot += l * metric.cfg.weight
            return loss_tot

        def __str__(self):
            return ", ".join([f"{k}: {v[-1] if len(v) > 0 else 0.:.3f}" for k, v in self.losses.items()])

    return ModelLoss(losses)

