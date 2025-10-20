from .loss import Loss, LossCfgCommon
from .loss_flow import LossFlow, LossFlowCfg
from .loss_image import LossImageCfg, LossImage
from .loss_reg import LossReg, LossRegCfg

LOSSES = {
    "flow": LossFlow,
    "image": LossImage,
    "reg": LossReg}

LossCfg = LossFlowCfg | LossImageCfg | LossRegCfg | LossCfgCommon


def get_losses(cfgs: tuple[list[LossCfg]], targets: int):
    losses = [LOSSES[cfg.name](cfg, targets) for cfg in cfgs]

    class ModelLoss:
        def __init__(self, metrics: list):
            self.metrics = metrics
            self.losses = {metric.cfg.name: [None] for metric in metrics}

        def __call__(self, batch, global_step: int, cameras):
            loss_tot = 0.
            for metric in self.metrics:
                l = metric(batch, global_step, cameras)
                if l is not None:
                    loss_tot += l * metric.cfg.weight
                    self.losses[metric.cfg.name].append(l.detach().cpu().numpy())
                else:
                    self.losses[metric.cfg.name].append(None)

            return loss_tot

        def __str__(self):
            return ", ".join([f"{k}: {v[-1]:.4f}" for k, v in self.losses.items() if v[-1] is not None])

    return ModelLoss(losses)

