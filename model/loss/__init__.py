import numpy as np

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

    class LossModel:
        def __init__(self, metrics: list):
            self.metrics = metrics
            self.losses = {metric.cfg.name: [] for metric in metrics}
            self.colors = ['green', 'red', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

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
            if any(l for l in self.losses.values()):
                return ", ".join([f"{k}: {v[-1]:.4f}" for k, v in self.losses.items() if v[-1] is not None])
            else:
                return "No losses computed yet."

        def plot(self, name=None, save_path=None):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            for (k, v), c in zip(self.losses.items(), self.colors):
                if len(v) > 0:
                    ax.plot(v, label=k, color=c)
            # Get all legend items
            handles, labels = ax.get_legend_handles_labels()
            plt.xlim(0, 160)
            plt.ylim(0, 0.6)
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.title(name)
            ax.legend(handles[:len(self.losses)], labels[:len(self.losses)], loc='upper right')
            if save_path is not None:
                name = name if name is not None else 'losses'
                plt.savefig(save_path + f'{name}.png')
            else:
                plt.show()

    return LossModel(losses)

