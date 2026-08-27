import gc
from pathlib import Path

import numpy as np
import torch
from ImagesCameras import ImageTensor
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_template import FigureCanvas

from .loss import Loss, LossCfgCommon
from .loss_flow import LossFlow, LossFlowCfg
from .loss_image import LossImageCfg, LossImage
from .loss_reg import LossReg, LossRegCfg
from .loss_scale import LossSemanticScale, LossSemanticScaleCfg
import matplotlib.pyplot as plt

LOSSES = {
    "flow": LossFlow,
    "image": LossImage,
    "reg": LossReg,
    "scale": LossSemanticScale}

LossCfg = LossFlowCfg | LossImageCfg | LossRegCfg | LossSemanticScaleCfg | LossCfgCommon

class Drawer:

    def __init__(self, validation_loss, training_loss):
        # 1. Reuse a single persistent Figure, Axis, and Canvas instance
        self.fig = plt.Figure(dpi=300)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasAgg(self.fig)
        self.colors_training = ['green', 'red', 'blue', 'orange']
        self.colors_validation = ['lightgreen', 'lightcoral', 'lightblue', 'lightsalmon']
        self.total_steps = 100
        self.x_lim = 0, 100
        self.y_lim = 0, 0.6
        self.fix_y_axis = False


        # 2. Pre-create line artists and static LaTeX labels ONCE during initialization
        # This prevents Matplotlib from parsing math expressions & leaking memory in loops
        self.val_lines = {}
        for (k, _), c in zip(
            validation_loss.items(), self.colors_validation
        ):
            label = fr"$\mathcal{{L}}_{{\mathrm{{{k}}}}}$ validation"
            (line,) = self.ax.plot(
                [],
                [],
                marker="o",
                linestyle="None",
                markerfacecolor="none",
                markeredgecolor=c,
                markeredgewidth=1.0,
                markersize=5,
                label=label,
            )
            self.val_lines[k] = line

        self.train_lines = {}
        for (k, _), c in zip(
            training_loss.items(), self.colors_training
        ):
            label = fr"$\mathcal{{L}}_{{\mathrm{{{k}}}}}$ training"
            (line,) = self.ax.plot([], [], label=label, color=c)
            self.train_lines[k] = line

        # 3. Configure static axis labels and legend ONCE
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("Loss")
        self.legend = self.ax.legend(loc="upper right")

    def plot(self, validation_loss, training_loss, screen_plot=None, name=None, save_path=None):
        min_y, max_y = self.y_lim

        # Update Validation lines in-place
        for k, v in validation_loss.items():
            if len(v) > 0:
                # Fast numpy array creation
                x = np.arange(1, len(v) + 1)
                self.val_lines[k].set_data(x, v)

                # Min/max calculation optimized (filters out Nones efficiently)
                valid_vals = [val for val in v if val is not None]
                if valid_vals:
                    min_y = min(min_y, self.y_lim[0], min(valid_vals))
                    max_y = max(max_y, self.y_lim[1], max(valid_vals) + 0.1)

        # Update Training lines in-place
        for k, v in training_loss.items():
            if len(v) > 0:
                x = np.arange(len(v))
                self.train_lines[k].set_data(x, v)

                valid_vals = [val for val in v if val is not None]
                if valid_vals:
                    min_y = min(min_y, self.y_lim[0], min(valid_vals))
                    max_y = max(max_y, self.y_lim[1], max(valid_vals) + 0.1)

        if not self.fix_y_axis:
            self.y_lim = (min_y, max_y)

        # Update axes limits & title without clearing artists
        self.ax.set_xlim(*self.x_lim)
        self.ax.set_ylim(*self.y_lim)
        if name:
            self.ax.set_title(name)

        # Render on persistent canvas
        self.canvas.draw()

        # Zero-copy buffer read
        rgba_buffer = self.canvas.buffer_rgba()
        img_array = np.asarray(rgba_buffer)[..., :3]

        # Convert to Tensor (No ax.cla(), fig.clf(), or gc.collect needed!)
        plot_image = ImageTensor(img_array).cpu()

        if screen_plot is not None:
            screen_plot.update(plot_image)
        else:
            screen_plot = plot_image.show(
                name="Losses", opencv=True, asyncr=True
            )

        if save_path is not None:
            name_str = name if name is not None else "losses"
            plot_image.save(f"{save_path}{name_str}.png")

        return screen_plot, plot_image

def get_losses(cfgs: tuple[list[LossCfg]], targets: int):
    losses = [LOSSES[cfg.name](cfg, targets) for cfg in cfgs]

    class LossModel:
        def __init__(self, metrics: list):
            self.metrics = metrics
            self.training_loss = {metric.cfg.name: [] for metric in metrics}
            self.validation_loss = {metric.cfg.name: [] for metric in metrics if metric.cfg.name in ['flow', 'image']}
            self.epochs = 1
            self.current_epoch = 0
            self.current_step = 0
            self.drawer = Drawer(self.validation_loss, self.training_loss)

        def __call__(self, batch, global_step: int, cameras, training: bool=True):
            loss_tot = 0.
            if training:
                self.current_epoch = (global_step + 1) // (self.total_steps // self.epochs)
                self.current_step = global_step + 1
            for metric in self.metrics:
                l = None
                if training:
                    l = metric(batch, self.current_epoch, cameras)
                else:
                    with torch.no_grad():
                        l = metric(batch, self.current_epoch, cameras)
                if l is not None:
                    loss_tot += l * metric.cfg.weight
                    if training:
                        self.training_loss[metric.cfg.name].append(l.detach().cpu().numpy())
                    elif metric.cfg.name in self.validation_loss:
                        self.validation_loss[metric.cfg.name].extend([None] * max(0, len(self.training_loss[metric.cfg.name]) - len(self.validation_loss[metric.cfg.name]) - 1))
                        self.validation_loss[metric.cfg.name].append(l.detach().cpu().numpy())
                else:
                    if training:
                        self.training_loss[metric.cfg.name].append(None)
                    elif metric.cfg.name in self.validation_loss:
                        self.validation_loss[metric.cfg.name].extend([None] * max(0, len(
                            self.training_loss[metric.cfg.name]) - len(self.validation_loss[metric.cfg.name])))
            return loss_tot

        def __str__(self):
            if any(l for l in self.training_loss.values()):
                return ", ".join([f"{k}: {v[-1]:.4f}" for k, v in self.training_loss.items() if v[-1] is not None])
            else:
                return "No losses computed yet."

        def set_total_steps(self, total_steps, epochs=1):
            self.total_steps = total_steps
            self.drawer.x_lim = 0, total_steps
            self.epochs = epochs

        def get_losses(self, per_epoch=False):
            if per_epoch:
                step = self.current_step // self.current_epoch
                step_validation = len(list(self.validation_loss.items())[0]) // self.current_epoch
                losses_training = {k: [np.mean([v[e * step + i] or 0 for i in range(step)]) for e in range(self.current_epoch)] for k, v in self.training_loss.items()}
                losses_training['total'] = [sum([losses_training[k][e] * self.metrics[i].cfg.weight for i, k in enumerate(losses_training.keys())]) for e in range(self.current_epoch)]
                losses_validation = {k: [np.mean([v[e * step_validation + i] or 0 for i in range(step_validation)]) for e in range(self.current_epoch)] for k, v in self.validation_loss.items()}
                losses_validation['total'] = [sum([losses_validation[k][e] * self.metrics[i].cfg.weight for i, k in enumerate(losses_validation.keys())]) for e in range(self.current_epoch)]
            else:
                losses_training = {k: v for k, v in self.training_loss.items()}
                losses_training['total'] = [sum([losses_training[k][i] * self.metrics[i].cfg.weight for i, k in enumerate(losses_training.keys())]) for i in range(self.current_step)]
                losses_validation = {k: v for k, v in self.validation_loss.items()}
                losses_validation['total'] = [sum([losses_validation[k][i] * self.metrics[i].cfg.weight for i, k in
                                                 enumerate(losses_validation.keys())]) for i in range(self.current_step)]
            return losses_training, losses_validation

        def save_losses(self, save_path: Path, per_epoch=False):
            losses_training, losses_validation = self.get_losses(per_epoch=per_epoch)
            save_validation = save_path / 'validation_loss.txt'
            save_training = save_path / 'training_loss.txt'
            #  validation save
            validation_exists = save_validation.exists()
            # Format the data array
            data = np.array([losses_validation[k] for k in losses_validation.keys()]).T
            # Open in append mode ('a')
            with open(save_validation, 'a') as f:
                np.savetxt(
                    f,
                    data,
                    header=('\n' if validation_exists else '') + '                      '.join(losses_validation.keys()),
                    comments=''  # Optional: prevents '#' before header if you don't want it
                )

            #  training save
            training_exists = save_training.exists()
            # Format the data array
            data = np.array([losses_training[k] for k in losses_training.keys()]).T
            # Open in append mode ('a')
            with open(save_training, 'a') as f:
                np.savetxt(
                    f,
                    data,
                    header=('\n' if training_exists else '') + '                      '.join(losses_training.keys()),
                    comments=''  # Optional: prevents '#' before header if you don't want it
                )

        def plot(self, screen_plot=None, name=None, save_path=None):
            return self.drawer.plot(self.validation_loss, self.training_loss, screen_plot=screen_plot, name=name, save_path=save_path)

    return LossModel(losses)

