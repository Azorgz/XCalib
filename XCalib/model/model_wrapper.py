import os
from dataclasses import dataclass
from pathlib import Path

import torch
import wandb
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import optim

from ..third_party.ImagesCameras import CameraSetup
from .model import Model, ModelExports
from ..config.common import CommonCfg
from ..dataset.dataset_cameras import CameraBundle
from ..loss import get_losses
from ..misc.image_io import prep_image
from ..misc.local_logger import LOG_PATH
from ..visualization import get_visualizers


@dataclass
class ModelWrapperCfg:
    lr: float | list
    patch_size: int


class ModelWrapper(LightningModule):
    def __init__(
            self,
            cfg: CommonCfg,
            cameras: CameraBundle,
    ) -> None:
        super().__init__()
        self.cfg = cfg.model_wrapper
        self.lr = self.cfg.lr
        self.model = Model(cfg.model, cameras)
        self.losses = get_losses(cfg.loss)
        self.visualizers = get_visualizers(cfg.visualizer)
        self.last_batch = None
        self.logger_init = False

    def to(self, device: torch.device) -> None:

        self.model.to(device)
        super().to(device)

    def training_step(self, batch):
        # Compute depths, poses, and intrinsics using the model.
        self.last_batch = batch
        model_output = self.model(self.last_batch, self.global_step)
        if isinstance(self.lr, list):
            self.log('learning_rate', self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0],
                     on_step=True, prog_bar=True)
        # Compute and log the loss.
        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(batch, model_output, self.global_step)
            self.log(loss_fn.cfg.name, float(loss.detach().cpu() if isinstance(loss, torch.Tensor) else loss),
                     on_step=True, prog_bar=True)
            self.logger.log_metrics(
                {loss_fn.cfg.name: float(loss.detach().cpu() if isinstance(loss, torch.Tensor) else loss)},
                step=self.global_step)
            total_loss = total_loss + loss
        return total_loss

    def validation_step(self, batch):
        if self.last_batch is not None:
            batch = self.last_batch
        else:
            self.last_batch = batch
        # Compute depths, poses, and intrinsics using the model.
        model_output = self.model(self.last_batch, self.global_step)
        if isinstance(self.logger, WandbLogger) and not self.logger_init:
            self.logger.watch(self.model, log="parameters", log_freq=1)
            self.logger_init = True

        # Generate visualizations.
        for visualizer in self.visualizers:
            visualizations = visualizer.visualize(
                batch,
                model_output,
                self.model,
                self.global_step,
            )
            for key, visualization_or_metric in visualizations.items():
                if isinstance(visualization_or_metric, torch.Tensor):
                    if visualization_or_metric.ndim == 0:
                        # If it has 0 dimensions, it's a metric.
                        self.logger.log_metrics(
                            {key: visualization_or_metric},
                            step=self.global_step,
                        )
                    else:
                        # If it has 3 dimensions, it's an image.
                        image = [prep_image(visualization_or_metric)]
                        self.logger.log_image(key, image, step=self.global_step)
                elif isinstance(visualization_or_metric, CameraSetup):
                    if isinstance(self.logger, WandbLogger):
                        Path(self.logger.save_dir).mkdir(exist_ok=True, parents=True)
                        visualization_or_metric.save(self.logger.save_dir, 'rig_calibrated.yaml')
                        artifact = wandb.Artifact(f"rig_calibrated_{wandb.run.id}", type="Camera Rig")
                        artifact.add_file(f"{self.logger.save_dir}/rig_calibrated.yaml", name="CameraRig.yaml")
                        wandb.log_artifact(artifact)
                        artifact.wait()
                    else:
                        path = f'{os.getcwd()}/{LOG_PATH}/Rig_calibrated/'
                        Path(path).mkdir(exist_ok=True, parents=True)
                        visualization_or_metric.save(path, f'rig_calibrated_{self.global_step}.yaml')

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if isinstance(self.lr, list):
            optimizer = optim.Adam(self.parameters(), lr=self.cfg.lr[0])
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                          lr_lambda=lambda step: self.cfg.lr[1]/self.cfg.lr[0] +
                                                                                 (1 - self.cfg.lr[1]/self.cfg.lr[0]) /
                                                                                 (1.0015 ** step))
            lr_scheduler_config = {
                # REQUIRED: The scheduler instance
                "scheduler": scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "step",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                "monitor": "val_loss",
                # If set to `True`, will enforce that the value specified 'monitor'
                # is available when the scheduler is updated, thus stopping
                # training if not found. If set to `False`, it will only produce a warning
                "strict": True,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": None,
            }
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
            return optimizer

    def export(self) -> ModelExports:
        return self.model.export(
            self.last_batch,
            self.global_step)
