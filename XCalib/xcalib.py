import os
import warnings

import hydra
import torch
import wandb
from lightning import Trainer
from omegaconf import DictConfig

from XCalib.config.common import get_typed_root_config
from XCalib.config.XCalib import XCalibCfg

from XCalib.dataset.data_module import DataModule, get_dataModule
from XCalib.export.cameras import export_to_cams
from XCalib.misc.common_training_setup import run_common_training_setup
from XCalib.model.model_wrapper import ModelWrapper


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="XCalib",
)
def fit_cams(cfg_dict: DictConfig) -> None:
    cfg = get_typed_root_config(cfg_dict, XCalibCfg)
    cfg.model.num_frames = cfg.frame_sampler.num_frames
    device = torch.device("cuda:0")
    datamodules = get_dataModule(device, cfg)
    if isinstance(datamodules, list):
        for i, datamodule in enumerate(datamodules):
            train(cfg, cfg_dict, datamodule, device, i)
    else:
        train(cfg, cfg_dict, datamodules, device)


def train(cfg, cfg_dict, datamodule: DataModule, device: torch.device, idx=None) -> None:
    cfg.model.num_cams = len(datamodule.dataset.cameras)
    cfg.model.spatial_transformer.bidirectional = cfg.loss[0].bidirectional or False
    cfg.model.spatial_transformer.cross_projection = cfg.loss[0].cross_loss or 0
    cfg.model.spatial_transformer.return_depth = cfg.loss[0].coeff[2] > 0

    # Set up the model.
    model_wrapper = ModelWrapper(cfg, datamodule.dataset.cameras)
    model_wrapper.to(device)

    val_check_interval = cfg.trainer.val_check_interval / (len(datamodule.dataset) * cfg.datamodule.train.rep_batch - 1)
    val_check_interval = val_check_interval if val_check_interval < 1 else min(len(datamodule.dataset) *
                                                                               cfg.datamodule.train.rep_batch,
                                                                               cfg.trainer.val_check_interval)
    cfg.wandb.name = cfg.wandb.name if cfg.wandb.name is not None else datamodule.dataset.name
    callbacks, logger, output_dir = run_common_training_setup(cfg, cfg_dict)
    trainer = Trainer(
        max_epochs=cfg.datamodule.epochs,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        strategy=(
            "ddp_find_unused_parameters_true"
            if torch.cuda.device_count() > 1
            else "auto"
        ),
        enable_model_summary=True,
        enable_checkpointing=False,
        callbacks=callbacks,
        val_check_interval=val_check_interval,
        max_steps=cfg.trainer.max_steps,
        log_every_n_steps=1)

    # Train the model.#####
    try:
        trainer.fit(model_wrapper, datamodule=datamodule)
    except torch.OutOfMemoryError:
        print("\nTraining stopped due to CUDA out of memory !")
    # Export the result.
    print("Exporting results.")
    model_wrapper.to(device)
    exports = model_wrapper.export()

    cam_path = output_dir
    export_to_cams(exports,
                   datamodule.dataset.cameras,
                   cam_path,
                   idx,
                   name=datamodule.dataset.name)

    if cfg.wandb.mode != "disabled":
        artifact = wandb.Artifact(f"rig_calibrated_{wandb.run.id}", type="Camera Rig")
        artifact.add_file(f'{cam_path}/rig_calibrated{f"_{idx}" if idx is not None else ""}.yaml',
                          name=f'rig_calibrated.yaml')
        wandb.log_artifact(artifact)
        artifact.wait()


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'expandable_segments:True'
    warnings.filterwarnings('ignore')
    fit_cams()
