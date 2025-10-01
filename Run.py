import os
import warnings
import torch

from model.XCalib import XCalib
from options.options import get_options
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'expandable_segments:True'


def fit_cams() -> XCalib:
    cfg = get_options()
    model = XCalib(cfg)
    model.to(cfg.model['device'])
    model.optimize_parameters()
    return model


# def train(cfg, cfg_dict, datamodule: DataModule, device: torch.device, idx=None) -> None:


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    warnings.filterwarnings('ignore')
    xCalib = fit_cams()
    xCalib.save_cameras_rig()
