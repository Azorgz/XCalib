import os
import warnings
import torch

from model.XCalib import XCalib
from options.options import get_options
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'expandable_segments:True'


def fit_cams(config) -> XCalib:
    model = XCalib(config)
    model = model.to(config.model['device'])
    model.optimize_parameters()
    return model


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    warnings.filterwarnings('ignore')
    cfg = get_options()
    if cfg.run_parameters['mode'] in ['all_in_one', 'calibration_only']:
        xcalib = fit_cams(cfg)
        if xcalib.cfg.run_parameters['save_calib']:
            xcalib.save_cameras_rig()
    else:
        xcalib = XCalib(cfg).to(cfg.model['device'])
    if cfg.run_parameters['mode'] in ['all_in_one', 'registration_only']:
        xcalib.wrap_all()
