import os
import warnings
from pathlib import Path

import torch

from hypercalib.paths import Paths
from model.XCalib import XCalib
from options.options import get_options
os.environ["PYTORCH_ALLOC_CONF"] = 'expandable_segments:True'


def fit_cams(config) -> XCalib:
    model = XCalib(cfg=config)
    model = model.to(config.model['device'])
    model.optimize_parameters()
    return model


if __name__ == "__main__":
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    warnings.filterwarnings('ignore')

    hcalib_paths = Paths.create()
    hcalib_root = hcalib_paths.root
    hcalib_data = hcalib_root / "data"
    hcalib_xcalib = hcalib_root / "configs/xcalib"

    dataset_subdir = "ultris-rpi-sequence-highres/fixed_target_coupled"
    cameras = ["ultris_sr5", "rpi_hq"]  # [Reference, Source]
    crop_name = "*"
    sf_name = "sf4"
    padding_mode = "border"

    camera_dirs = [hcalib_data / dataset_subdir / cam for cam in cameras]
    pattern = f"preview_{crop_name}_{sf_name}"
    folder_names = sorted([f.stem for f in camera_dirs[0].glob(pattern=pattern)])

    XCALIB_ROOT = Path(__file__).parent
    for folder_name in folder_names:
        exp_name = f"{dataset_subdir}/{folder_name}"
        output_subdir = hcalib_xcalib.relative_to(XCALIB_ROOT)

        cfg = get_options(
            name_experiment=exp_name,
            path_to_calib=f"{output_subdir}/{exp_name}/parameters.yaml",  # Relative path from 'output' to calib file
            output=f"{output_subdir}",  # Relative path from ROOT to general outputs
            root_cameras=[f"{p}/{folder_name}" for p in camera_dirs],
        )

        if cfg.run_parameters['mode'] in ['all_in_one', 'calibration_only']:
            xcalib = fit_cams(cfg)
        else:
            xcalib = XCalib(cfg=cfg).to(cfg.model['device'])
        if cfg.run_parameters['mode'] in ['all_in_one', 'registration_only']:
            xcalib.wrap_all(padding_mode=padding_mode)
