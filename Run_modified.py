import os
import sys
import warnings
from pathlib import Path

import torch

from hypercalib.configs import load_possible_dataset_list
from hypercalib.loading import load_yaml
from hypercalib.parser import build_parser
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
    """
    Make sure to check mainConf.yaml, and whether the mode is calibration_only or otherwise.
    Make sure to check the number of PNG images being used to fit, compared to the buffer in mainConf.yaml.
    Make sure to switch the mode back to registration_only once fitting is done.
    """
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    warnings.filterwarnings('ignore')

    XCALIB_ROOT = Path(__file__).parent
    hcalib_paths = Paths.create()
    hcalib_root = hcalib_paths.root
    hcalib_data = hcalib_root / "data"
    hcalib_xcalib = hcalib_root / "configs/xcalib"
    output_subdir = hcalib_xcalib.relative_to(XCALIB_ROOT)

    parser = build_parser()

    # Detect IDE run (no CLI args)
    if len(sys.argv) == 1:
        input_dataset_names = [
            # "ultris_rpi_fixed",
            # "ultris_rpi_fixed_degraded",
            "real",
        ]
        input_scale_factor_names = [
            "x4",
            "x8",
        ]
        input_model_names = []
        args = parser.parse_args([
            # Comment the option that you want as None: means "all options"
            "-d", *input_dataset_names,
            "-sf", *input_scale_factor_names,
        ])
    else:
        args = parser.parse_args()

    input_dataset_names = args.input_dataset_names
    input_scale_factor_names = args.input_scale_factor_names
    padding_mode = "zeros"

    # All possible options
    datasets_options = load_possible_dataset_list()

    # Filtering
    if input_dataset_names is not None:
        datasets_options = [
            ds for ds in datasets_options
            if ds["name"] in input_dataset_names
        ]

    # Execution loop
    for dataset_options in datasets_options:
        crops_options = load_yaml(filepath=hcalib_root / dataset_options["crops_subdir"])
        sfs_options = load_yaml(filepath=hcalib_root / dataset_options["sf_subdir"])
        if input_scale_factor_names is not None:
            sfs_options = [
                sf for sf in sfs_options
                if sf["name"] in input_scale_factor_names
            ]

        for crop_cfg in crops_options:
            for sf_cfg in sfs_options:
                dataset_cfg_path = hcalib_root / dataset_options["subdir"]
                dataset_cfg = load_yaml(filepath=dataset_cfg_path)
                dataset = dataset_options["class"](**dataset_cfg)
                dataset.update_crop(crop_cfg=crop_cfg)
                dataset.update_scale_factor(scale_factor_cfg=sf_cfg)

                dataset_id = dataset.id
                crop_name = crop_cfg["name"]
                sf_name = dataset.scale_factor

                print(f"Dataset: '{dataset_id}'. Length = {len(dataset)}.")
                print(f"Crop: '" + crop_cfg["name"] + "'.")
                print(f"Scale Factor: x{dataset.scale_factor}.")

                calibration_folder_name = dataset.compose_xcalib_id(crop_name, sf_name)
                folder_name = f"xcalib/{calibration_folder_name}"
                camera_png_dirs = [
                    dataset.hsi_dir / folder_name,  # reference
                    dataset.rgb_dir / folder_name,  # source
                ]
                camera_names = [
                    dataset.hsi_camera_name,
                    dataset.rgb_camera_name,
                ]
                cfg = get_options(
                    name_experiment=calibration_folder_name,
                    path_to_calib=f"{output_subdir}/{calibration_folder_name}/calibration.yaml",  # Relative path from 'output' to calib file
                    output=f"{output_subdir}",  # Relative path from ROOT to general outputs
                    root_cameras=[f"{d}" for d in camera_png_dirs],  # Where to find the PNG files
                    cameras_name=camera_names,  # Camera names
                )

                if cfg.run_parameters['mode'] in ['all_in_one', 'calibration_only']:
                    xcalib = fit_cams(cfg)
                else:
                    xcalib = XCalib(cfg=cfg).to(cfg.model['device'])
                if cfg.run_parameters['mode'] in ['all_in_one', 'registration_only']:
                    xcalib.wrap_all(padding_mode=padding_mode)

                print()
