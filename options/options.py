import os
from argparse import Namespace

import yaml

from XCalib2.options import get_sampler_opt, get_dataset_opt, get_depth_options, get_loss_options


def get_options():
    with open(os.getcwd() + "/options/mainConf.yaml", "r") as file:
        options = yaml.safe_load(file)

    ## SAMPLER OPTIONS
    options = get_sampler_opt(options)

    ## DATASET OPTIONS
    options = get_dataset_opt(options)

    ## Depth MODEL OPTIONS
    options = get_depth_options(options)

    ## Loss OPTIONS
    options = get_loss_options(options)

    return Namespace(**options)
