import os
from argparse import Namespace

import yaml
from torch.utils.data import DataLoader

from options import get_sampler_opt, get_dataset_opt, get_depth_options, get_loss_options, get_train_opt, \
    get_validation_opt


def get_options(*args, **kwargs) -> Namespace:
    with open(os.getcwd() + "/options/mainConf.yaml", "r") as file:
        options = yaml.safe_load(file)

    if args:
        if isinstance(args[0], DataLoader):
            options['data']['name'] = 'from_data'

    ## SAMPLER OPTIONS
    options = get_sampler_opt(options)

    ## DATASET OPTIONS
    options = get_dataset_opt(options, data=args[0] if args else None)

    ## Depth MODEL OPTIONS
    options = get_depth_options(options)

    ## Loss OPTIONS
    options = get_loss_options(options)

    ## Data OPTIONS
    options = get_train_opt(options)
    options = get_validation_opt(options)

    return Namespace(**options)
