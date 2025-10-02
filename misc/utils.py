import argparse
import math
import ntpath
import os
import stat
import subprocess as sp

import torch
import yaml


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [float(x.split()[0])/1024 for i, x in enumerate(memory_free_info)]
    return memory_free_values


def configure_parser(parser, config, path_config=None, dict_vars=None):
    dict_pars = {}
    if parser is not None:
        if isinstance(parser, argparse.ArgumentParser):
            dict_pars = vars(parser.parse_args())
        elif isinstance(parser, dict):
            dict_pars = parser
    config_vars = {}
    if path_config:
        if isinstance(path_config, str or os.path):
            with open(path_config, 'r') as file:
                config_vars = yaml.safe_load(file)
        else:
            raise TypeError("A path or a String is expected for the config file")
    if not dict_vars:
        dict_vars = {}
    if not config:
        config = {}
    config_vars = config_vars | dict_vars
    config_vars = config_vars | config
    for key, value in config_vars.items():
        try:
            dict_pars[key] = value
        except KeyError:
            print(f"The Key {key} in the config file doesn't exist in this parser")
    return argparse.Namespace(**dict_pars)


# Names managing functions ##################################
def name_generator(idx, max_number=10e4):
    k_str = str(idx)
    digits = 1 if max_number < 10 else int(math.log10(max_number)) + 1
    current_digits = 1 if idx < 10 else int(math.log10(idx)) + 1
    for i in range(digits - current_digits):
        k_str = '0' + k_str
    return k_str


def time2str(t, optimize_unit=True):
    if not optimize_unit:
        return str(round(t, 3)) + ' sec'
    else:
        unit = 0
        unit_dict = {-1: " h", 0: " s", 1: " ms", 2: " us", 3: " ns"}
        while t < 1:
            t *= 1000
            unit += 1

        if t > 3600:
            t /= 3600
            unit = -1
            str_time = str(int(t)) + unit_dict[unit] + str(t % 1) + unit_dict[unit + 1]
        else:
            str_time = str(round(t, 3)) + unit_dict[unit]
        return str_time


def select_device(gpu_id=None, verbose=False):
    if gpu_id is None or gpu_id == '-1' or gpu_id == '-1,-1':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        assert torch.cuda.is_available(), f"CUDA is not available. Please check your CUDA installation or set gpu_id to -1"
        if verbose:
            n_gpus = torch.cuda.device_count()
            list_free_mem = get_gpu_memory()
            if n_gpus > 1:
                print(f"Device {device} selected, {n_gpus} GPUs are available with memory (in Gb) {list_free_mem}")
            else:
                print(f"Device {device} selected, {list_free_mem[0]:.2f}Gb free")
    return device


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def make_writable(folder_path):
    os.chmod(folder_path, stat.S_IRWXU)
    dirs = os.listdir(folder_path)
    for d in dirs:
        os.chmod(os.path.join(folder_path, d), stat.S_IRWXU)


def update_name(path):
    i = 0
    path_exp = path + f"({i})"
    path_ok = not os.path.exists(path_exp)
    while not path_ok:
        i += 1
        path_exp = path + f"({i})"
        path_ok = not os.path.exists(path_exp)
    return path_exp


def update_name_tree(sample: dict, suffix):
    for im in sample.values():
        im.name += '-' + suffix
    return sample


def path_leaf(path):
    if os.path.isdir(path):
        res = ntpath.split(path)[-1]
    else:
        res = ntpath.split(path)[-1].split(".")[:-1]
    if isinstance(res, list):
        if len(res) > 1:
            res = ''.join(res)
        else:
            res = res[0]
    return res


def add_ext(word, ext):
    if ext != '':
        return f'{word}_{ext}'
    else:
        return word