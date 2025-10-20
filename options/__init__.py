import os
from argparse import Namespace
import yaml
from matplotlib import colormaps
from model.backbone import BackboneCfg
from misc.Mytypes import CamerasCfg


def get_train_opt(opt):
    cfg_train_data = {'buffer_size': opt['model']['buffer_size'],
                      'nb_cam': opt['data'].nb_cam,
                      'batch_size': opt['model']['train']['batch_size'],
                      'target': opt['model']['target'],
                      'cameras_names': opt['data'].cameras_name}
    opt['train_collector'] = Namespace(**cfg_train_data)
    opt['model']['train']['lr'] = float(opt['model']['train']['lr'])
    opt['model']['train']['lr_after_unfreeze'] = float(opt['model']['train']['lr_after_unfreeze'])
    return opt


def get_validation_opt(opt):
    if opt['model']['validation']['buffer_idx'] is not None:
        assert opt['model']['validation']['buffer_idx']
        opt['model']['validation']['buffer_size'] = len(opt['model']['validation']['buffer_idx'])
    assert opt['model']['validation']['mode_fusion'] in ['alpha_blending', 'chessboard', 'cross', 'crossfused', 'ldiag', 'rdiag', 'vstrip', 'hstrip'], 'This fusion mode does not exist'
    assert opt['model']['validation']['color_map_infrared'] in list(colormaps), 'This color does not exist'
    cfg_val_data = {'buffer_size': opt['model']['validation']['buffer_size'],
                    'nb_cam': opt['data'].nb_cam,
                    'mode_fusion': opt['model']['validation']['mode_fusion'],
                    'color_map_infrared': opt['model']['validation']['color_map_infrared'],
                    'batch_size': opt['model']['validation']['buffer_size'],
                    'target': opt['model']['target'],
                    'cameras_names': opt['data'].cameras_name,
                    'buffer_idx': opt['model']['validation']['buffer_idx']}
    opt['val_collector'] = Namespace(**cfg_val_data)
    return opt


def get_sampler_opt(opt):
    sampler = opt['model']['train']['frame_sampler']
    if sampler == 'random':
        opt['frame_sampler'] = {
            'name': 'random',
            'num_frames': opt['model']['train']['batch_size']}
    else:
        with open(os.getcwd() + "/options/frame_sampler/sequential.yaml", "r") as file:
            sampler_opt = yaml.safe_load(file)
        opt['frame_sampler'] = sampler_opt
        opt['frame_sampler']['num_frames'] = opt['model']['train']['batch_size']
    opt['frame_sampler'] = Namespace(**opt['frame_sampler'])
    return opt


def get_dataset_opt(opt):
    dataset = opt['data']['name']
    with open(os.getcwd() + f"/options/dataset/{dataset}.yaml", "r") as file:
        dataset_opt = yaml.safe_load(file)
    opt['data'].update(dataset_opt)
    opt['data']['from_file'] = opt['run_parameters']['path_to_calib'] if opt['run_parameters']['mode'] == 'registration_only' else None
    opt['data'] = CamerasCfg(**opt['data'])
    return opt


def get_depth_options(opt):
    depth_model = opt['model']['depth']
    with open(os.getcwd() + f"/options/depth/{depth_model}.yaml", "r") as file:
        depth_opt = yaml.safe_load(file)
    opt['model']['depth'] = BackboneCfg[depth_model](**depth_opt)
    return opt


def get_loss_options(opt):
    losses = opt['model']['train']['loss']

    with open(os.getcwd() + f"/options/loss/losses.yaml", "r") as file:
        losses_opt = yaml.safe_load(file)

    for l in losses:
        loss_opt = losses_opt[l] if l in losses_opt else {}
        l_cfg = {'name': l}
        enable_after = eval(loss_opt['enable_after']) if type(loss_opt['enable_after']) is str else float(loss_opt['enable_after'])
        loss_opt['enable_after'] = int(enable_after * opt['model']['train']['epochs']) if (
                enable_after < 1) else enable_after
        if loss_opt['disable_after'] is not None:
            disable_after = eval(loss_opt['disable_after']) if (type(loss_opt['disable_after'])
                                                                is str) else float(loss_opt['enable_after'])
        else:
            disable_after = None
        if disable_after is not None:
            loss_opt['disable_after'] = int(disable_after * (opt['model']['train']['epochs'] - 1)) if (
                    disable_after < 1) else disable_after
        else:
            loss_opt['disable_after'] = None
        l_cfg.update(loss_opt)
        l_cfg = Namespace(**l_cfg)
        idx = opt['model']['train']['loss'].index(l)
        opt['model']['train']['loss'][idx] = l_cfg
    opt['model']['train']['loss'] = tuple(opt['model']['train']['loss'])

    return opt
