import math
from os.path import isfile

import cv2
import numpy as np
import torch
from ImagesCameras import ImageTensor, CameraSetup
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from backbone import get_backbone
from depth_refiner.refiner import DepthRefiner
from enhancer import get_enhancer
from frame_sampler import get_frame_sampler
from loss import get_losses
from misc.Mytypes import Batch
from misc.utils import check_path, select_device
from model.cameras import Cameras
from model.spatial_transformer import depth_warp


class XCalib(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        #  Base options
        self.experiment = cfg.name_experiment
        self.scheduler = None
        self.depth_scheduler = None
        self.optimizer = None
        self.depth_optimizer = None
        self.path = cfg.output + '/' + cfg.name_experiment
        check_path(self.path)
        self.device = select_device(cfg.model['device'])
        self.cfg = cfg
        self.output_path = cfg.output + cfg.data.name
        self.train_parameters = self.cfg.model['train']
        self.validation_parameters = self.cfg.model['validation']
        #  Models
        if cfg.run_parameters['mode'] == 'registration_only':
            assert isfile(cfg.run_parameters['path_to_calib']), \
                "Please provide a valid path to a calibration file to run registration only"
        self.cameras = Cameras(cfg.data, get_frame_sampler(cfg.frame_sampler))
        self.depthModel = get_backbone(cfg.model['depth'])
        self.LossModel = get_losses(self.train_parameters['loss'], self.cfg.model['target'])
        if cfg.run_parameters['enhance_result_quality']:
            self.enhancer = get_enhancer()
        else:
            self.enhancer = None
        self.depth_refiner = DepthRefiner(self.cfg.model['target'])
        # Data
        self.number_cameras = len(self.cameras.cameras)
        self.images = DataCollector(cfg.train_collector)
        self.validation = DataCollector(cfg.val_collector)
        self.camera_target = self.cfg.model['target']
        self.define_optimizers()

    def to(self, device: torch.device) -> 'XCalib':
        return super().to(device)

    def buffer_batch(self, batch: Batch):
        self.images.add(batch)

    def load_images(self, idx) -> Batch:
        return self.cameras[idx]

    def load_images_by_indices(self, indices):
        return self.cameras.load_images(indices)

    @torch.no_grad()
    def compute_depths(self, batch: Batch):
        with torch.no_grad():
            depths = self.depthModel(batch.images[self.camera_target])
        batch.depths = [depths if i == self.camera_target else None for i in range(self.number_cameras)]
        return batch

    def buffer_all(self):
        waitbar = tqdm(total=self.images.size + self.validation.size, desc='Buffering Images and Depths')
        idx_batch = 0
        if self.validation_parameters['visualize_validation'] and not self.validation.isfull:
            if self.validation_parameters['buffer_idx'] is not None:
                indices = self.validation_parameters['buffer_idx']
            else:
                indices = np.random.randint(low=0, high=self.cameras.total_frames,
                                            size=self.validation_parameters['buffer_size'])
            batch = self.load_images_by_indices(indices)
            batch = self.compute_depths(batch)
            self.validation.add(batch)
            waitbar.update(self.validation.size)
        while not self.images.isfull:
            batch = self.load_images(idx_batch)
            batch = self.compute_depths(batch)
            self.buffer_batch(batch)
            idx_batch += 1
            waitbar.update(self.train_parameters['batch_size'])
        waitbar.close()

    def optimize_parameters(self):
        self.buffer_all()
        optimization_step = 0
        waitbar = tqdm(total=len(self.images)*self.train_parameters['epochs'],
                       desc=f'Optimization of the parameters epoch {0}, lr: {self.optimizer.param_groups[0]["lr"]*1000:.3f}e-3, {self.LossModel}')
        self.cameras.initial_freeze()
        screen = None
        for e in range(self.train_parameters['epochs']):
            if e == int(self.train_parameters['unfreeze']*self.train_parameters['epochs']):
                self.cameras.unfreeze()
                self.optimizer.param_groups[0]["lr"] = self.train_parameters['lr_after_unfreeze']

            for j in range(math.ceil(len(self.images)/self.train_parameters['batch_size'])):
                waitbar.desc = f'Optimization of the parameters epoch {e}, lr: {self.optimizer.param_groups[0]["lr"] * 1000:.3f}e-3, {self.LossModel}'
                optimization_step += 1
                batch = self.images.get_batch(j)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # Wrap images
                batch = self.wrap_frame_to_target(batch)
                # Computes losses
                loss = self.LossModel(batch, e, self.cameras)
                loss.backward()
                self.optimizer.step()
                self.step_scheduler()
                waitbar.update(self.train_parameters['batch_size'])
                if self.validation_parameters['visualize_validation'] and optimization_step % self.validation_parameters['step_visualize'] == 0:
                    screen, valid_image = self.validation_step(screen)
        waitbar.close()
        self.cameras.freeze()
        screen.close()
        if self.train_parameters['refine_depth']:
            self.optimize_depth_refiner()
        elif self.validation_parameters['visualize_validation']:
            valid_image.show(name=f'Final result', opencv=True)

    def optimize_depth_refiner(self):
        self.buffer_all()
        optimization_step = 0
        waitbar = tqdm(total=len(self.images)*self.train_parameters['epoch_refine_depth'],
                       desc=f'Optimization of the parameters epoch {0}, lr: {self.depth_optimizer.param_groups[0]["lr"]*1e6:.3f}e-6, {self.LossModel}')
        screen = None
        for e in range(self.train_parameters['epoch_refine_depth']):
            for j in range(math.ceil(len(self.images)/self.train_parameters['batch_size'])):
                waitbar.desc = f'Optimization of the parameters epoch {e}, lr: {self.depth_optimizer.param_groups[0]["lr"]*1e6:.3f}e-6, {self.LossModel}'
                optimization_step += 1
                batch = self.images.get_batch(j)
                # zero the parameter gradients
                self.depth_optimizer.zero_grad()
                # Wrap images
                batch = self.wrap_frame_to_target(batch, refine_depth=True)
                # Computes losses
                loss = self.LossModel(batch, 0, self.cameras)
                loss.backward()
                self.depth_optimizer.step()
                self.step_depth_scheduler()
                waitbar.update(self.train_parameters['batch_size'])
                if self.validation_parameters['visualize_validation'] and optimization_step % self.validation_parameters['step_visualize'] == 0:
                    screen, valid_image = self.validation_step(screen, refine_depth=True)
        if self.validation_parameters['visualize_validation']:
            screen.close()
            valid_image.show(name=f'Final result', opencv=True)
        waitbar.close()

    def validation_step(self, screen, refine_depth=False):
        batch_val = self.validation.get_batch(0)
        batch_val = self.wrap_frame_to_target(batch_val, refine_depth=refine_depth)
        img = ImageTensor(batch_val.projections[0] / 2) + ImageTensor(batch_val.projections[1] / 2)
        if screen is None:
            screen = img.show(name=f'Optimization on going...', opencv=True, asyncr=True)
        else:
            screen.update(img)
        return screen, img

    def wrap_all(self):
        old_setup = self.cameras.cfg
        self.cameras.cfg.normalize_visible = self.cameras.cfg.equalize_visible = False
        self.cameras.cfg.normalize_infrared = self.cameras.cfg.equalize_infrared = False
        waitbar = tqdm(total=self.cameras.total_frames, desc='Wrapping all frames to target camera')
        for batch in self.cameras:
            batch = self.compute_depths(batch)
            batch = self.wrap_frame_to_target(batch)
            for i in range(self.number_cameras):
                if i != self.camera_target:
                    projections = batch.projections[i].split(1, 0)
                    for p, path in zip(projections, batch.frame_paths[i]):
                        name = path.split('/')[-1].split('.')[0]
                        if self.enhancer is not None:
                            p = self.enhancer(p)
                        if self.cameras.modality[i] != 'Visible':
                            p = ImageTensor(p).GRAY()
                        else:
                            p = ImageTensor(p)
                        p.save(self.output_path, name=name, depth=8)
            waitbar.update(len(batch.indices))
        self.cameras.cfg = old_setup

    def wrap_frame_to_target(self, batch: Batch, refine_depth=False) -> Batch:
        proj = []
        if refine_depth:
            batch = self.depth_refiner(batch)
        for i, image in enumerate(batch.images):
            if i == self.camera_target:
                proj.append(image)
                continue
            image_proj = depth_warp(batch, self.cameras.cameras[self.camera_target],
                                    self.cameras.cameras[i], from_=i, to_=self.camera_target)
            proj.append(image_proj)
        batch.projections = proj
        return batch

    def define_optimizers(self):
        parameters = self.cameras.named_parameters()
        lr_start = self.train_parameters['lr']
        self.optimizer = torch.optim.Adam(parameters, lr=lr_start)
        self.scheduler = ExponentialLR(self.optimizer, gamma=self.train_parameters['lr_decay'])
        if self.train_parameters['refine_depth']:
            self.depth_optimizer = torch.optim.Adam(self.depth_refiner.parameters(), lr=5e-3)#self.train_parameters['lr_after_unfreeze'])
            self.depth_scheduler = ExponentialLR(self.depth_optimizer, gamma=1 - (1 - self.train_parameters['lr_decay'])/2)

    def step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def step_depth_scheduler(self):
        if self.depth_scheduler is not None:
            self.depth_scheduler.step()

    def save_cameras_rig(self):
        cams = self.cameras.cameras.list
        extrinsics = [cam.extrinsics for cam in cams]
        rig = CameraSetup(*cams)
        rig.update_camera_relative_position(cams[0].id, extrinsics=extrinsics[0])
        for i, cam in enumerate(cams[1:]):
            rig.update_camera_relative_position(cam.id, extrinsics=extrinsics[i+1])
        rig.save(self.path, f'{self.experiment}.yaml')
        return rig


class ImageBuffer:
    def __init__(self, size: int, batch_size: int):
        self.size = size
        self.images = {}
        self.batch_size = batch_size
        self.buffer_idx = []

    def __len__(self):
        return len(self.buffer_idx)

    def __add__(self, *args):
        images, indices = args[0]
        new_indices = [i for i in indices if i not in set(self.buffer_idx)]
        if len(self) <= self.size - len(new_indices):
            self.images.update({i: image.unsqueeze(0) for i, image in zip(indices, images)})
            self.buffer_idx.extend(new_indices)
        else:
            pop_key = self.buffer_idx[:len(self) + len(new_indices) - self.size]
            for k in pop_key:
                self.images.pop(k)
            self.images.update({i: image.unsqueeze(0) for i, image in zip(indices, images)})
            self.buffer_idx = self.buffer_idx[len(self) + len(new_indices) - self.size:] + new_indices
        return self

    def clear(self):
        self.images = {}
        self.buffer_idx = []

    def get_batch(self, idx):
        """
        get images by batch
        """
        start = (idx * self.batch_size) % self.size
        end = (start + self.batch_size) % self.size
        if end <= start:
            idx = self.buffer_idx[start:] + self.buffer_idx[:end]
        else:
            idx = self.buffer_idx[start:end]
        images = torch.cat([self.images[i] for i in idx], dim=0)
        return images, idx

    def __getitem__(self, idx):
        if idx not in self.buffer_idx:
            assert idx <= len(self), "Index out of range and not in buffer"
            index = self.buffer_idx[idx]
        else:
            index = idx
        return self.images[index]

    def sort(self):
        self.buffer_idx.sort()
        self.images = {k: self.images[k] for k in self.buffer_idx}


class DataCollector:
    def __init__(self, cfg):
        self.cams = []
        self.modality = []
        for i in range(cfg.nb_cam):
            buffer_dict = {'images': ImageBuffer(cfg.buffer_size, cfg.batch_size)}
            if i == cfg.target:
                buffer_dict['depths'] = ImageBuffer(cfg.buffer_size, cfg.batch_size)
            self.__setattr__(f'{cfg.cameras_names[i]}', buffer_dict)
            self.cams.append(cfg.cameras_names[i])

    def add(self, batch: Batch):
        for i, (images, indices, cam, mod) in enumerate(zip(batch.images, batch.indices, batch.cameras, batch.modality)):
            self.modality.append(mod)
            cam_buffer = self.__getattribute__(f'{cam}')
            cam_buffer['images'] = cam_buffer['images'] + (images, indices)
            if 'depths' in cam_buffer and batch.depths is not None:
                cam_buffer['depths'] = cam_buffer['depths'] + (batch.depths[i], indices)
            if 'flows' in cam_buffer and batch.flows is not None:
                cam_buffer['flows'] = cam_buffer['flows'] + (batch.flows[i], indices)

    @property
    def size(self):
        return self.__getattribute__(f'{self.cams[0]}')['images'].size

    @property
    def buffer_size(self):
        return self.__getattribute__(f'{self.cams[0]}')['images'].buffer_size

    @property
    def isfull(self):
        return len(self) == self.__getattribute__(f'{self.cams[0]}')['images'].size

    @property
    def idx(self):
        return self.__getattribute__(f'{self.cams[0]}')['images'].buffer_idx

    def get_batch(self, idx):
        indices = []
        cameras = []
        depths = []
        flows = []
        images = []
        for cam in self.cams:
            cameras.append(cam)
            cam_buffer = self.__getattribute__(f'{cam}')
            im, idx_seq = cam_buffer['images'].get_batch(idx)
            images.append(im)
            if 'depths' in cam_buffer:
                depth, _ = cam_buffer['depths'].get_batch(idx)
                depths.append(depth)
            else:
                depths.append(None)
            indices.append(idx_seq)
        return Batch(images=images,
                     indices=indices,
                     cameras=cameras,
                     modality=self.modality,
                     depths=depths if depths else None, flows=flows if flows else None)

    def __len__(self):
        return len(self.__getattribute__(f'{self.cams[0]}')['images'])

    def sort(self):
        for cam in self.cams:
            cam_buffer = self.__getattribute__(f'{cam}')
            cam_buffer['images'].sort()
            if 'depths' in cam_buffer:
                cam_buffer['depths'].sort()

    def clear(self):
        for cam in self.cams:
            cam_buffer = self.__getattribute__(f'{cam}')
            cam_buffer['images'].clear()
            if 'depths' in cam_buffer:
                cam_buffer['depths'].clear()
