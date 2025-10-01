import numpy as np
import torch
from ImagesCameras import ImageTensor, CameraSetup
from datasets import tqdm
from torch import nn
from torch.nn.functional import interpolate
from torch.optim.lr_scheduler import LambdaLR

from XCalib2.Mytypes import Batch
from XCalib2.backbone import get_backbone
from XCalib2.frame_sampler import get_frame_sampler
from XCalib2.loss import get_losses
from XCalib2.model.cameras import Cameras
from XCalib2.model.spatial_transformer import depth_warp


class XCalib(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.experiment = cfg.name_experiment
        self.scheduler = None
        self.optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg.model
        self.output_path = cfg.output + cfg.data.name
        self.train_parameters = self.cfg['train']
        self.depthModel = get_backbone(cfg.model['depth'])
        self.LossModel = get_losses(self.train_parameters['loss'], self.cfg['target'])
        self.cameras = Cameras(cfg.data, get_frame_sampler(cfg.frame_sampler))
        self.number_cameras = len(self.cameras.cameras)
        self.images = DataCollector(cfg)
        self.camera_target = self.cfg['target']
        self.define_optimizers()

    def to(self, device: torch.device):
        super().to(device)

    def buffer_batch(self, batch: Batch):
        self.images.add(batch)

    def load_images(self, idx):
        return self.cameras[idx]

    @torch.no_grad()
    def compute_depths(self, batch: Batch):
        with torch.no_grad():
            depths = self.depthModel(batch.images[self.camera_target])
        batch.depths = [depths if i == self.camera_target else None for i in range(self.number_cameras)]
        return batch

    @torch.no_grad()
    def compute_flows(self, batch: Batch):
        flows = []
        for i in range(self.number_cameras):
            if i == self.camera_target:
                flows.append(None)
            else:
                h, w = batch.images[i].shape[-2:]
                img1, img2 = (interpolate(batch.images[self.camera_target], (h // 2, w // 2)),
                              interpolate(batch.images[i], (h // 2, w // 2)))
                flow = self.FlowModel(img1, img2)['flow']
                flows.append(interpolate(flow, (h, w)) * 2)
        batch.flows = flows
        return batch

    def optimize_parameters(self):
        waitbar = tqdm(total=self.images.size, desc='Precomputing images')
        while not self.images.isfull:
            idx_batch = int(np.random.randint(0, len(self.cameras) - 1, 1))
            batch = self.load_images(idx_batch)
            batch = self.compute_depths(batch)
            # batch = self.compute_flows(batch)
            self.buffer_batch(batch)
            waitbar.update(self.train_parameters['batch_size'])
        waitbar.close()

        optimization_step = 0
        waitbar = tqdm(total=len(self.images)//self.train_parameters['batch_size']*self.train_parameters['epochs'],
                       desc=f'Optimization of the parameters epoch {0}')
        self.cameras.initial_freeze()
        screen = None
        for e in range(self.train_parameters['epochs']):
            waitbar.desc = f'Optimization of the parameters epoch {e}'
            if e == self.train_parameters['unfreeze']:
                self.cameras.unfreeze()
            for j in range(len(self.images)//self.train_parameters['batch_size']):
                optimization_step += 1
                # self.cameras.update_parameters()
                batch = self.images.get_batch(j)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # Wrap images
                batch = self.wrap_frame_to_target(batch)
                # Computes losses
                loss = self.LossModel(batch, e)
                loss.backward()
                self.optimizer.step()
                self.step_scheduler()
                waitbar.update(1)

            if not e % 2:
                img = ImageTensor(batch.projections[0] / 2 + batch.projections[1] / 2)
                if screen is None:
                    screen = img.show(name=f'Projection at epoch {e}', opencv=True, asyncr=True)
                else:
                    screen.update(img, name=f'Projection at epoch {e}')
        self.cameras.freeze()

    def wrap_all(self):
        old_setup = self.cameras.cfg
        self.cameras.cfg.normalize_visible = self.cameras.cfg.equalize_visible = False
        self.cameras.cfg.normalize_infrared = self.cameras.cfg.equalize_infrared = False
        for batch in self.cameras:
            batch = self.compute_depths(batch)
            batch = self.wrap_frame_to_target(batch)
            for i in range(self.number_cameras):
                if i != self.camera_target:
                    projections = batch.projections[i].split(1, 0)
                    for p, path in zip(projections, batch.frame_paths[i]):
                        name = path.split('/')[-1].split('.')[0]
                        ImageTensor(p).save(self.output_path, name=name, depth=8)
        self.cameras.cfg = old_setup

    def wrap_frame_to_target(self, batch: Batch):
        proj = []
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
        if isinstance(self.train_parameters['lr'], list):
            lr_start, lr_end = eval(self.train_parameters['lr'][0]), eval(self.train_parameters['lr'][1])
        else:
            lr_start = eval(self.train_parameters['lr'])
            lr_end = None

        # optim_params = [parameters, lr= lr_start]
        self.optimizer = torch.optim.Adam(parameters, lr=lr_start)

        if lr_end is not None:
            self.scheduler = LambdaLR(self.optimizer,
                                      lr_lambda=lambda step: lr_end / lr_start + (1 - lr_end / lr_start) / (
                                                  1.0015 ** step))
        else:
            self.scheduler = None

    def step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def save_cameras_rig(self):
        cams = self.cameras.cameras.list
        rig = CameraSetup(*cams)
        rig.update_camera_relative_position(cams[0].id, extrinsics=cams[0].extrinsics)
        for cam, extrinsic in zip(cams[1:], cams[1:].extrinsics):
            rig.update_camera_relative_position(cam.id, extrinsics=extrinsic)
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
        for i in range(cfg.data.nb_cam):
            buffer_dict = {'images': ImageBuffer(cfg.model['buffer_size'], cfg.model['train']['batch_size'])}
            if i == cfg.model['target']:
                buffer_dict['depths'] = ImageBuffer(cfg.model['buffer_size'], cfg.model['train']['batch_size'])
            # else:
            #     buffer_dict['flows'] = ImageBuffer(cfg.model['buffer_size'], cfg.model['train']['batch_size'])
            self.__setattr__(f'{cfg.data.cameras_name[i]}', buffer_dict)
            self.cams.append(cfg.data.cameras_name[i])

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
            # if 'flows' in cam_buffer:
            #     flow, _ = cam_buffer['flows'].get_batch(idx)
            #     flows.append(flow)
            # else:
            #     flows.append(None)
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
