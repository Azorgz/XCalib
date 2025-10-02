import os

import numpy as np
import yaml
from ImagesCameras.Cameras import LearnableCamera
from torch import nn
from torch.nn import ModuleList
from torch.utils.data import Dataset, default_collate
from ImagesCameras import Camera, CameraSetup
from frame_sampler import FrameSampler

import torch
from misc.Mytypes import CamerasCfg, Batch


class CameraBundle(nn.Module):

    def __init__(self, cameras: list[LearnableCamera]):
        nn.Module.__init__(self)
        self.cameras = ModuleList(cameras)
        self.__dict__.update({f'{camera.name}': camera for camera in cameras})
        self.list = [camera for camera in cameras]

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx: int) -> LearnableCamera:
        return self.list[idx]


class Cameras(Dataset, nn.Module):
    cfg: CamerasCfg
    cameras: CameraBundle

    def __init__(
            self,
            cfg: CamerasCfg,
            frame_sampler: FrameSampler,
            *args,
            cameras=None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.name = self.cfg.name
        self.frame_sampler = frame_sampler
        self.modality = []
        self.stage = cfg.stage
        if cfg.from_file is not None:
            self.from_file(cfg.from_file)
        elif cameras is None:
            self.load_camera(*args)
        else:
            self.cameras = cameras['cameras']
            self.modality = cameras['modality']
        self.indices = self.frame_sampler.sample(len(self.cameras[0].files),
                                                 torch.device("cpu"),
                                                 self.cameras[0].data.fps, 0)
        self.cfg.root_cameras = [root.replace('local::', os.getcwd()) for root in self.cfg.root_cameras]
        self.path = self.cfg.root_cameras[0]

    def from_file(self, path: str | os.PathLike):
        setup = CameraSetup(from_file=path)
        from_setup = [(cam.path, cam.intrinsics[0], cam.extrinsics[0, :, :] if i != 0 else setup.base2Ref[0, 0, :, :], cam.id, cam.name) for i, cam in enumerate(setup.cameras.values())]
        cams = [LearnableCamera(v[0], intrinsics=v[1], extrinsics=v[2], id=v[3], name=v[4]) for v in from_setup]
        self.cameras = CameraBundle(cams)
        self.modality = [cam.modality for cam in cams]

    def load_camera(self, *args):
        assert len(self.cfg.cameras_name) == len(
            self.cfg.root_cameras), "The number of camera names must be equal to the number of camera folder."
        cams = [LearnableCamera(str(root), id=cam_id, name=cam_name) for _, root, cam_id, cam_name in
                zip(range(self.cfg.nb_cam),
                    self.cfg.root_cameras,
                    self.cfg.cameras_name,
                    self.cfg.cameras_name)]
        self.cameras = CameraBundle(cams)
        for cam, cam_name in zip(self.cameras, self.cfg.cameras_name):
            if 'rgb' in cam_name.lower():
                self.modality.append('Visible')
            elif 'ir' in cam_name.lower():
                self.modality.append('IR')
            else:
                self.modality.append(cam.modality if cam.modality == 'Visible' else 'IR')

    def __len__(self):
        return len(self.indices)

    @property
    def total_frames(self):
        return len(self.cameras[0].files)

    def update_parameters(self):
        for cam in self.cameras:
            cam.update_parameters()

    def initial_freeze(self):
        for cam in self.cameras:
            # freeze z translation
            cam.freeze_z = True
            # freeze z translation
            cam.freeze_rz = True
            # freeze c
            cam.freeze_c = True
            # freeze skew
            cam.freeze_skew = True

    def freeze(self):
        for cam in self.cameras:
            cam.freeze()

    def unfreeze(self):
        for cam in self.cameras:
            cam.unfreeze()

    def read_image(
            self,
            camera: Camera,
            frame_index_in_sequence: int):
        # Read the image.
        if camera.modality == 'Visible':
            image = camera.__getitem__(frame_index_in_sequence).RGB()
            if self.cfg.equalize_visible:
                image = image.histo_equalization()
            elif self.cfg.normalize_visible:
                image = image.normalize()
            modality = 'Visible'
        else:
            image = camera.__getitem__(frame_index_in_sequence).RGB('gray')
            if self.cfg.equalize_infrared:
                image = image.histo_equalization()
            elif self.cfg.normalize_infrared:
                image = image.normalize()
            modality = 'IR'

        return {"image": image,
                "frame_path": str(camera.files[frame_index_in_sequence]),
                "camera": camera.name,
                "modality": modality}

    def __getitem__(self, idx: int):
        if idx >= len(self):
            idx = idx - len(self)
        indices = self.indices[idx].numpy().tolist()
        items = [default_collate([self.read_image(cam, index) for index in indices]) for cam in self.cameras]
        images = [item['image'][:, 0] for item in items]
        paths = [item['frame_path'] for item in items]
        cameras = [item['camera'][0] for item in items]
        modality = [item['modality'][0] for item in items]

        result = {'images': images,
                  'cameras': cameras,
                  'frame_paths': paths,
                  'indices': [indices] * len(self.cameras),
                  'modality': modality}
        return Batch(**result)

    def load_images(self, idx: int | list | np.ndarray):
        if not isinstance(idx, np.ndarray):
            idx = np.array(idx)
        indices = [i % self.total_frames for i in idx]
        items = [default_collate([self.read_image(cam, index) for index in indices]) for cam in self.cameras]
        images = [item['image'][:, 0] for item in items]
        paths = [item['frame_path'] for item in items]
        cameras = [item['camera'][0] for item in items]
        modality = [item['modality'][0] for item in items]

        result = {'images': images,
                  'cameras': cameras,
                  'frame_paths': paths,
                  'indices': [indices] * len(self.cameras),
                  'modality': modality}
        return Batch(**result)
