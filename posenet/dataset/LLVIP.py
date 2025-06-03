import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from ThirdParty.ImagesCameras import Camera
from utils.misc import name_generator
from . import Cameras, CamerasCfg
from .dataset_cameras import CameraBundle
from ..frame_sampler import FrameSampler


@dataclass
class LLVIPCfg(CamerasCfg):
    name: Literal["LLVIP"]
    root_cameras: str | Path
    dataset_name: str = "LLVIP"
    folder: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]


class LLVIP(Cameras):

    def __init__(
            self,
            cfg: CamerasCfg,
            frame_sampler: FrameSampler,
    ) -> None:
        self.cfg = cfg
        path_ir = str(self.cfg.root_cameras) + '/LLVIP/infrared/'
        list_ir = [path_ir + 'train/' + p for p in os.listdir(path_ir + 'train/') if int(p[:2]) == self.cfg.folder]
        list_ir.extend([path_ir + 'test/' + p for p in os.listdir(path_ir + 'test/') if int(p[:2]) == self.cfg.folder])
        list_ir = sorted(list_ir, key=lambda x: int(x.split('/')[-1].split('.')[0]))

        path_rgb = str(self.cfg.root_cameras) + '/LLVIP_raw_images/'
        list_rgb = [path_rgb + 'train/visible/' + p for p in os.listdir(path_rgb + 'train/visible/') if
                    int(p[:2]) == self.cfg.folder]
        list_rgb.extend([path_rgb + 'test/visible/' + p for p in os.listdir(path_rgb + 'test/visible/') if
                         int(p[:2]) == self.cfg.folder])
        list_rgb = sorted(list_rgb, key=lambda x: int(x.split('/')[-1].split('.')[0]))

        num_frame_rgb = [int(f.split('/')[-1].split('.')[0][2:]) for f in list_rgb]
        num_frame_ir = [int(f.split('/')[-1].split('.')[0][2:]) for f in list_ir]

        list_rgb = [path for path, num in zip(list_rgb, num_frame_rgb) if num in num_frame_ir]
        list_ir = [path for path, num in zip(list_ir, num_frame_ir) if num in num_frame_rgb]

        list_im = [list_rgb, list_ir] if self.cfg.cameras_name[0] == 'visible' else [list_ir, list_rgb]

        super().__init__(cfg, frame_sampler, *list_im)
        self.path = str(cfg.root_cameras)
        self.name = f"LLVIP_{self.cfg.folder}"

    def load_camera(self, *args):
        if self.cfg.cameras_name:
            if len(self.cfg.cameras_name) >= len(self.cfg.root_cameras):
                cameras_name = self.cfg.cameras_name
            else:
                cameras_name = self.cfg.cameras_name
                cameras_name.extend(
                    [str(root).split('/')[-1] for root in self.cfg.root_cameras[len(self.cfg.cameras_name):]])
        else:
            cameras_name = [str(root).split('/')[-1] for root in self.cfg.root_cameras]
        if self.cfg.cameras_id:
            if len(self.cfg.cameras_id) == len(self.cfg.cameras_name):
                cameras_id = self.cfg.cameras_id
            else:
                cameras_id = self.cfg.cameras_id
                cameras_id.extend([name for name in self.cfg.cameras_name[len(self.cfg.cameras_id):]])
        else:
            cameras_id = cameras_name

        cams = [Camera(files=root, id=cam_id, name=cam_name) for _, root, cam_id, cam_name in
                zip(range(self.cfg.nb_cam),
                    args,
                    cameras_id,
                    cameras_name)]
        self.cameras = CameraBundle(cams)
        for cam, cam_id, cam_name in zip(self.cameras, cameras_id, cameras_name):
            if 'rgb' in cam_name.lower() or 'rgb' in cam_id.lower():
                self.modality.append('Visible')
            elif 'ir' in cam_name.lower() or 'ir' in cam_id.lower():
                self.modality.append('IR')
            else:
                self.modality.append(cam.modality if cam.modality == 'Visible' else 'IR')

    def read_image(
            self,
            camera: Camera,
            frame_index_in_sequence: int):
        # Read the image.
        if camera.modality == 'Visible':
            image = camera.__getitem__(frame_index_in_sequence).RGB()
            if self.cfg.normalize_visible:
                image = image.normalize()
        else:
            image = camera.__getitem__(frame_index_in_sequence).RGB()
            if self.cfg.normalize_infrared:
                image = image.normalize()

        image += 1e-6
        image /= image.max()
        # Load the camera metadata.
        image_size = camera.sensor_resolution[1], camera.sensor_resolution[0]
        image = image.resize(self.cfg.image_shape)
        return {"videos": image,
                "indices": torch.tensor(frame_index_in_sequence),
                "frame_paths": str(camera.files[frame_index_in_sequence]),
                "image_size": image_size,
                "camera": camera.name
                }