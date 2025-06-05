from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from ..third_party.ImagesCameras import Camera
from . import Cameras, CamerasCfg
from .dataset_cameras import CameraBundle
from ..frame_sampler import FrameSampler


@dataclass
class FLIRCfg(CamerasCfg):
    name: Literal["FLIR"]
    folder: Literal["clip1", "clip2", "clip3", "clip4", "video0", "video1", "video2", "video3"]
    image_shape: tuple[int, int] | None
    root_cameras: list[Path | str]
    cameras_name: list[str] | None
    cameras_id: list[str] | None
    dataset_name: str = "FLIR"


class FLIR(Cameras):

    def __init__(
            self,
            cfg: CamerasCfg,
            frame_sampler: FrameSampler,
    ) -> None:

        if cfg.folder in ["video0", "video1", "video2", "video3"]:
            folder = f'/{cfg.root_cameras[1]}/{cfg.folder}/'
            cfg.root_cameras = [str(folder) + cam for cam in cfg.cameras_id]
        else:
            folder = f'/{cfg.root_cameras[0]}/{cfg.folder}_night/'
            cfg.root_cameras = [str(folder) + cam for cam in cfg.cameras_name]
        cfg.dataset_name = cfg.dataset_name + f"_{cfg.folder}"
        super().__init__(cfg, frame_sampler)
        self.path = str(cfg.root_cameras[0])

    def load_camera(self):
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
        if "video" in self.cfg.folder:
            cameras_id = list(reversed(cameras_id))
            cameras_name = list(reversed(cameras_name))
            self.cfg.root_cameras = list(reversed(self.cfg.root_cameras))
        cams = [Camera(str(root), id=cam_id, name=cam_name) for _, root, cam_id, cam_name in zip(range(self.cfg.nb_cam),
                                                                                                 self.cfg.root_cameras,
                                                                                                 cameras_id,
                                                                                                 cameras_name)]
        self.cameras = CameraBundle(cams)  #if self.cfg.folder == 'video0' else CameraBundle(reversed(cams))
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
            image = camera.__getitem__(frame_index_in_sequence)
        else:
            if self.cfg.normalize_infrared:
                image = camera.__getitem__(frame_index_in_sequence).histo_equalization().RGB('gray')
            else:
                image = camera.__getitem__(frame_index_in_sequence).RGB('gray')

        image_size = camera.sensor_resolution[1], camera.sensor_resolution[0]
        image = image.resize(self.cfg.image_shape)

        return {"videos": image,
                "indices": torch.tensor(frame_index_in_sequence),
                "frame_paths": str(camera.files[frame_index_in_sequence]),
                "image_size": image_size,
                "camera": camera.name
                }
