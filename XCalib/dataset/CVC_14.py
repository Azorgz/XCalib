from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from ..third_party.ImagesCameras import Camera, ImageTensor
from . import Cameras, CamerasCfg
from .dataset_cameras import CameraBundle
from ..frame_sampler import FrameSampler


@dataclass
class CVC_14Cfg(CamerasCfg):
    name: Literal["CVC_14"]
    folder: Literal["Night", "Day"]
    image_shape: tuple[int, int] | None
    root_cameras: list[Path | str]
    cameras_name: list[str] | None
    cameras_id: list[str] | None
    dataset_name: str = "CVC_14"


class CVC_14(Cameras):

    def __init__(
            self,
            cfg: CamerasCfg,
            frame_sampler: FrameSampler,
    ) -> None:
        cfg.root_cameras = [str(cfg.root_cameras[0]) + '/' + cfg.folder + '/' + cam + '/NewTest/FramesPos/' for cam in
                            cfg.cameras_name]
        cfg.dataset_name = cfg.dataset_name + f"_{cfg.folder}"
        super().__init__(cfg, frame_sampler)
        self.path = str(cfg.root_cameras[0]) + '/' + cfg.folder

    def load_camera(self):
        cameras_name = self.cfg.cameras_name
        cameras_id = self.cfg.cameras_id
        cams = [Camera(str(root), id=cam_id, name=cam_name) for _, root, cam_id, cam_name in zip(range(self.cfg.nb_cam),
                                                                                                 self.cfg.root_cameras,
                                                                                                 cameras_id,
                                                                                                 cameras_name)]
        self.cameras = CameraBundle(cams)

    def read_image(
            self,
            camera: Camera,
            frame_index_in_sequence: int):
        # Read the image.
        if camera.modality == 'Visible':
            image = camera.__getitem__(frame_index_in_sequence)
        else:
            image = ImageTensor(camera.__getitem__(frame_index_in_sequence).to_tensor()).normalize().RGB('gray')

        # Load the camera metadata.
        image_size = camera.sensor_resolution[1], camera.sensor_resolution[0]

        return {"videos": image,
                "indices": torch.tensor(frame_index_in_sequence),
                "frame_paths": str(camera.files[frame_index_in_sequence]),
                "image_size": image_size,
                "camera": camera.name
                }
