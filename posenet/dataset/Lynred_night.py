from dataclasses import dataclass
from typing import Literal

import torch

from ThirdParty.ImagesCameras import Camera
from . import Cameras, CamerasCfg
from ..frame_sampler import FrameSampler


@dataclass
class LynredNightCfg(CamerasCfg):
    name: Literal["Lynred_night"]
    dataset_name: str = "Lynred_night"


class LynredNight(Cameras):

    def __init__(
            self,
            cfg: CamerasCfg,
            frame_sampler: FrameSampler,
    ) -> None:
        super().__init__(cfg, frame_sampler)

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

        # Load the camera metadata.
        image_size = camera.sensor_resolution[1], camera.sensor_resolution[0]
        image = image.resize(self.cfg.image_shape, keep_ratio=True)
        return {"videos": image,
                "indices": torch.tensor(frame_index_in_sequence),
                "frame_paths": str(camera.files[frame_index_in_sequence]),
                "image_size": image_size,
                "camera": camera.name
                }