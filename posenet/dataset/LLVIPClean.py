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
class LLVIPCleanCfg(CamerasCfg):
    name: Literal["LLVIPClean"]
    root_cameras: list[Path]
    dataset_name: str = "LLVIPClean"
    folder: None


class LLVIPClean(Cameras):

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
