from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from ImagesCameras import Camera, ImageTensor
from . import Cameras, CamerasCfg
from ..frame_sampler import FrameSampler


@dataclass
class KAIST_V000Cfg(CamerasCfg):
    name: Literal["KAIST_V000"]
    image_shape: tuple[int, int] | None
    root_cameras: list[Path | str]
    cameras_name: list[str] | None
    cameras_id: list[str] | None
    dataset_name: str = "KAIST_V000"


class KAIST_V000(Cameras):

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