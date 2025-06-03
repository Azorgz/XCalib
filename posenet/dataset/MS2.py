from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from . import Cameras, CamerasCfg
from ..frame_sampler import FrameSampler


@dataclass
class MS2Cfg(CamerasCfg):
    name: Literal["MS2"]
    folder: Literal["clip1", "clip2", "clip3"]
    image_shape: tuple[int, int] | None
    root_cameras: list[Path | str]
    cameras_name: list[str] | None
    cameras_id: list[str] | None
    dataset_name: str = "MS2"


folders = {'clip1': '_2021-08-06-10-59-33', 'clip2': '_2021-08-06-11-23-45', 'clip3': '_2021-08-06-11-37-46'}


class MS2(Cameras):

    def __init__(
            self,
            cfg: CamerasCfg,
            frame_sampler: FrameSampler,
    ) -> None:

        folder = f'/{cfg.root_cameras[0]}/{folders[cfg.folder]}/'
        cfg.root_cameras = [str(folder) + cam.split('_')[0] +
                            ('/img_left/' if cam.split('_')[1] == 'left' else '/img_right/') for cam in cfg.cameras_name]
        cfg.dataset_name = cfg.dataset_name + f"_{cfg.folder}"
        super().__init__(cfg, frame_sampler)
        self.path = str(cfg.root_cameras[0])
