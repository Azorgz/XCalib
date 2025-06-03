from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from . import Cameras, CamerasCfg
from ..frame_sampler import FrameSampler


@dataclass
class LynredSequenceDayCfg(CamerasCfg):
    name: Literal["Lynred_sequence_day"]
    root_cameras: str | Path
    dataset_name: str = "Lynred_sequence"
    folder: Literal[1, 2, 3, 4]


class LynredSequenceDay(Cameras):

    def __init__(
            self,
            cfg: CamerasCfg,
            frame_sampler: FrameSampler,
    ) -> None:
        folder = f'/{str(cfg.root_cameras)}/sequence_{cfg.folder}/sorted/'
        cfg.root_cameras = [str(folder) + cam for cam in cfg.cameras_name]

        cfg.dataset_name = cfg.dataset_name + f"_{cfg.folder}"
        self.path = str(f'/{str(cfg.root_cameras)}/sequence_{cfg.folder}/')
        super().__init__(cfg, frame_sampler)
