from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from . import Cameras, CamerasCfg
from ..frame_sampler import FrameSampler


@dataclass
class LynredDayCfg(CamerasCfg):
    name: Literal["Lynred_day"]
    dataset_name: str = "Lynred_day"


class LynredDay(Cameras):

    def __init__(
            self,
            cfg: CamerasCfg,
            frame_sampler: FrameSampler,
    ) -> None:
        super().__init__(cfg, frame_sampler)
