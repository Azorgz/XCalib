from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from . import Cameras, CamerasCfg
from ..frame_sampler import FrameSampler


@dataclass
class RoadSceneCfg(CamerasCfg):
    name: Literal["RoadScene"]
    image_shape: tuple[int, int] | None
    root_cameras: list[Path | str]
    cameras_name: list[str] | None
    cameras_id: list[str] | None
    dataset_name: str = "RoadScene"


class RoadScene(Cameras):

    def __init__(
            self,
            cfg: RoadSceneCfg,
            frame_sampler: FrameSampler,
    ) -> None:
        super().__init__(cfg, frame_sampler)
