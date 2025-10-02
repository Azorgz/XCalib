from dataclasses import dataclass
from typing import Literal

import torch
from click import Path
from jaxtyping import Float, Int64
from torch import Tensor

from ImagesCameras import Cameras
from ..misc.manipulable import Manipulable


@dataclass
class Batch(Manipulable):
    videos: Float[Tensor, "2*batch frame 3 height=_ width=_"]
    indices: Int64[Tensor, "batch frame"]
    datasets: str
    image_sizes: dict[str: tuple[int, int]]
    modality: list[Literal["Visible", "IR"]]
    cameras: list[Cameras]
    frame_paths: tuple[str | Path | None] = (None, None)
    keypoints: Manipulable | None = None
    flows: Manipulable | None = None
    tracks: Manipulable | None = None
    depths: Float[Tensor, "batch frame height width"] | None = None


def collate_batch_fn(batches: list[dict]):
    return {'videos': torch.cat([b['videos'] for b in batches], dim=0),
            'indices': torch.cat([b['indices'] for b in batches], dim=0),
            'datasets': batches[0]['datasets'],
            'cameras': batches[0]['cameras'],
            'image_sizes': batches[0]['image_sizes'],
            'frame_paths': batches[0]['frame_paths'],
            'modality': batches[0]['modality']}

