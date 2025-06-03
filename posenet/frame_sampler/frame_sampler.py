from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from jaxtyping import Int64
from torch import Tensor

T = TypeVar("T")


@dataclass
class FrameSamplerCfg:
    num_frames: int | None
    start: int | float | None
    name: str


class FrameSampler(ABC, Generic[T]):
    """A frame sampler picks the frames that should be sampled from a dataset's video.
    It makes sense to break the logic for frame sampling into an interface because
    pre-training and fine-tuning require different frame sampling strategies (generally,
    whole video vs. batch of video segments of same length).
    """

    cfg: T

    def __init__(self, cfg: T) -> None:
        self.cfg = cfg
        self.name = cfg.name

    @abstractmethod
    def sample(
        self,
        num_frames_in_video: int,
        device: torch.device,
        frame_rate: int,
        iterations: int
    ) -> Int64[Tensor, " frame"]:  # frame indices
        pass
