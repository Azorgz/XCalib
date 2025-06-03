from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Int64
from torch import Tensor

from .frame_sampler import FrameSampler, FrameSamplerCfg


@dataclass
class FrameSamplerRandomCfg(FrameSamplerCfg):
    name: Literal["random"]


class FrameSamplerRandom(FrameSampler[FrameSamplerRandomCfg]):

    def __init__(self, cfg: FrameSamplerRandomCfg) -> None:
        super().__init__(cfg)
        self.num_frames = cfg.num_frames
        self.start = cfg.start
        self.permutation = None

    @torch.no_grad()
    def sample(
        self,
        num_frames_in_video: int,
        device: torch.device,
        frame_rate: int,
        iterations: int
    ) -> Int64[Tensor, " frame"]:
        permutation = torch.randperm(num_frames_in_video)
        num_frames = min(self.num_frames, num_frames_in_video)
        indices = permutation.split(num_frames, 0)
        return indices
