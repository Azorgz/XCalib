from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Int64
from torch import Tensor

from .frame_sampler import FrameSampler, FrameSamplerCfg


@dataclass
class FrameSamplerSequenceCfg(FrameSamplerCfg):
    name: Literal["sequential"]
    step: int | float | None
    overlap: int | None = None


class FrameSamplerSequence(FrameSampler[FrameSamplerSequenceCfg]):

    def __init__(self, cfg: FrameSamplerSequenceCfg) -> None:
        super().__init__(cfg)
        self.start = cfg.start
        self.num_frames = cfg.num_frames
        self.overlap = cfg.overlap

        self.name = f'{int(1 / self.cfg.step) if self.cfg.step else 30}_fps_sampler_{self.overlap}_overlap'

    # def sample(
    #     self,
    #     num_frames_in_video: int,
    #     device: torch.device,
    #     frame_rate: int,
    #     iterations: int
    # ) -> Int64[Tensor, " frame"]:
    #     start = self.start or 0
    #     step = int((self.cfg.step or 1)*frame_rate)
    #     num_frames = int(min(self.num_frames * step, num_frames_in_video)/step)
    #     end = min(start + num_frames * step, num_frames_in_video)
    #     self.start = start + min(num_frames - self.overlap, 1) * step
    #
    #     return torch.arange(
    #         start,
    #         end,
    #         step,
    #         device=device,
    #         dtype=torch.int64)
    @torch.no_grad()
    def sample(
        self,
        num_frames_in_video: int,
        device: torch.device,
        frame_rate: int,
        iterations: int
    ) -> Int64[list, " frame"]:
        start = self.start or 0
        step = int((self.cfg.step*frame_rate)) or 1
        indices = torch.arange(start, num_frames_in_video, step, device=device, dtype=torch.int64)
        return list(indices.unfold(0, self.num_frames, self.num_frames-self.overlap))


        # num_frames = int(min(self.num_frames * step, num_frames_in_video)/step)
        # end = start + num_frames * step
        # overlap = min(self.overlap, (end-start)//step)
        # indices = []
        # while end <= num_frames_in_video:
        #     indices.append(torch.arange(
        #         start,
        #         end,
        #         step,
        #         device=device,
        #         dtype=torch.int64))
        #     start = max(end - overlap * step, start + 1)
        #     end = max(end + 1, start + num_frames * step)
        # return indices
