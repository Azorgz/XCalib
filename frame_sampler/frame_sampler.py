from abc import abstractmethod
from dataclasses import dataclass

import torch
from jaxtyping import Int64
from torch import Tensor


class FrameSampler:
    """A frame sampler picks the frames that should be sampled from a dataset's video.
    It makes sense to break the logic for frame sampling into an interface because
    pre-training and fine-tuning require different frame sampling strategies (generally,
    whole video vs. batch of video segments of same length).
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.num_frames = cfg.num_frames
        self.name = cfg.name

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


class FrameSamplerSequence(FrameSampler):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.start = cfg.start
        self.overlap = cfg.overlap
        self.step = cfg.step
        self.name = f'{int(1 / self.cfg.step) if self.cfg.step else 30}_fps_sampler_{self.overlap}_overlap'

    @torch.no_grad()
    def sample(
        self,
        num_frames_in_video: int,
        device: torch.device,
        frame_rate: int,
        iterations: int
    ) -> Int64[list, " frame"]:
        start = self.start or 0
        step = int((self.step*frame_rate)) or 1
        indices = torch.arange(start, num_frames_in_video, step, device=device, dtype=torch.int64)
        return list(indices.unfold(0, self.num_frames, self.num_frames-self.overlap))
