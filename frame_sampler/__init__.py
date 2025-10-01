from typing import Any

from XCalib2.frame_sampler.frame_sampler import FrameSampler, FrameSamplerSequence

FRAME_SAMPLER = {
    "random": FrameSampler,
    "sequential": FrameSamplerSequence,
}


def get_frame_sampler(cfg) -> FrameSampler:
    return FRAME_SAMPLER[cfg.name](cfg)
