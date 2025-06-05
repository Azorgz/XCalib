from typing import Any

from .frame_sampler import FrameSampler
from .frame_sampler_random import FrameSamplerRandom, FrameSamplerRandomCfg
from .frame_sampler_sequence import FrameSamplerSequence, FrameSamplerSequenceCfg

FRAME_SAMPLER = {
    "random": FrameSamplerRandom,
    "sequential": FrameSamplerSequence,
}

FrameSamplerCfg = FrameSamplerRandomCfg | FrameSamplerSequenceCfg


def get_frame_sampler(cfg: FrameSamplerCfg) -> FrameSampler[Any]:
    return FRAME_SAMPLER[cfg.name](cfg)
