from .intrinsics import Intrinsics
from .intrinsics_regressed import IntrinsicsRegressed, IntrinsicsRegressedCfg
from .intrinsics_softmin import IntrinsicsSoftmin, IntrinsicsSoftminCfg

INTRINSICS = {"regressed": IntrinsicsRegressed,
              "softmin": IntrinsicsSoftmin,
              }

IntrinsicsCfg = IntrinsicsRegressedCfg | IntrinsicsSoftminCfg


def get_intrinsics(cfg: IntrinsicsCfg, num_cams: int) -> Intrinsics:
    return INTRINSICS[cfg.name](cfg, num_cams)
