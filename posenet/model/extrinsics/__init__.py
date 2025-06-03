from .extrinsics import Extrinsics
from .extrinsics_regressed import ExtrinsicsRegressedCfg, ExtrinsicsRegressed
from .extrinsics_procrustes import ExtrinsicsProcrustes, ExtrinsicsProcrustesCfg

EXTRINSICS = {"procrustes": ExtrinsicsProcrustes,
              "regressed": ExtrinsicsRegressed}

ExtrinsicsCfg = ExtrinsicsProcrustesCfg | ExtrinsicsRegressedCfg


def get_extrinsics(cfg: ExtrinsicsCfg, frames) -> Extrinsics:
    return EXTRINSICS[cfg.name](cfg, frames)
