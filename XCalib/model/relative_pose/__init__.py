from .relative_pose import RelativePose
from .relative_pose_regressed import RelativePoseRegressed, RelativePoseRegressedCfg
from ...dataset.dataset_cameras import CameraBundle

RELATIVE_POSE = {"regressed": RelativePoseRegressed}

RelativePoseCfg = RelativePoseRegressedCfg


def get_relative_pose(cfg: RelativePoseCfg, num_cams: int, cameras: CameraBundle) -> RelativePose:
    return RELATIVE_POSE[cfg.name](cfg, num_cams, cameras)
