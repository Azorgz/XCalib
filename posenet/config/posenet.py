from dataclasses import dataclass
from pathlib import Path

from ..backbone import BackboneCfg
from ..dataset import DataModuleCfg
from posenet.flow import FlowPredictorCfg
from ..feature import FeatureCfg
from ..keypoints.keypoints import KeypointsCfg
from ..model.model_wrapper import ModelWrapperCfg
from posenet.tracking import TrackPredictorCfg
from .common import CommonCfg


@dataclass
class PoseNetCfg(CommonCfg):
    datamodule: DataModuleCfg
    depth: BackboneCfg
    tracking: TrackPredictorCfg
    flow: FlowPredictorCfg
    keypoints: KeypointsCfg
    feature: FeatureCfg
    model_wrapper: ModelWrapperCfg
    local_save_root: Path | None

