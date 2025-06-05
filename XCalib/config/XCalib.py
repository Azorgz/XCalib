from dataclasses import dataclass
from pathlib import Path

from ..backbone import BackboneCfg
from ..dataset import DataModuleCfg
from ..model.model_wrapper import ModelWrapperCfg
from .common import CommonCfg


@dataclass
class XCalibCfg(CommonCfg):
    datamodule: DataModuleCfg
    depth: BackboneCfg
    model_wrapper: ModelWrapperCfg
    local_save_root: Path | None

