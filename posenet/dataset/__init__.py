from dataclasses import dataclass

from .dataset_cameras import Cameras, CamerasCfg
from ..frame_sampler import FrameSamplerCfg, get_frame_sampler
from .types import Batch

###################################################
from .FLIR import FLIR, FLIRCfg
from .KAIST_V000 import KAIST_V000, KAIST_V000Cfg
from .MSRS import MSRS, MSRSCfg
from .CATS import CATS, CATSCfg
from .CVC_14 import CVC_14, CVC_14Cfg
from .MS2 import MS2, MS2Cfg
from .InfraParis import InfraParis, InfraParisCfg
from .LLVIP import LLVIP, LLVIPCfg
from .LLVIPClean import LLVIPClean, LLVIPCleanCfg
from .MyDataset import MyDataset, MyDatasetCfg

from .Lynred_day import LynredDay, LynredDayCfg
from .Lynred_sequence_day import LynredSequenceDay, LynredSequenceDayCfg
from .local import local, localCfg
from .RoadScene import RoadScene, RoadSceneCfg
from .Lynred_night import LynredNight, LynredNightCfg

DATASETS = {"Lynred_day": LynredDay,
            "Lynred_sequence_day": LynredSequenceDay,
            "local": local,
            "RoadScene": RoadScene,
            "Lynred_night": LynredNight,
            "InfraParis": InfraParis,
            "FLIR": FLIR,
            "KAIST_V000": KAIST_V000,
            "MSRS": MSRS,
            "CATS": CATS,
            "CVC_14": CVC_14,
            'MS2': MS2,
            'LLVIP': LLVIP,
            'LLVIPClean': LLVIPClean,
            'MyDataset': MyDataset,
            }
#"...":...   Add more datasets as needed

DatasetCfg = (LynredDayCfg | localCfg | RoadSceneCfg | LynredNightCfg | FLIRCfg | LynredSequenceDayCfg | MyDatasetCfg |
              KAIST_V000Cfg | MSRSCfg | CATSCfg | InfraParisCfg | CVC_14Cfg | MS2Cfg | LLVIPCfg | LLVIPCleanCfg)


def get_dataset(
        dataset_cfgs: list[DatasetCfg],
        frame_sampler_cfg: FrameSamplerCfg,
) -> Cameras:
    frame_sampler = get_frame_sampler(frame_sampler_cfg)
    return DATASETS[dataset_cfgs[0].name](dataset_cfgs[0], frame_sampler)


@dataclass
class DataLoaderCfg:
    seed: int | None
    rep_batch: int = 10


@dataclass
class DataModuleCfg:
    train: DataLoaderCfg
    val: DataLoaderCfg
    cache_path: str
    nb_cam: int
    shuffle: bool = False
    epochs: int = -1
    normalize_visible: bool = False
    normalize_infrared: bool = False
    equalize_visible: bool = False
    equalize_infrared: bool = False
