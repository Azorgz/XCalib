from dataclasses import dataclass
from typing import Literal, Type, TypeVar

from omegaconf import DictConfig
from ..dataset import DatasetCfg
from ..frame_sampler import FrameSamplerCfg
from ..loss import LossCfg
from ..model.model import ModelCfg
from ..visualization import VisualizerCfg
from .tools import get_typed_config, separate_multiple_defaults


@dataclass
class WandbCfg:
    project: str
    mode: Literal["online", "offline", "disabled"]
    name: str | None
    group: str | None
    tags: list[str] | None
    key: str | None


@dataclass
class TrainerCfg:
    val_check_interval: int
    max_steps: int


@dataclass
class CommonCfg:
    wandb: WandbCfg
    trainer: TrainerCfg
    dataset: list[DatasetCfg]
    frame_sampler: FrameSamplerCfg
    model: ModelCfg
    loss: list[LossCfg]
    visualizer: list[VisualizerCfg]
    image_shape: list[int, int] | None

T = TypeVar("T")


def get_typed_root_config(cfg_dict: DictConfig, cfg_type: Type[T]) -> T:
    return get_typed_config(
        cfg_type,
        cfg_dict,
        {
            list[DatasetCfg]: separate_multiple_defaults(DatasetCfg),
            list[LossCfg]: separate_multiple_defaults(LossCfg),
            list[VisualizerCfg]: separate_multiple_defaults(VisualizerCfg),
        },
    )
