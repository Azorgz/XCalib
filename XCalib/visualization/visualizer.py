from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import Batch
from ..model.model import Model, ModelOutput

T = TypeVar("T")


class Visualizer(ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def visualize(
        self,
        batch: Batch,
        model_output: ModelOutput,
        model: Model,
        global_step: int,
    ) -> dict[str, Float[Tensor, "3 _ _"] | Float[Tensor, ""]]:
        pass
