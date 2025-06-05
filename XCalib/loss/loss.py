from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn

from ..dataset.types import Batch
from ..model.model import ModelOutput


@dataclass
class LossCfgCommon:
    enable_after: int | list
    disable_after: int | list | None
    weight: float


T = TypeVar("T", bound=LossCfgCommon)


class Loss(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg
        stepper = []
        if isinstance(self.cfg.enable_after, list):
            assert self.cfg.disable_after is not None
            if isinstance(self.cfg.disable_after, list):
                assert len(self.cfg.enable_after) == len(self.cfg.disable_after) + 1
                for i in range(len(self.cfg.disable_after)):
                    assert self.cfg.enable_after[i] <= self.cfg.disable_after[i]
                    stepper.append(np.linspace(self.cfg.enable_after[i], self.cfg.disable_after[i],
                                               self.cfg.disable_after[i] + 1 - self.cfg.enable_after[i]))
                if self.cfg.enable_after[-1] > self.cfg.disable_after[-1]:
                    self.to_the_end = True
                else:
                    self.to_the_end = False
            else:
                assert len(self.cfg.enable_after) == 2
                assert self.cfg.enable_after[0] < self.cfg.disable_after < self.cfg.enable_after[1]
                stepper.append(np.linspace(self.cfg.enable_after[0],
                                           self.cfg.disable_after,
                                           self.cfg.disable_after + 1 - self.cfg.enable_after[0]))
                stepper.append(np.array([0]))
        elif self.cfg.disable_after is not None:
            assert isinstance(self.cfg.disable_after, int)
            assert self.cfg.enable_after < self.cfg.disable_after
            stepper = np.linspace(self.cfg.enable_after, self.cfg.disable_after,
                                  self.cfg.disable_after + 1 - self.cfg.enable_after)
            self.to_the_end = False
        else:
            stepper = np.array([self.cfg.enable_after])
            self.to_the_end = True
        self.stepper = np.array(stepper).flatten()

    def forward(
            self,
            batch: Batch,
            model_output: ModelOutput,
            global_step: int,
    ) -> Float[Tensor, ""]:
        # Before the loss is enabled, don't compute the loss.
        if (global_step in self.stepper) or (global_step >= self.stepper[-1] and self.to_the_end):
            # Multiply the computed loss value by the weight.
            loss = self.compute_unweighted_loss(
                batch, model_output, global_step)
            return self.cfg.weight * loss
        else:
            return torch.tensor(0, dtype=torch.float32, device=batch.videos.device)

    @abstractmethod
    def compute_unweighted_loss(
            self,
            batch: Batch,
            model_output: ModelOutput,
            global_step: int,
    ) -> Float[Tensor, ""]:
        pass
