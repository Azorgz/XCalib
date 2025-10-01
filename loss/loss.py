from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn
from XCalib2.Mytypes import Batch


@dataclass
class LossCfgCommon:
    enable_after: int | list
    disable_after: int | list | None
    weight: float


T = TypeVar("T", bound=LossCfgCommon)


class Loss(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T, targets: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.targets = targets
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
            epoch: int,
    ) -> Float[Tensor, ""]:
        # Before the loss is enabled, don't compute the loss.
        if (epoch in self.stepper) or (epoch >= self.stepper[-1] and self.to_the_end):
            # Multiply the computed loss value by the weight.
            loss = self.compute_unweighted_loss(
                batch, epoch)
            return self.cfg.weight * loss
        else:
            return torch.tensor(0, dtype=torch.float32, device=batch.images[0][0].device)

    @abstractmethod
    def compute_unweighted_loss(
            self,
            batch: Batch,
            global_step: int,
    ) -> Float[Tensor, ""]:
        pass
