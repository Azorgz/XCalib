from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, Literal

import torch
from jaxtyping import Float
from torch import Tensor, nn

from posenet.dataset.types import Batch

T = TypeVar("T")


@dataclass
class SpaceSamplerCfg:
    name: Literal["spacesampler"]
    translation: dict
    rotation: dict
    focal_length: dict
    f_lambda: list


class SpaceSampler:

    def __init__(self, cfg: SpaceSamplerCfg, device) -> None:
        self.nb_called: int = 0
        self.cfg = cfg
        self.device = device
        self.span_active = {'translation': True, 'focal_length': True, 'rotation': True}
        # Translation
        tx_span = torch.arange(0, cfg.translation['x'][1] * 2 + 1) / (cfg.translation['x'][1] * 2) - 0.5 if \
            cfg.translation['x'][1] != 0 else torch.tensor([0.])
        self.tx_span = (tx_span * cfg.translation['x'][0]).to(device)
        ty_span = torch.arange(0, cfg.translation['y'][1] * 2 + 1) / (cfg.translation['y'][1] * 2) - 0.5 if \
            cfg.translation['y'][1] != 0 else torch.tensor([0.])
        self.ty_span = (ty_span * cfg.translation['y'][0]).to(device)
        tz_span = torch.arange(0, cfg.translation['z'][1] * 2 + 1) / (cfg.translation['z'][1] * 2) - 0.5 if \
            cfg.translation['z'][1] != 0 else torch.tensor([0.])
        self.tz_span = (tz_span * cfg.translation['z'][0]).to(device)

        # Rotation
        rx_span = torch.arange(0, cfg.rotation['x'][1] * 2 + 1) / (cfg.rotation['x'][1] * 2) - 0.5 if cfg.rotation['x'][
                                                                                                          1] != 0 else torch.tensor(
            [0.])
        self.rx_span = (rx_span * cfg.rotation['x'][0]).to(device)
        ry_span = torch.arange(0, cfg.rotation['y'][1] * 2 + 1) / (cfg.rotation['y'][1] * 2) - 0.5 if cfg.rotation['y'][
                                                                                                          1] != 0 else torch.tensor(
            [0.])
        self.ry_span = (ry_span * cfg.rotation['y'][0]).to(device)
        rz_span = torch.arange(0, cfg.rotation['z'][1] * 2 + 1) / (cfg.rotation['z'][1] * 2) - 0.5 if cfg.rotation['z'][
                                                                                                          1] != 0 else torch.tensor(
            [0.])
        self.rz_span = (rz_span * cfg.rotation['z'][0]).to(device)

        # focal_length
        focal_length_span = torch.arange(0, cfg.focal_length['values'][1] * 2 + 1) / (cfg.focal_length['values'][1] * 2) - 0.5 if (
                cfg.focal_length['values'][1] != 0) else torch.tensor([0.])
        self.focal_length_span = (focal_length_span * cfg.focal_length['values'][0]).to(device)

        # Lambda
        self.f_lambda = lambda i: cfg.f_lambda[1] + cfg.f_lambda[0] / (i + 1)

    def __call__(self):
        rx, ry, rz = self.rotation()
        tx, ty, tz = self.translation()
        focal_length = self.focal_length()
        self.update_span(self.nb_called)
        self.nb_called += 1
        return rx, ry, rz, tx, ty, tz, focal_length

    def rotation(self):
        return self.rx_span, self.ry_span, self.rz_span

    def translation(self):
        return self.tx_span, self.ty_span, self.tz_span

    def focal_length(self):
        return self.focal_length_span

    def update_span(self, i):
        l = 1 - self.f_lambda(i)
        if self.span_active['rotation']:
            self.rx_span = self.limit_rotation(self.rx_span * l)
            self.ry_span = self.limit_rotation(self.ry_span * l)
            self.rz_span = self.limit_rotation(self.rz_span * l)
            if len(self.rx_span) + len(self.ry_span) + len(self.rz_span) == 3:
                self.span_active['rotation'] = False
        if self.span_active['translation']:
            self.tx_span = self.limit_translation(self.tx_span * l)
            self.ty_span = self.limit_translation(self.ty_span * l)
            self.tz_span = self.limit_translation(self.tz_span * l)
            if len(self.tx_span) + len(self.ty_span) + len(self.tz_span) == 3:
                self.span_active['translation'] = False
        if self.span_active['focal_length']:
            self.focal_length_span = self.limit_focal_length(self.focal_length_span * l)
            if len(self.focal_length_span) == 1:
                self.span_active['focal_length'] = False

    def limit_translation(self, vec):
        if len(vec) > 1:
            if vec[1] - vec[0] < self.cfg.translation['minimum_delta']:
                return torch.tensor([0.]).to(self.device)
            else:
                return vec
        else:
            return vec

    def limit_rotation(self, vec):
        if len(vec) > 1:
            if vec[1] - vec[0] < self.cfg.rotation['minimum_delta']:
                return torch.tensor([0.]).to(self.device)
            else:
                return vec
        else:
            return vec

    def limit_focal_length(self, vec):
        if len(vec) > 1:
            if vec[1] - vec[0] < self.cfg.focal_length['minimum_delta']:
                return torch.tensor([0.]).to(self.device)
            else:
                return vec
        else:
            return vec
