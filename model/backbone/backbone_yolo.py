from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn, Tensor
from torch.nn.functional import interpolate
from ultralytics import YOLO

from model.backbone import Backbone


@dataclass
class BackboneYoloCfg:
    name: Literal["yolo"]
    model: str


class BackboneYolo(Backbone[BackboneYoloCfg]):

    def __init__(
            self,
            cfg: BackboneYoloCfg,
    ) -> None:
        super().__init__(cfg)
        self.model_depth = YOLO(cfg.model)
        self.model = lambda x: {'depth': torch.stack([y.depth.data[None] for y in self.model_depth(x, verbose=False)], dim=0)}

    def to(self, device) -> nn.Module:
        self.device = device
        return super().to(device)

    def forward(self, images: Tensor, *args, **kwargs) -> Tensor:
        """
        :param batch: input of batched videos
        :param flows: input of batched flows
        """
        h, w = images.shape[-2:]
        if h % 32 != 0 or w % 32 != 0:
            images = interpolate(images, size=(h - h % 32, w - w % 32), mode='bilinear', align_corners=False)
        out_model = self.infer_depth_memory_save(images, 0.5)['depth']
        out_model = interpolate(out_model, size=(h, w), mode='bilinear', align_corners=False)
        return out_model
