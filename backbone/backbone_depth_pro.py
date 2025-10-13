import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from backbone.ml_depth_pro.src import depth_pro
import torch
from einops import rearrange
from torch import nn, Tensor
from backbone import Backbone
from depth_pro.depth_pro import DepthProConfig


@dataclass
class BackboneDepthProCfg:
    name: Literal["pro"]
    pretrained: bool
    checkpoint: Path | str


class BackboneDepthPro(Backbone[BackboneDepthProCfg]):

    def __init__(
            self,
            cfg: BackboneDepthProCfg,
    ) -> None:
        super().__init__(cfg)
        self.transform = None
        self.model = None
        self.buildcore()

    def buildcore(self):
        config = DepthProConfig(
            patch_encoder_preset="dinov2l16_384",
            image_encoder_preset="dinov2l16_384",
            checkpoint_uri=os.getcwd() + self.cfg.checkpoint,
            decoder_features=256,
            use_fov_head=True,
            fov_encoder_preset="dinov2l16_384",
        )
        depth_model, self.transform = depth_pro.create_model_and_transforms(config=config,
                                                                            device=torch.device('cuda:0'))
        depth_model.eval()
        self.model = depth_model.infer

    def to(self, device) -> nn.Module:
        self.device = device
        return super().to(device)

    def forward(self, images: Tensor, *args, **kwargs) -> dict:
        """
        :param batch: input of batched videos
        :param flows: input of batched flows
        :return: BackboneOutput
        """
        images_t = self.transform(images)
        out_model = self.infer_depth_memory_save(images_t, 5)
        out = out_model['depth'][:, None]

        return out
