import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from XCalib.backbone.ml_depth_pro.src import depth_pro
import torch
from depth_pro.depth_pro import DepthProConfig
from einops import rearrange
from torch import nn, Tensor
from XCalib.backbone import Backbone


@dataclass
class BackboneProCfg:
    name: Literal["pro"]
    pretrained: bool
    inference_size: tuple[int, int]
    checkpoint: Path | str


class BackbonePro(Backbone[BackboneProCfg]):

    def __init__(
            self,
            cfg: BackboneProCfg,
    ) -> None:
        super().__init__(cfg)
        self.transform = None
        self.model = None
        self.buildcore()

    def buildcore(self):
        config = DepthProConfig(
            patch_encoder_preset="dinov2l16_384",
            image_encoder_preset="dinov2l16_384",
            checkpoint_uri=self.cfg.checkpoint,
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

    def forward(self, batch: Tensor, *args, **kwargs) -> dict:
        """
        :param batch: input of batched videos
        :param flows: input of batched flows
        :return: BackboneOutput
        """
        b, f, _, h, w = batch.shape
        videos = rearrange(batch, "b f c h w -> (b f) c h w")
        videos = self.transform(videos)
        out_model = self.infer_depth_memory_save(videos, 5)
        out = {'metric_depth': rearrange(out_model['depth'], "(b f) h w -> b f h w", b=b, f=f)}

        return out
