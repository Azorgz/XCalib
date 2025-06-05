import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from einops import rearrange
from torch import nn, Tensor
from XCalib.backbone.Depth_anything.metric_depth.zoedepth.models.builder import build_model
from XCalib.backbone.Depth_anything.metric_depth.zoedepth.utils.config import get_config
from XCalib.backbone import Backbone
from XCalib.backbone.backbone import BackboneOutput
from utils.misc import configure_parser


@dataclass
class BackboneZoeCfg:
    name: Literal["zoe"]
    pretrained: bool
    inference_size: tuple[int, int]
    checkpoint_base: Path
    checkpoint_metric: Path


class BackboneZoe(Backbone[BackboneZoeCfg]):

    def __init__(
            self,
            cfg: BackboneZoeCfg,
    ) -> None:
        super().__init__(cfg)
        self.buildcore()

    def buildcore(self):
        sys.path.append(os.getcwd() + '/XCalib/backbone/Depth_anything/metric_depth')
        parser = get_config('zoedepth', "infer")
        config = configure_parser(parser,
                                  None,
                                  path_config=os.getcwd() + '/XCalib/backbone/Depth_anything/config_Depth_anything.yml',
                                  dict_vars=None)
        config.pretrained_resource = config.path_checkpoint
        # Depth model
        depth_model = build_model(config)
        depth_model.requires_grad = False
        self.add_module('model', depth_model)

    def to(self, device) -> nn.Module:
        self.device = device
        return super().to(device)

    def forward(self, batch: Tensor, *args, **kwargs) -> BackboneOutput:
        """
        :param batch: input of batched videos
        :param flows: input of batched flows
        :return: BackboneOutput
        """
        b, f, _, h, w = batch.shape
        videos = rearrange(batch, "b f c h w -> (b f) c h w")
        out = self.infer_depth_memory_save(videos)
        out['metric_depth'] = rearrange(out['metric_depth'], "(b f) () h w -> b f h w", b=b, f=f)
        return out
