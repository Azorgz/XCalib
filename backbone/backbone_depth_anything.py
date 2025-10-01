import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from einops import rearrange
from torch import nn, Tensor
from torch.nn.functional import interpolate

from ..backbone.Depth_anything.metric_depth.zoedepth.models.builder import build_model
from ..backbone.Depth_anything.metric_depth.zoedepth.utils.config import get_config
from ..backbone import Backbone
from utils.misc import configure_parser


@dataclass
class BackboneZoeCfg:
    name: Literal["zoe"]
    pretrained: bool
    max_depth: float
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
        sys.path.append(os.getcwd() + '/backbone/Depth_anything/metric_depth')
        parser = get_config('zoedepth', "infer")
        config = configure_parser(parser,
                                  None,
                                  path_config=os.getcwd() + '/backbone/Depth_anything/config_Depth_anything.yml',
                                  dict_vars=None)
        config.pretrained_resource = config.path_checkpoint
        # Depth model
        depth_model = build_model(config)
        depth_model.requires_grad = False
        self.add_module('model', depth_model)

    def to(self, device) -> nn.Module:
        self.device = device
        return super().to(device)

    def forward(self, images: Tensor, *args, **kwargs) -> Tensor:
        """
        :param images: input of batched images of shape (b, 3, h, w)
        """
        b, c, h, w = images.shape
        out = self.infer_depth_memory_save(images)
        out = interpolate(out['metric_depth'], size=(h, w), mode='bilinear', align_corners=False)
        # out['metric_depth'] = rearrange(out['metric_depth'], "(b f) () h w -> b f h w", b=b, f=f)
        return out
