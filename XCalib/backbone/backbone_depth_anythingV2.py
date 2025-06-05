import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import nn, Tensor

from XCalib.backbone import Backbone
from XCalib.backbone.Depth_anythingV2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from XCalib.backbone.backbone import BackboneOutput
from utils.misc import get_gpu_memory


@dataclass
class BackboneDepthAnythingV2Cfg:
    name: Literal["depthAnythingV2"]
    inference_size: tuple[int, int]
    checkpoint: Path
    encoder: Literal['vits', 'vitb', 'vitl', 'vitg']
    max_depth: float
    stage: Literal["indoor", "outdoor"]


class BackboneDepthAnythingV2(Backbone[BackboneDepthAnythingV2Cfg]):

    def __init__(
            self,
            cfg: BackboneDepthAnythingV2Cfg,
    ) -> None:
        super().__init__(cfg)
        self.buildcore()

    def buildcore(self):
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        path = os.getcwd() + str(self.cfg.checkpoint) + f'/depth_anything_v2_metric_{"hypersim" if self.cfg.stage == "indoor" else "vkitti"}_{self.cfg.encoder}.pth'
        depth_anything = DepthAnythingV2(**{**model_configs[self.cfg.encoder], 'max_depth': self.cfg.max_depth})
        depth_anything.load_state_dict(torch.load(path, map_location='cpu'))
        self.add_module('model', depth_anything.eval())

    def to(self, device) -> nn.Module:
        self.device = device
        return super().to(device)

    def forward(self, batch: Tensor, *args, **kwargs) -> BackboneOutput:
        """
        :param batch: input of batched videos
        :return: BackboneOutput
        """
        b, f, _, h, w = batch.shape
        videos = rearrange(batch, "b f c h w -> (b f) c h w")
        out = self.infer_depth_memory_save(videos)
        out['metric_depth'] = rearrange(out['metric_depth'], "(b f) () h w -> b f h w", b=b, f=f)
        return out

    def infer_depth_memory_save(self, videos: Float[Tensor, "batch channel height width"]) -> (
            Float)[Tensor, "batch frame height width"]:
        max_batch = max(int(np.array([m // 1 for m in get_gpu_memory()]).sum()), 1)
        videos_splited = videos.split(max_batch, dim=0)
        depths = []
        for video in videos_splited:
            depths.append(self.model.infer_tensor(video, input_size=self.cfg.inference_size))
        return {f'{k}': torch.cat([v[f'{k}'] for v in depths]) for k in depths[0].keys()}
