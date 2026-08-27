import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn, Tensor
from torch.nn.functional import interpolate
from .Depth_Anything_V2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from model.backbone import Backbone

ROOT_DIR = Path(__file__).parent.parent.parent

@dataclass
class BackboneDepthAnythingCfg:
    name: Literal["DepthAnything"]
    encoder: Literal["vits", "vitb", "vitl"] = "vits"
    input_size: int = 518
    scene: str = "outdoor"  # 'indoor' for indoor model, 'outdoor' for outdoor model


class BackboneDepthAnything(Backbone[BackboneDepthAnythingCfg]):

    def __init__(
            self,
            cfg: BackboneDepthAnythingCfg,
    ) -> None:
        super().__init__(cfg)
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }
        encoder = cfg.encoder
        dataset = 'vkitti' if cfg.scene == 'outdoor' else 'hypersim'
        max_depth = 80  # 20 for indoor model, 80 for outdoor model

        model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
        model.load_state_dict(
            torch.load(f'{ROOT_DIR}/model/backbone/Depth_Anything_V2/metric_depth/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
        self.depth_model = model.eval()
        self.model = lambda x: {'depth': self.depth_model(x)[:, None]}

    def _get_opt(self):
        parser = argparse.ArgumentParser(description='Video Depth Anything')
        parser.add_argument('--input_video', type=str, default='./assets/example_videos/davis_rollercoaster.mp4')
        parser.add_argument('--output_dir', type=str, default='./outputs')
        parser.add_argument('--input_size', type=int, default=518)
        parser.add_argument('--max_res', type=int, default=1280)
        parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
        parser.add_argument('--max_len', type=int, default=-1,
                            help='maximum length of the input video, -1 means no limit')
        parser.add_argument('--target_fps', type=int, default=-1,
                            help='target fps of the input video, -1 means the original fps')
        parser.add_argument('--metric', default=True, help='use metric model')
        parser.add_argument('--fp32', action='store_true',
                            help='model infer with torch.float32, default is torch.float16')
        parser.add_argument('--grayscale', action='store_true', help='do not apply colorful palette')
        parser.add_argument('--save_npz', action='store_true', help='save depths as npz')
        parser.add_argument('--save_exr', action='store_true', help='save depths as exr')
        parser.add_argument('--focal-length-x', default=470.4, type=float,
                            help='Focal length along the x-axis.')
        parser.add_argument('--focal-length-y', default=470.4, type=float,
                            help='Focal length along the y-axis.')
        return parser.parse_args()

    def to(self, device) -> nn.Module:
        self.device = device
        return super().to(device)

    def forward(self, images: Tensor, *args, **kwargs) -> Tensor:
        """
        :param images: input of batched images of shape (b, 3, h, w)
        """
        b, c, h, w = images.shape
        images = interpolate(images, size=(self.cfg.input_size, self.cfg.input_size), mode='bilinear', align_corners=False)
        out = self.infer_depth_memory_save(images)
        out = interpolate(out['depth'], size=(h, w), mode='bilinear', align_corners=False)
        return out
