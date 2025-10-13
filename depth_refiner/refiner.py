import torch
from torch import nn

from misc.Mytypes import Batch


class DepthRefiner(nn.Module):
    def __init__(self, target: int) -> None:
        super(DepthRefiner, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=6, padding=2, stride=2),
            nn.Sigmoid())
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=6, padding=2, stride=2),
            nn.Sigmoid())
        self.res_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Sigmoid())
        self.conv2_1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=6, padding=2, stride=2, output_padding=0),
            nn.Sigmoid())
        self.conv2_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=6, padding=2, stride=2, output_padding=0),
            nn.Sigmoid())
        self.target = target

    def forward(self, batch: Batch) -> Batch:
        depth_map, images = batch.depths[self.target], batch.images[self.target]
        depth_map_inv = 2 / (depth_map + 1e-6)
        depth_map_norm = (depth_map_inv - depth_map_inv.mean()) / (depth_map_inv.std() + 1e-6)
        x = torch.cat((depth_map_norm, images*2 - 0.5), dim=1)
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x1)
        x3 = self.res_conv(x2) + x2
        x4 = self.conv2_1(x3) + x1
        refined_depth = (self.conv2_2(x4) * 0.5 + 1) * depth_map
        batch.depths[self.target] = refined_depth
        return batch