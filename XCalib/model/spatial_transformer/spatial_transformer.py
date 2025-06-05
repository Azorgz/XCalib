import itertools
from abc import abstractmethod
from dataclasses import dataclass
from typing import TypeVar

import torch
from jaxtyping import Float
from torch import Tensor, nn

from ..projection import projection_frame_to_frame, focal_lengths_to_scaled_intrinsics
from ..relative_pose.relative_pose import RelativePoseOutput
from ...dataset.types import Batch

T = TypeVar("T")


@dataclass
class SpatialTransformerOutput:
    images: Float[Tensor, "b f c h w"]
    depths: Float[Tensor, "b f 1 h w"] | None


@dataclass
class SpatialTransformerCfg:
    cross_projection: int
    bidirectional: bool
    return_depth: bool


class SpatialTransformer(nn.Module):

    def __init__(self, cfg: SpatialTransformerCfg, image_shape) -> None:
        super().__init__()
        self.cfg = cfg
        self.image_shape = image_shape

    @abstractmethod
    def forward(
            self,
            batch: Batch,
            relative_pose: RelativePoseOutput,
            initial_pose: Float[Tensor, "batch 4 4"],
            global_step: int,
    ) -> SpatialTransformerOutput:
        b, f, _, h, w = batch.videos.shape
        if self.cfg.cross_projection and self.cfg.cross_projection <= global_step:
            list_idx = itertools.combinations([i for i in range(b)], 2)
        else:
            list_idx = [(0, i + 1) for i in range(b - 1)]
        res = {'images': [], 'depths': []}
        for idx in list_idx:
            to_, from_ = idx
            temp = self.projection(batch, relative_pose, initial_pose, from_, to_)
            res['images'].append(temp['images'])
            res['depths'].append(temp['depths'])
        res['images'] = torch.cat(res['images'], 0)
        if self.cfg.return_depth:
            res['depths'] = torch.cat(res['depths'], 0)
        else:
            res['depths'] = None
        return SpatialTransformerOutput(**res)

    def projection(self, batch, relative_pose, initial_pose, from_: int, to_: int):
        # The grid search algorithm tends to focus on x translation as it's meant for rig mounted on car

        b, f, *_ = batch.videos.shape
        # INTRINSICS batch for scaling
        intrinsics1 = focal_lengths_to_scaled_intrinsics(relative_pose.focal_length[to_],
                                                         batch.image_sizes[batch.cameras[to_].name],
                                                         center=relative_pose.center[to_] if relative_pose.center is not None else None,
                                                         shear=relative_pose.shear[to_] if relative_pose.shear is not None else None).squeeze()
        intrinsics2 = focal_lengths_to_scaled_intrinsics(relative_pose.focal_length[from_],
                                                         batch.image_sizes[batch.cameras[from_].name],
                                                         center=relative_pose.center[from_] if relative_pose.center is not None else None,
                                                         shear=relative_pose.shear[from_] if relative_pose.shear is not None else None).squeeze()

        intrinsics = torch.cat([intrinsics1[None, None], intrinsics2[None, None]], dim=0)
        relative_pose_ = (initial_pose[to_].inverse() @ initial_pose[from_])[None]
        *images, idx = projection_frame_to_frame(batch,
                                                 relative_pose_,
                                                 intrinsics,
                                                 return_depths=self.cfg.return_depth,
                                                 bidirectional=self.cfg.bidirectional,
                                                 idx='all',
                                                 size=self.image_shape,
                                                 from_=from_, to_=to_, from_file=False)
        if self.cfg.bidirectional:
            if self.cfg.return_depth:
                images1_2, images2_1, depths1_2, depths2_1 = images
                return {'images': torch.stack([images1_2, images2_1]), 'depths': torch.stack([depths1_2, depths2_1])}
            else:
                images1_2, images2_1 = images
                return {'images': torch.stack([images1_2, images2_1]), 'depths': None}
        else:
            if self.cfg.return_depth:
                images2_1, depths2_1 = images
                return {'images': images2_1[None], 'depths': depths2_1[None]}
            else:
                images2_1 = images[0]
                return {'images': images2_1[None], 'depths': None}