from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor

from ImagesCameras import ImageTensor
from .layout import add_border, hcat, vcat
from .visualizer import Visualizer
from ..dataset.types import Batch
from ..model.model import Model, ModelOutput
from ..model.projection import projection_frame_to_frame, focal_lengths_to_scaled_intrinsics


@dataclass
class VisualizerSpatialTransformerCfg:
    name: Literal["spatialtransformer"]
    num_vis_frames: int
    cam1Tocam2: bool


class VisualizerSpatialTransformer(Visualizer[VisualizerSpatialTransformerCfg]):
    @torch.no_grad()
    def visualize(
            self,
            batch: Batch,
            model_output: ModelOutput,
            model: Model,
            global_step: int,
    ) -> dict[str, Float[Tensor, "3 _ _"] | Float[Tensor, ""]]:
        # For now, only support batch size 1 for visualization.
        b, f, _, h, w = batch.videos.shape
        device = model_output.depths.device
        # Camera matrix from intrinsics
        idx = torch.randperm(len(batch.frame_paths[0]))[: self.cfg.num_vis_frames]
        size = batch.videos.shape[-2:]
        visualization = []
        for i in range(b-1):
            intrinsics = torch.stack(
                [focal_lengths_to_scaled_intrinsics(model_output.relative_pose.focal_length[0].detach().clone(),
                                                    batch.image_sizes[batch.cameras[0].name],
                                                    model_output.relative_pose.center[0].detach().clone() if model_output.relative_pose.center is not None else None,
                                                    model_output.relative_pose.shear[0].detach().clone() if model_output.relative_pose.shear is not None else None),
                 focal_lengths_to_scaled_intrinsics(model_output.relative_pose.focal_length[i+1].detach().clone(),
                                                    batch.image_sizes[batch.cameras[i+1].name],
                                                    model_output.relative_pose.center[i+1].detach().clone() if model_output.relative_pose.center is not None else None,
                                                    model_output.relative_pose.shear[i+1].detach().clone() if model_output.relative_pose.shear is not None else None)])
            images_reg, _ = projection_frame_to_frame(batch, model_output.relative_pose.Rt[i][None], intrinsics,
                                                      return_depths=False,
                                                      bidirectional=False,
                                                      size=size, idx=idx,
                                                      from_=i+1, to_=0)
            images_reg = (ImageTensor(images_reg, batched=images_reg.shape[0] > 1).RGB('gray')
                          .resize(batch.image_sizes[batch.cameras[0].name]))

            images_1 = ImageTensor(batch.videos[0, idx], device=device).resize(batch.image_sizes[batch.cameras[0].name])
            images_2 = ImageTensor(batch.videos[i+1, idx], device=device).resize(batch.image_sizes[batch.cameras[0].name])
            images_fus_reg = images_1 / 2 + images_reg / 2
            images_fus = images_1 / 2 + images_2 / 2

            for i, (im2, im_fus, im1, im_fus_reg, im_reg) in enumerate(
                    zip(images_2, images_fus, images_1, images_fus_reg, images_reg)):
                visualization.append(hcat(im2, im_fus, im1, im_fus_reg, im_reg))
        return {"images projected": add_border(vcat(*visualization))}
