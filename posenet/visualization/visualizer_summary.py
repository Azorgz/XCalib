from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange
from flow_vis_torch import flow_to_color
from jaxtyping import Float
from torch import Tensor
from torch.nn.functional import interpolate

from ..dataset.types import Batch
from ..misc.cropping import resize_flow
from ..model.model import Model, ModelOutput
from ..model.projection import compute_backward_flow, sample_image_grid
from .color import apply_color_map_to_image
from .depth import color_map_depth
from .layout import add_border, add_label, hcat, vcat
from .visualizer import Visualizer


def flow_with_key(
        flow: Float[Tensor, "frame height width 2"],
) -> Float[Tensor, "3 height vis_width"]:
    _, h, w, _ = flow.shape
    length = min(h, w)
    x = torch.linspace(-1, 1, length, device=flow.device)
    y = torch.linspace(-1, 1, length, device=flow.device)
    key = torch.stack(torch.meshgrid((x, y), indexing="xy"), dim=0)
    flow = rearrange(flow, "f h w xy -> f xy h w")
    return hcat(
        *(flow_to_color(flow) / 255),
        flow_to_color(key) / 255,
    )


@dataclass
class VisualizerSummaryCfg:
    name: Literal["summary"]
    num_vis_frames: int


class VisualizerSummary(Visualizer[VisualizerSummaryCfg]):
    def visualize(
            self,
            batch: Batch,
            model_output: ModelOutput,
            model: Model,
            global_step: int,
    ) -> dict[str, Float[Tensor, "3 _ _"] | Float[Tensor, ""]]:
        # For now, only support batch size 1 for visualization.
        b, f, h, w = model_output.backward_correspondence_weights.shape
        # Pick a random interval to visualize.
        frames = torch.ones(f+1, dtype=torch.bool, device=batch.videos.device)
        pairs = torch.ones(f, dtype=torch.bool, device=batch.videos.device)
        flows = resize_flow(batch.flows, (h, w))
        if self.cfg.num_vis_frames < f+1:
            start = torch.randint(f+1 - self.cfg.num_vis_frames, (1,)).item()
            frames[:] = False
            frames[start: start + self.cfg.num_vis_frames] = True
            pairs[:] = False
            pairs[start: start + self.cfg.num_vis_frames - 1] = True
        visualization = []
        for i in range(b):
            # Color-map the ground-truth optical flow.
            # fwd_gt = flow_with_key(flows.forward[0, pairs])
            bwd_gt = flow_with_key(flows.backward[i, pairs])

            # Color-map the pose-induced optical flow.
            xy_flowed_backward = compute_backward_flow(
                model_output.surfaces[i][None],
                model_output.extrinsics[i][None],
                model_output.intrinsics[i][None],
            )
            xy_flowed_backward = xy_flowed_backward / torch.tensor([w, h], device=xy_flowed_backward.device)
            xy, _ = sample_image_grid((h, w), batch.videos.device)
            bwd_hat = flow_with_key((xy_flowed_backward - xy)[0, pairs])

            # Color-map the depth.
            depth = color_map_depth(model_output.depths[i, frames])

            # Color-map the correspondence weights.
            bwd_weights = apply_color_map_to_image(
                model_output.backward_correspondence_weights[i, pairs], "gray"
            )
            visualization.append(vcat(
                add_label(hcat(*interpolate(batch.videos[i, frames], (h, w))), "Video (Ground Truth)"),
                add_label(hcat(*interpolate(depth, (h, w))), "Depth (Predicted)"),
                add_label(bwd_gt, "Backward Flow (Ground Truth)"),
                add_label(bwd_hat, "Backward Flow (Predicted)"),
                add_label(hcat(*bwd_weights), "Backward Correspondence Weights"),
            ))

        return {"summary": add_border(hcat(*visualization))}
