from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from jaxtyping import Float
from matplotlib.figure import Figure
from torch import Tensor

from ThirdParty.ImagesCameras import ImageTensor
from .drawing.points import draw_cam, draw_simple_cam, draw_rig
from .visualizer import Visualizer
from ..dataset.types import Batch
from ..export.cameras import export_to_cams
from ..model.model import Model, ModelOutput


def generate_plot(
        batch: Batch,
        poses: Float[Tensor, "batch f 4 4"],
        trajectory: Float[Tensor, "frame 3"],
        labels: list[str],
        add_cam_overlay: bool,
        simple_cam_overlay: bool,
        display_rig: bool = True,
) -> Figure:
    """
    Generate a 3D plot of the two trajectories.
    """
    b, f, _, _ = poses.shape
    fig = plt.figure(figsize=plt.figaspect(1.0), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_proj_type("ortho")
    trajectory = trajectory[..., :3, 3].cpu().numpy()
    if trajectory.shape[0] > 1:
        names = [f"cam{i+1}'s" for i in range(trajectory.shape[0])]
    else:
        names = ["Rig's"]
    colors = ['green', 'red', 'orange', 'blue', 'cyan']
    scale = 0.1

    if simple_cam_overlay:
        for i, (p, c) in enumerate(zip(poses, colors)):
            image = [ImageTensor(t) for t in batch.videos[i]]
            ax = draw_simple_cam(ax, scale=scale, color=c, transforms=p, images=[image[0]])
    elif add_cam_overlay:
        for p, c in zip(poses, colors):
            ax = draw_cam(ax, scale=scale, color=c, transforms=p)
    for n, t in zip(names, trajectory):
        xyz = rearrange(t, "f xyz -> xyz f")
        ax.plot3D(*xyz, label=f'{n} trajectory ')
    XYZ = []
    for p, c, l in zip(poses, colors, labels):
        xyz_1 = rearrange(p[:, :3, 3].cpu().numpy(), "f xyz -> xyz f")
        ax.scatter3D(*xyz_1, c=xyz_1[-1], cmap=(c + 's').capitalize(), label=l)
        XYZ.append(np.moveaxis(xyz_1, 0, -1))
    if display_rig:
        draw_rig(ax, cam1=XYZ[0], cam2=XYZ[-1])
    # Set square axes.
    # points1 = xyz_1.copy()
    # points2 = xyz_2.copy()
    # minima1 = points1.min(axis=0)
    # maxima1 = points1.max(axis=0)
    # minima2 = points2.min(axis=0)
    # maxima2 = points2.max(axis=0)
    # xlim = (min(minima1[0], minima2[0])-margin*abs(min(minima1[0], minima2[0])),
    #         max(maxima1[0], maxima2[0])+margin*abs(max(maxima1[0], maxima2[0])))
    # ylim = (min(minima1[1], minima2[1])-margin*abs(min(minima1[1], minima2[1])),
    #         max(maxima1[1], maxima2[1])+margin*abs(max(maxima1[1], maxima2[1])))
    # zlim = (min(minima1[2], minima2[2])-margin*abs(min(minima1[2], minima2[2])),
    #         max(maxima1[2], maxima2[2])+margin*abs(max(maxima1[2], maxima2[2])))
    #

    XYZ = np.concatenate(XYZ)
    xs = XYZ[:, 0]
    ys = XYZ[:, 1]
    zs = XYZ[:, 2]
    # ax.set_xlim(max(xs)+0.1, min(xs)-0.1)
    # ax.set_ylim(min(ys)-0.1, max(ys)+0.1)
    # ax.set_zlim(min(zs)-0.1, max(zs)+0.1)
    ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
    fig.legend()
    return fig


@dataclass
class VisualizerRigCfg:
    name: Literal["rig"]
    generate_plot: bool
    add_cam_overlay: bool
    simple_cam_overlay: bool


class VisualizerRig(Visualizer[VisualizerRigCfg]):
    def __init__(self, cfg: VisualizerRigCfg) -> None:
        super().__init__(cfg)

    def visualize(
            self,
            batch: Batch,
            model_output: ModelOutput,
            model: Model,
            global_step: int,
    ) -> dict[str, Float[Tensor, "3 _ _"] | Float[Tensor, ""]]:
            rig = export_to_cams(model_output, batch.cameras)
            result = {'rig_calibrated': rig}
            return result
