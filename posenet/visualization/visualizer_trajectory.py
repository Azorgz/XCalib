from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from jaxtyping import Float
from matplotlib.figure import Figure
from torch import Tensor

from ThirdParty.ImagesCameras import ImageTensor
from .drawing.points import draw_cam, draw_simple_cam, draw_rig
from .layout import add_border
from .visualizer import Visualizer
from ..dataset.types import Batch
from ..misc.image_io import fig_to_image
from ..model.model import Model, ModelOutput


def generate_plot(
        poses: Float[Tensor, "batch f 4 4"],
        trajectory: Float[Tensor, "frame 3"],
        labels: list[str],
        add_cam_overlay: bool,
        simple_cam_overlay: bool,
        images: Float[Tensor, "batch f c h w"] = None,
        trajectory_projected: bool = True,
        display_rig: bool = False,
) -> Figure:
    """
    Generate a 3D plot of the two trajectories.
    """
    b, f, _, _ = poses.shape
    fig = plt.figure(figsize=plt.figaspect(1.0), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_proj_type("ortho")
    traj = trajectory[None, :, :3, 3].cpu().numpy()
    ymin = traj[..., 1].min() - 0.5
    xmin = traj[..., 0].min() - 0.5
    if traj.shape[0] > 1:
        names = [f"cam{i+1}'s" for i in range(traj.shape[0])]
    else:
        names = ["Rig's"]
    colors = ['green', 'red', 'orange', 'blue', 'cyan']
    # scale = [np.sqrt(((trajectory[:, i+1] - trajectory[:, i])**2).sum()) for i in range(f-1)]
    # if len(scale) > 1:
    #     scale = min(*scale)
    # else:
    #     scale = scale[0]
    if add_cam_overlay and simple_cam_overlay:
        assert images is not None
        for i, (p, c) in enumerate(zip(poses, colors)):
            image = [ImageTensor(im) for im in images[i]]
            ax = draw_simple_cam(ax, color=c, transforms=p, images=image)
    elif add_cam_overlay:
        for p, c in zip(poses, colors):
            ax = draw_cam(ax, color=c, transforms=p)
    for n, t in zip(names, traj):
        xyz = rearrange(t, "f xyz -> xyz f")[[0, 2, 1], :]
        ax.plot3D(*xyz, label=f'{n} trajectory')
    XYZ = []
    for p, c, l in zip(poses, colors, labels):
        xyz_1 = rearrange(p[:, :3, 3].cpu().numpy(), "f xyz -> xyz f")[[0, 2, 1], :]
        ax.scatter3D(*xyz_1, c=xyz_1[-2], cmap=(c + 's').capitalize(), label=l)
        # if trajectory_projected:
        #     ax.plot(xyz_1[0], xyz_1[2], zdir='x', color=c)
        #     ax.plot(xyz_1[1], xyz_1[2], zdir='z', color=c)
        XYZ.append(np.moveaxis(xyz_1, 0, -1))
    if display_rig:
        draw_rig(ax, trajectory)

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
    # XYZ = np.concatenate(XYZ)
    # xs = XYZ[:, 0]
    # ys = XYZ[:, 2]
    # zs = XYZ[:, 1]
    # xy = max(np.ptp(xs), np.ptp(zs))*1.1
    # ax.set_box_aspect((xy, xy, np.ptp(ys)*1.1))
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.zaxis.set_inverted(True)
    ax.set_zlabel("Y")
    fig.legend()
    ax.set_xticks([])
    # ax.set_yticks([])
    ax.set_zticks([])
    return fig


@dataclass
class VisualizerTrajectoryCfg:
    name: Literal["trajectory"]
    generate_plot: bool
    add_cam_overlay: bool
    simple_cam_overlay: bool
    # This is used to dump ATEs for the paper.
    ate_save_path: Path | None
    display_rig: bool = False


class VisualizerTrajectory(Visualizer[VisualizerTrajectoryCfg]):
    def __init__(self, cfg: VisualizerTrajectoryCfg) -> None:
        super().__init__(cfg)
        self.ates = []

    def visualize(
            self,
            batch: Batch,
            model_output: ModelOutput,
            model: Model,
            global_step: int,
    ) -> dict[str, Float[Tensor, "3 _ _"] | Float[Tensor, ""]]:
        poses = model_output.extrinsics
        trajectory = model_output.trajectory
        # Visualize the trajectory.
        if self.cfg.simple_cam_overlay:
            images = batch.videos
        else:
            images = None
        fg = generate_plot(
            poses,
            trajectory,
            [cam.name for cam in batch.cameras],
            self.cfg.add_cam_overlay,
            self.cfg.simple_cam_overlay,
            images= images,
            display_rig=self.cfg.display_rig
        )
        visualization = fig_to_image(fg)
        plt.close(fg)
        result = {"trajectory": add_border(visualization)}
        return result
