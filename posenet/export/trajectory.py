from pathlib import Path

import torch

from ThirdParty.ImagesCameras import ImageTensor
from posenet.dataset import Cameras
from posenet.misc.image_io import fig_to_image
from posenet.model.model import ModelExports
from posenet.visualization.visualizer_trajectory import generate_plot


def export_trajectory(export: ModelExports,
                      dataset: Cameras,
                      path: Path | str = None):
    indexes = sorted([int(k) for k in export.full_trajectory.keys()])
    trajectory = torch.stack([export.full_trajectory[str(id)] for id in indexes]).detach().cpu()
    poses = export.poses.detach().cpu()
    extrinsics = torch.stack([trajectory @ p.inverse() for p in poses])
    cam_names = [cam.name for cam in dataset.cameras]
    fg = generate_plot(
        extrinsics,
        trajectory,
        cam_names,
        True,
        False,
        images=None,
        display_rig=False
    )

    visualization = fig_to_image(fg)

    # Save image .png
    if path is not None:
        ImageTensor(visualization).save(str(path), name='Final trajectory', ext='png')


