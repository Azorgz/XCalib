from pathlib import Path

from ImagesCameras import Camera, CameraSetup

from ..model.projection import focal_lengths_to_scaled_intrinsics
from XCalib.dataset.dataset_cameras import CameraBundle
from XCalib.model.model import ModelExports


def export_to_cams(export: ModelExports,
                   cameras: CameraBundle,
                   path: Path | str = None,
                   idx: int = None,
                   name: str = 'rig_calibrated'):
    focals = [f.detach().cpu() for f in export.relative_pose.focal_length]
    if export.relative_pose.center is not None:
        centers = [c.detach().cpu() for c in export.relative_pose.center]
    else:
        centers = [None] * len(focals)
    if export.relative_pose.shear is not None:
        shear = [s.detach().cpu() for s in export.relative_pose.shear]
    else:
        shear = [None] * len(focals)
    cams = []
    for cam, f, c, s in zip(cameras, focals, centers, shear):
        intrinsics = focal_lengths_to_scaled_intrinsics(f, (cam.sensor_resolution[0],
                                                            cam.sensor_resolution[1]),
                                                        center=c, shear=s, fourbyfour=True)
        cams.append(Camera(cam.path, files=cam.data.generator, name=cam.name, id=cam.id, intrinsics=intrinsics.numpy(), HFOV=45 * f.numpy()))

    rig = CameraSetup(*cams)
    rig.update_camera_relative_position(cams[0].id, extrinsics=export.poses[0][None].inverse())
    for cam, pose in zip(cams[1:], export.poses[1:]):
        rig.update_camera_relative_position(cam.id, extrinsics=pose[None].inverse())
    if path is not None:
        path.mkdir(exist_ok=True, parents=True)
        rig.save(path, f'{name}{f"_{idx}" if idx is not None else ""}.yaml')
    else:
        return rig
