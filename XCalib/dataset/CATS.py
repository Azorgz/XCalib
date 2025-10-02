from dataclasses import dataclass
from typing import Literal

from ImagesCameras import Camera
from . import Cameras, CamerasCfg
from .dataset_cameras import CameraBundle
from ..frame_sampler import FrameSampler


@dataclass
class CATSCfg(CamerasCfg):
    name: Literal["CATS"]
    folder: (Literal[
                 "CARS", "COURTYARD", "CREEK", "GARDEN", "HOUSE", "ISE", "PATIO", "SHED", "BOOKS", "ELECTRONICS", "HALLOWEEN", "MATERIALS", "MISC", "PLANTS", "STATUES", "STORAGE_ROOM", "TOOLS", "TOYS"] |
             list[Literal[
                 "CARS", "COURTYARD", "CREEK", "GARDEN", "HOUSE", "ISE", "PATIO", "SHED", "BOOKS", "ELECTRONICS", "HALLOWEEN", "MATERIALS", "MISC", "PLANTS", "STATUES", "STORAGE_ROOM", "TOOLS", "TOYS"]])
    dataset_name: str = "CATS"


class CATS(Cameras):

    def __init__(
            self,
            cfg: CamerasCfg,
            frame_sampler: FrameSampler,
    ) -> None:
        if not isinstance(cfg.folder, list):
            cfg.folder = [cfg.folder]
        folders = [f'/OUTDOOR/{f}/' if f in ["CARS", "COURTYARD", "CREEK", "GARDEN", "HOUSE", "ISE", "PATIO", "SHED"]
                   else f'/INDOOR/{f}/' for f in cfg.folder]
        path = str(cfg.root_cameras[0])
        cfg.root_cameras = [
            [str(cfg.root_cameras[0]) + folder + ("All_wo_dark/" if folder.split('/')[1] == 'INDOOR' else "All/") + cam for
             folder in folders] for cam in cfg.cameras_name]
        cfg.dataset_name = cfg.dataset_name + f"_{'_'.join(cfg.folder)}" if len(
            cfg.folder) < 3 else cfg.dataset_name + f"_{cfg.folder[0]}_group"
        super().__init__(cfg, frame_sampler)
        self.path = path + '/OUTDOOR/'

    def load_camera(self):
        cameras_name = self.cfg.cameras_name
        cameras_id = self.cfg.cameras_id
        cams = [Camera(root, id=cam_id, name=cam_name) for _, root, cam_id, cam_name in zip(range(self.cfg.nb_cam),
                                                                                                 self.cfg.root_cameras,
                                                                                                 cameras_id,
                                                                                                 cameras_name)]
        self.cameras = CameraBundle(cams)
        for cam, cam_id, cam_name in zip(self.cameras, cameras_id, cameras_name):
            if 'rgb' in cam_name.lower() or 'rgb' in cam_id.lower():
                self.modality.append('Visible')
            elif 'ir' in cam_name.lower() or 'ir' in cam_id.lower():
                self.modality.append('IR')
            else:
                self.modality.append(cam.modality if cam.modality == 'Visible' else 'IR')
