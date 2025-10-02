import glob
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from imagesize import imagesize

from ImagesCameras import Camera
from . import Cameras, CamerasCfg
from .dataset_cameras import CameraBundle
from ..frame_sampler import FrameSampler


@dataclass
class MyDatasetCfg(CamerasCfg):
    name: Literal["MyDataset"]
    image_shape: tuple[int, int] | None
    root_cameras: list[Path | str]
    cameras_name: list[str] | None
    cameras_id: list[str] | None
    dataset_name: str = "MyDataset"


class MyDataset(list):
    """
    A Class to load unknown number of camera pair from folders.
    It will output a list a viable dataset to train on.
    """

    def __init__(
            self,
            cfg: CamerasCfg,
            frame_sampler: FrameSampler,
    ) -> None:
        self.cfg = cfg
        self.build_list()
        self.cameras = []
        self.load_camera()
        cameras = []
        for i, cam in enumerate(self.cameras):
            cfg = self.cfg
            cfg.dataset_name = self.cfg.dataset_name + f'_{"_".join([str(r) for r in cam[0].sensor_resolution])}'
            cfg.image_shape = [512, 640]  # cam[1].sensor_resolution
            cameras.append(Cameras(self.cfg, frame_sampler, cameras={'cameras': cam, 'modality': ['Visible', 'IR']}))
        super().__init__(cameras)

    def load_camera(self):
        for idx, files in enumerate(zip(self.list_rgb, self.list_ir)):
            files = list(files)
            cameras_id = [str(self.cfg.cameras_id[i]) for i in range(2)]
            cameras_name = [str(self.cfg.cameras_name[i]) for i in range(2)]
            cams = [Camera(files=file_list, id=cam_id, name=cam_name) for file_list, cam_id, cam_name in
                    zip(files, cameras_id, cameras_name)]
            cameras = CameraBundle(cams)
            self.cameras.append(cameras)

    def build_list(self):
        list_file_ir = sorted(glob.glob(str(self.cfg.root_cameras[1]) + '/*'))
        list_file_rgb = sorted(glob.glob(str(self.cfg.root_cameras[0]) + '/*'))
        im_size_ir = {}
        for i, f in enumerate(list_file_ir):
            im_size = imagesize.get(f)
            if im_size in im_size_ir.keys():
                im_size_ir[im_size].append((i, f))
            else:
                im_size_ir[im_size] = [(i, f)]
        print(f'Found {len(im_size_ir)} different image sizes in IR images')
        im_size_rgb = {}
        for i, f in enumerate(list_file_rgb):
            im_size = imagesize.get(f)
            if im_size in im_size_rgb.keys():
                im_size_rgb[im_size].append((i, f))
            else:
                im_size_rgb[im_size] = [(i, f)]
        print(f'Found {len(im_size_rgb)} different image sizes in RGB images')
        if len(im_size_ir) == 1:
            im_size_ir = list(im_size_ir.values())[0]
            if len(im_size_rgb) == 1:
                self.list_ir = [[v for _, v in im_size_ir]]
                self.list_rgb = [[v for _, v in list(im_size_rgb.values())[0]]]
            else:
                self.list_rgb = [[v for _, v in im_list] for im_list in im_size_rgb.values()]
                self.list_ir = [[im_size_ir[i][1] for i, _ in im_list] for im_list in im_size_rgb.values()]
        else:
            if len(im_size_rgb) == 1:
                im_size_rgb = list(im_size_rgb.values())
                self.list_ir = [[v for _, v in im_list] for im_list in im_size_ir.values()]
                self.list_rgb = [[im_size_rgb[i][1] for i, _ in im_list] for im_list in im_size_ir.values()]
            else:
                combinations = list(itertools.product(im_size_ir.values(), im_size_rgb.values()))
                self.list_ir = [[ir for i, ir in im_list[0] if i in [index for index, _ in im_list[1]]] for im_list in
                                combinations]
                self.list_ir = [l for l in self.list_ir if len(l) > 0]
                self.list_rgb = [[vis for i, vis in im_list[1] if i in [index for index, _ in im_list[0]]] for im_list
                                 in combinations]
                self.list_rgb = [l for l in self.list_rgb if len(l) > 0]
        print(f'{len(self.list_ir)} pairs of cameras are generated')
