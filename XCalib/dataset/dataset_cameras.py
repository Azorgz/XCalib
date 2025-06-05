from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset, default_collate

from ..third_party.ImagesCameras import Camera, ImageTensor
from ..frame_sampler import FrameSampler


class CameraBundle(list):

    def __init__(self, cameras: list[Camera]):
        super().__init__(cameras)
        self.__dict__.update({f'{camera.name}': camera for camera in cameras})


@dataclass
class CamerasCfg:
    name: Literal["cameras"]
    image_shape: tuple[int, int] | None
    root_cameras: list[Path | str]
    cameras_name: list[str] | None
    cameras_id: list[str] | None
    stage: Literal['indoor', 'outdoor']
    dataset_name: str = "CamerasDataset"
    nb_cam: int = 2
    load_frame_paths: bool = True
    normalize_visible: bool = False
    normalize_infrared: bool = False
    equalize_visible: bool = False
    equalize_infrared: bool = False
    folder: str = None


class Cameras(Dataset):
    cfg: CamerasCfg
    frame_sampler: FrameSampler
    cameras: CameraBundle

    def __init__(
            self,
            cfg: CamerasCfg,
            frame_sampler: FrameSampler,
            *args,
            cameras=None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.name = self.cfg.dataset_name if self.cfg.dataset_name else str(self.cfg.name)
        self.frame_sampler = frame_sampler
        self.modality = []
        self.stage = cfg.stage
        if cameras is None:
            self.load_camera(*args)
        else:
            self.cameras = cameras['cameras']
            self.modality = cameras['modality']
        self.indices = self.frame_sampler.sample(len(self.cameras[0].files),
                                                 torch.device("cpu"),
                                                 self.cameras[0].data.fps,
                                                 0)
        self.path = self.cfg.root_cameras[0]

    def load_camera(self, *args):
        if self.cfg.cameras_name:
            if len(self.cfg.cameras_name) >= len(self.cfg.root_cameras):
                cameras_name = self.cfg.cameras_name
            else:
                cameras_name = self.cfg.cameras_name
                cameras_name.extend(
                    [str(root).split('/')[-1] for root in self.cfg.root_cameras[len(self.cfg.cameras_name):]])
        else:
            cameras_name = [str(root).split('/')[-1] for root in self.cfg.root_cameras]
        if self.cfg.cameras_id:
            if len(self.cfg.cameras_id) == len(self.cfg.cameras_name):
                cameras_id = self.cfg.cameras_id
            else:
                cameras_id = self.cfg.cameras_id
                cameras_id.extend([name for name in self.cfg.cameras_name[len(self.cfg.cameras_id):]])
        else:
            cameras_id = cameras_name
        cams = [Camera(str(root), id=cam_id, name=cam_name) for _, root, cam_id, cam_name in zip(range(self.cfg.nb_cam),
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

    def __len__(self):
        return len(self.indices)

    def read_image(
            self,
            camera: Camera,
            frame_index_in_sequence: int):
        # Read the image.
        if camera.modality == 'Visible':
            image = camera.__getitem__(frame_index_in_sequence).RGB()
            if self.cfg.equalize_visible:
                image = image.histo_equalization()
            elif self.cfg.normalize_visible:
                image = image.normalize()
        else:
            image = camera.__getitem__(frame_index_in_sequence).RGB('gray')
            if self.cfg.equalize_infrared:
                image = image.histo_equalization()
            elif self.cfg.normalize_infrared:
                image = image.normalize()

        # Load the camera metadata.
        image_size = camera.sensor_resolution[1], camera.sensor_resolution[0]
        image = image.resize(self.cfg.image_shape)
        return {"videos": image,
                "indices": torch.tensor(frame_index_in_sequence),
                "frame_paths": str(camera.files[frame_index_in_sequence]),
                "image_size": image_size,
                "camera": camera.name
                }

    def __getitem__(self, index: int):
        result = {}
        indices = self.indices[index]
        items = [default_collate([self.read_image(cam, index.item()) for index in indices]) for cam in self.cameras]
        result.update(
            {f'image_sizes': {item['camera'][0]: (int(item['image_size'][1][0]), int(item['image_size'][0][0]))
                              for i, item in enumerate(items)}})
        result['indices'] = items[0]['indices']
        result["videos"] = torch.stack(
            [ImageTensor(item["videos"], batched=len(indices)) for item in items], dim=0)
        if self.cfg.load_frame_paths:
            result['frame_paths'] = [item['frame_paths'] for item in items]
        result["datasets"] = self.name
        result["modality"] = self.modality
        result["cameras"] = self.cameras
        return result
