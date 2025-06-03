from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from kornia.geometry import rotation_matrix_to_quaternion, quaternion_to_rotation_matrix, Rt_to_matrix4x4, \
    deg2rad, axis_angle_to_quaternion, quaternion_to_axis_angle, axis_angle_to_rotation_matrix
from torch import Tensor
from tqdm import tqdm

from ThirdParty.ImagesCameras import ImageTensor
from ThirdParty.ImagesCameras.Metrics import Metric_ssim_tensor, Metric_nec_tensor
from . import RelativePose, RelativePoseRegressed, RelativePoseRegressedCfg
from posenet.model.relative_pose.spacesampler import SpaceSampler, SpaceSamplerCfg
from .relative_pose import RelativePoseOutput
from ..intrinsics.common import scale_intrinsics, focal_lengths_to_scaled_intrinsics
from ..projection import projection_frame_to_frame
from ...dataset.types import Batch


@dataclass
class RelativePoseGridSearchCfg:
    name: Literal["gridsearch"]
    split_search: bool
    max_batch_size: int
    initial_pose: list | None
    initial_focal_length: float | None
    softmin: bool
    last: bool
    max_depth: float | None
    sampler: SpaceSamplerCfg
    regressedCfg: RelativePoseRegressedCfg
    regression: dict | None
    enable_after: int = 0
    window: int = 100


class RelativePoseGridSearch(RelativePose[RelativePoseGridSearchCfg]):
    def __init__(self, cfg: RelativePoseGridSearchCfg, num_cams: int) -> None:
        super().__init__(cfg, num_cams)
        self.rotation = None
        self.translation = None
        self.focal_length = None
        self.ssim = None
        self.nec = None
        self.sampler = None
        self.window = {'rotation': [], 'translation': [], 'focal_length': [], 'index': []}
        if self.cfg.regression is not None:
            self.relativePoseRegressed = RelativePoseRegressed(cfg.regressedCfg, self.num_cams)
        else:
            self.relativePoseRegressed = None
        self.count = 0

    def forward(
            self,
            batch: Batch,
            global_step: int,
            params=None) -> RelativePoseOutput:
        if ((self.rotation is None or self.translation is None or self.focal_length is None or global_step == 0)
                and global_step >= self.cfg.enable_after):
            self.init_(batch.depths.device)
        return self.fct_(batch, batch.depths.device, global_step)

    @torch.no_grad()
    def init_(self, device: torch.device) -> None:
        if self.cfg.initial_pose is not None:
            pose = torch.tensor(self.cfg.initial_pose, dtype=torch.float32, device=device)
        else:
            pose = torch.eye(4, dtype=torch.float32, device=device)[None].repeat(self.num_cams-1, 1, 1)
        self.rotation = rotation_matrix_to_quaternion(pose[:, :3, :3])
        self.translation = pose[:, :3, 3]
        if self.cfg.initial_focal_length is not None:
            self.focal_length = torch.tensor(self.cfg.initial_focal_length, device=device)[None].repeat(self.num_cams, 1)
        else:
            self.focal_length = torch.tensor(1., device=device)[None].repeat(self.num_cams, 1)
        self.sampler = SpaceSampler(self.cfg.sampler, device=device)
        self.ssim = Metric_ssim_tensor(device=device)
        self.nec = Metric_nec_tensor(device=device)

    def fct_(self, batch: Batch,
             device: torch.device,
             global_step: int):
        if global_step >= self.cfg.enable_after:
            if self.relativePoseRegressed is not None and ((self.sampler.span_active['rotation'] is False and
                                                            self.sampler.span_active['translation'] is False and
                                                            self.sampler.span_active['focal_length'] is False and
                                                            self.cfg.regression['after_sampler_end']) or self.count >=
                                                           self.cfg.regression['after_step']):
                if not self.relativePoseRegressed.init:
                    # if self.cfg.last:
                    self.relativePoseRegressed.initialization(self.rotation, self.translation, self.focal_length)
                    self.relativePoseRegressed.init = True
                        #     torch.from_numpy(np.array(self.window['rotation'][-1])).to(device),
                        #     torch.from_numpy(np.array(self.window['translation'][-1])).to(device),
                        #     torch.from_numpy(np.array(self.window['scale'][-1])).to(device))


                    # else:
                    #     weights = 1 - torch.from_numpy(np.stack(self.window['index'], axis=0))
                    #     if self.cfg.softmin:
                    #         weights = F.softmin((weights - torch.min(weights, dim=0, keepdim=True).values))[:, None]
                    #
                    #     else:
                    #         weights = (weights == torch.min(weights, dim=0, keepdim=True).values)[:, None]
                    #
                    #     translation = \
                    #         (torch.from_numpy(np.stack(self.window['translation'], axis=0)).squeeze() * weights).sum(
                    #             dim=0)[None]
                    #     rotation = (torch.from_numpy(
                    #         np.stack(self.window['rotation'], axis=0)).squeeze() * weights).sum(dim=0)
                    #     scale = (torch.from_numpy(np.stack(self.window['scale'], axis=0)) * weights.squeeze()).sum()
                    #     self.relativePoseRegressed.translation.data = translation.to(device)
                    #     self.relativePoseRegressed.rotation.data = rotation.to(device)
                    #     self.relativePoseRegressed.scale.data = scale.to(device)
                    #     self.relativePoseRegressed.init = True
                else:
                    self.relativePoseRegressed.init = True
                return self.relativePoseRegressed(batch, global_step)
            else:
                self.count += 1
                return self.grid_search(batch, device, global_step)
        else:
            return self.export()

    @torch.no_grad()
    def grid_search(self, batch: Batch,
                    device: torch.device,
                    global_step: int):
        # The grid search algorithm tends to focus on x translation as it's meant for rig mounted on car
        rx, ry, rz, tx, ty, tz, focal_length = self.sampler()
        _, _, _, *size = batch.videos.shape

        intrinsics = focal_lengths_to_scaled_intrinsics(torch.tensor([1.]).to(device), size).squeeze()
        depth_dst = F.interpolate(batch.depths, size=size, mode='bilinear', align_corners=True)[0].squeeze()
        weights_far = (depth_dst - depth_dst.min() / 2) / (depth_dst.max() - depth_dst.min() / 2)
        weights_close = 1 - weights_far
        for i in range(self.num_cams - 1):
            if self.cfg.split_search:
                # GRID SEARCH FOR ROTATION
                if self.sampler.span_active['rotation'] or self.sampler.span_active['focal_length']:
                    rx0, ry0, rz0 = quaternion_to_axis_angle(self.rotation[i])
                    rx_, ry_, rz_ = torch.meshgrid([deg2rad(rx), deg2rad(ry), deg2rad(rz)])
                    quaternion = axis_angle_to_quaternion(torch.stack([(rx_ + rx0).flatten(),
                                                                       (ry_ + ry0).flatten(),
                                                                       (rz_ + rz0).flatten()], dim=-1))
                    rotations = quaternion_to_rotation_matrix(quaternion)  # N 3 3
                    translations = self.translation[i].repeat([rotations.shape[0], 1])[:, :, None]  # N 3 1
                    relative_poses = Rt_to_matrix4x4(rotations, translations)

                    # INTRINSICS batch for scaling
                    focal_lengths = focal_length + self.focal_length[i+1]
                    scaled_intrinsics = scale_intrinsics(intrinsics, focal_lengths)
                    scaled_intrinsics = scaled_intrinsics[None] if len(scaled_intrinsics.shape) == 2 else scaled_intrinsics
                    scaled_intrinsics = torch.stack([intrinsics[None].repeat(focal_lengths.shape[0], 1, 1), scaled_intrinsics],
                                                    dim=0)

                    nec = self.compute_nec(batch,
                                           relative_poses,
                                           scaled_intrinsics,
                                           size=size,
                                           weights=weights_far,
                                           from_=i+1, to_=0)
                    focal_length_idx = nec.argmax() // rotations.shape[0]
                    rotation_idx = nec.argmax() % rotations.shape[0]
                    self.focal_length[i+1] = focal_lengths[focal_length_idx]
                    rotation = rotations[rotation_idx]
                    self.rotation[i] = rotation_matrix_to_quaternion(rotation)
                else:
                    rotation = quaternion_to_rotation_matrix(self.rotation[i][None])

                # GRID SEARCH FOR TRANSLATION
                if self.sampler.span_active['translation']:
                    tx += self.translation[i, 0]  # 7 values of x translation
                    ty += self.translation[i, 1]  # 5 values of y translation
                    tz += self.translation[i, 2]  # 5 values of z translation
                    x, y, z = torch.meshgrid([tx, ty, tz])
                    translations = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1).unsqueeze(-1)
                    rotations = rotation[None].repeat(translations.shape[0], 1, 1)
                    relative_poses = Rt_to_matrix4x4(rotations, translations)
                    scaled_intrinsics = scale_intrinsics(intrinsics, self.focal_length[i+1])[None]
                    scaled_intrinsics = torch.stack([intrinsics[None], scaled_intrinsics], dim=0)
                    nec = self.compute_nec(batch,
                                           relative_poses,
                                           scaled_intrinsics,
                                           size=size,
                                           mask_in=True,
                                           weights=weights_close,
                                           from_=i+1, to_=0)
                    translation = translations[nec.argmax()]
                    self.translation[i] = translation.squeeze()
            else:
                if (self.sampler.span_active['rotation'] or self.sampler.span_active['focal_length'] or
                        self.sampler.span_active['translation']):
                    rx0, ry0, rz0 = quaternion_to_axis_angle(self.rotation[i])
                    tx += self.translation[i, 0, 0]  # 7 values of x translation
                    ty += self.translation[i, 0, 1]  # 5 values of y translation
                    tz += self.translation[i, 0, 2]  # 5 values of z translation
                    x, y, z, rx, ry, rz = torch.meshgrid(
                        [tx, ty, tz, deg2rad(rx) + rx0, deg2rad(ry) + ry0, deg2rad(rz) + rz0])
                    rotations = axis_angle_to_rotation_matrix(torch.stack([rx.flatten(),
                                                                           ry.flatten(),
                                                                           rz.flatten()], dim=-1))
                    translations = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1).unsqueeze(-1)
                    relative_poses = Rt_to_matrix4x4(rotations, translations)

                    # INTRINSICS batch for scaling
                    focal_lengths = focal_length + self.focal_length[i+1]
                    scaled_intrinsics = scale_intrinsics(intrinsics, focal_lengths)
                    scaled_intrinsics = scaled_intrinsics[None] if len(scaled_intrinsics.shape) == 2 else scaled_intrinsics
                    scaled_intrinsics = torch.stack([intrinsics[None].repeat(focal_lengths.shape[0], 1, 1), scaled_intrinsics],
                                                    dim=0)
                    nec = self.compute_nec(batch,
                                           relative_poses,
                                           scaled_intrinsics,
                                           size=size,
                                           weights=None,
                                           mask_in=True,
                                           from_=i+1, to_=0)
                    focal_length_idx = nec.argmax() // rotations.shape[0]
                    rotation_idx = nec.argmax() % rotations.shape[0]
                    self.focal_length[i+1] = focal_lengths[focal_length_idx]
                    rotation = rotations[rotation_idx]
                    self.rotation[i] = rotation_matrix_to_quaternion(rotation)
                    translation = translations[rotation_idx]
                    self.translation[i] = translation.squeeze()
        return self.fct_(batch, device, global_step)
                # else:
                #     rotation = quaternion_to_rotation_matrix(self.rotation[i])
                #     translation = self.translation[i]

        # self.window['translation'].append(translation.cpu().numpy())
        # self.window['scale'].append(self.scale.cpu().numpy())
        # self.window['rotation'].append(rotation_matrix_to_quaternion(rotation).cpu().numpy())
        # self.window['index'].append(nec.max().detach().cpu().numpy())
        # if len(self.window['index']) > self.cfg.window:
        #     idx = 0
        #     self.window['translation'].pop(idx)
        #     self.window['scale'].pop(idx)
        #     self.window['rotation'].pop(idx)
        #     self.window['index'].pop(idx)

    @torch.no_grad()
    def compute_nec(self, batch: Batch,
                    relative_poses: Tensor,
                    scaled_intrinsics: Tensor,
                    size: Tensor | tuple = (480, 640),
                    weights: Tensor | None = None,
                    mask_in: bool = False,
                    from_: int = 1, to_: int = 0):
        b, f, *_ = batch.videos.shape
        # if not self.cfg.split_search:
        #     split_size = (self.cfg.max_batch_size // (scaled_intrinsics.shape[1] * f)) or 1
        # else:
        split_size = (self.cfg.max_batch_size // scaled_intrinsics.shape[1]) or 1
        relative_poses_list = relative_poses.split(split_size, 0)
        nec = []
        images_computed = []
        if not self.cfg.split_search:
            with tqdm(desc='Grid Search', total=scaled_intrinsics.shape[1] * relative_poses.shape[0],
                      leave=True) as bar:
                images, index = projection_frame_to_frame(batch,
                                                          relative_poses_list[0],
                                                          scaled_intrinsics,
                                                          return_depths=False,
                                                          bidirectional=False, size=size,
                                                          from_=from_, to_=to_)
                images_computed = ImageTensor(images, batched=images.shape[0] > 1)
                bar.update(scaled_intrinsics.shape[1] * split_size)
                gt = ImageTensor(batch.videos[0][index].repeat(images.shape[0], 1, 1, 1), batched=images.shape[0] > 1)
                nec.append(rearrange(self.nec(images_computed, gt,
                                              weights=weights[index] if weights is not None else None),
                                     '(n_scale n_pose) -> n_scale n_pose',
                                     n_scale=scaled_intrinsics.shape[1], n_pose=relative_poses_list[0].shape[0]))
                for r in relative_poses_list[1:]:
                    images, _ = projection_frame_to_frame(batch,
                                                          r,
                                                          scaled_intrinsics,
                                                          return_depths=False,
                                                          bidirectional=False, size=size, idx=index,
                                                          from_=from_, to_=to_)
                    images_computed = ImageTensor(images, batched=images.shape[0] > 1)
                    bar.update(scaled_intrinsics.shape[1] * split_size)
                    gt = ImageTensor(batch.videos[0][index].repeat(images.shape[0], 1, 1, 1),
                                     batched=images.shape[0] > 1)
                    nec.append(rearrange(self.nec(images_computed, gt,
                                                  weights=weights[index] if weights is not None else None),
                                         '(n_scale n_pose) -> n_scale n_pose',
                                         n_scale=scaled_intrinsics.shape[1], n_pose=r.shape[0]))
        else:
            images, index = projection_frame_to_frame(batch,
                                                      relative_poses_list[0],
                                                      scaled_intrinsics,
                                                      return_depths=False,
                                                      bidirectional=False, size=size,
                                                      from_=from_, to_=to_)
            images_computed.append(ImageTensor(images, batched=images.shape[0] > 1))
            if mask_in:
                mask_in_computed = (images > 0).prod(dim=0).squeeze()
            for r in relative_poses_list[1:]:
                images, _ = projection_frame_to_frame(batch,
                                                      r,
                                                      scaled_intrinsics,
                                                      return_depths=False,
                                                      bidirectional=False, size=size, idx=index,
                                                      from_=from_, to_=to_)
                images_computed.append(ImageTensor(images, batched=images.shape[0] > 1))
                if mask_in:
                    mask_in_computed = mask_in_computed * (images > 0).prod(dim=0).squeeze()
            gt = ImageTensor(batch.videos[0][index].repeat(images.shape[0], 1, 1, 1), batched=images.shape[0] > 1)
            for r, im in zip(relative_poses_list, images_computed):
                if gt.shape[0] != im.shape[0]:
                    gt = ImageTensor(batch.videos[0][index].repeat(im.shape[0], 1, 1, 1), batched=images.shape[0] > 1)
                nec.append(rearrange(self.nec(im, gt,
                                              weights=weights[index] if weights is not None else None,
                                              mask_in=mask_in_computed if mask_in else None),
                                     '(n_scale n_pose) -> n_scale n_pose',
                                     n_scale=scaled_intrinsics.shape[1], n_pose=r.shape[0]))
        return torch.cat(nec, dim=1)
