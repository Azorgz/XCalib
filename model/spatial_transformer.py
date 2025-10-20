from typing import Literal
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Float
from kornia.geometry import transform_points, normalize_pixel_coordinates, compose_transformations, \
    inverse_transformation, depth_to_3d_v2, convert_points_from_homogeneous
from torch import Tensor

from misc.Mytypes import Batch


def depth_warp(batch: Batch, camera_target, camera_src, from_=1, to_=0):
    intrinsics = torch.cat([camera_target.intrinsics, camera_src.intrinsics], dim=0)
    dst_trans_src: Tensor = compose_transformations(
        inverse_transformation(camera_target.extrinsics), camera_src.extrinsics)
    return projection_frame_to_frame(batch,
                                     dst_trans_src,
                                     intrinsics,
                                     from_=from_, to_=to_)


def depth_to_3d(depths: Float[Tensor, "*batch h w"],
                intrinsics: Float[Tensor, "*batch 3 3"]):
    *batch, h, w = depths.shape
    b = np.prod(batch)
    depths = depths.reshape(b, h, w)
    intrinsics = intrinsics.repeat(b, 1, 1)
    surfaces = depth_to_3d_v2(depths, intrinsics)
    return surfaces.reshape(*batch, h, w, 3)


def project_points(point_3d: torch.Tensor, camera_matrix: torch.Tensor) -> torch.Tensor:
    point_2d_norm: torch.Tensor = convert_points_from_homogeneous(point_3d)

    # unpack coordinates
    x_coord: Tensor = point_2d_norm[..., 0]
    y_coord: Tensor = point_2d_norm[..., 1]

    # unpack intrinsics
    fx: Tensor = camera_matrix[..., 0, 0]
    fy: Tensor = camera_matrix[..., 1, 1]
    cx: Tensor = camera_matrix[..., 0, 2]
    cy: Tensor = camera_matrix[..., 1, 2]

    # apply intrinsics ans return
    if fx.ndim == 0:
        u_coord: Tensor = x_coord * fx + cx
        v_coord: Tensor = y_coord * fy + cy
    else:
        u_coord: Tensor = x_coord * fx[:, None, None] + cx[:, None, None]
        v_coord: Tensor = y_coord * fy[:, None, None] + cy[:, None, None]

    return torch.stack([u_coord, v_coord], dim=-1)


def projection_frame_to_frame(batch: Batch,
                              relative_poses: Float[Tensor, "n_pose 4 4"],
                              intrinsics: Float[Tensor, "batch n_scale 3 3"],
                              from_: int = 1, to_: int = 0) -> Tensor | tuple:
    b, c, h, w = batch.images[0].shape
    device = batch.images[0].device
    #  extract images from batch
    images = batch.images[from_]

    depths = batch.depths[to_]

    # Camera matrix from intrinsics
    cam_dst, cam_src = intrinsics[..., :3, :3].split(1, 0)
    image_shape = images.shape[-2:]

    # convert depth to 3d points
    points_3d_dst: Tensor = depth_to_3d(depths, cam_dst)[:, 0]  # Bx3xHxW

    # apply transformation to the 3d points
    points_3d_dst_trans = transform_points(relative_poses, points_3d_dst)  # BxHxWx3

    points_2d_dst_trans: Tensor = torch.stack(
        [project_points(points_3d_dst_trans[i], cam_src) for i in range(b)])  # BxHxWx2
    points_2d_dst_trans_norm: Tensor = normalize_pixel_coordinates(points_2d_dst_trans, *image_shape).to(
        depths.dtype)[:, 0]  # BxHxWx2
    images2_1 = F.grid_sample(images, points_2d_dst_trans_norm, align_corners=True)
    return images2_1

#
# @dataclass
# class SpatialTransformerOutput:
#     images: Float[Tensor, "b f c h w"]
#     depths: Float[Tensor, "b f 1 h w"] | None
#
#
# @dataclass
# class SpatialTransformerCfg:
#     cross_projection: int
#     bidirectional: bool
#     return_depth: bool


# class SpatialTransformer(nn.Module):
#
#     def __init__(self, cfg: SpatialTransformerCfg, image_shape) -> None:
#         super().__init__()
#         self.cfg = cfg
#         self.image_shape = image_shape
#
#     @abstractmethod
#     def forward(
#             self,
#             batch: Batch,
#             relative_pose: RelativePoseOutput,
#             initial_pose: Float[Tensor, "batch 4 4"],
#             global_step: int,
#     ) -> SpatialTransformerOutput:
#         b, f, _, h, w = batch.videos.shape
#         if self.cfg.cross_projection and self.cfg.cross_projection <= global_step:
#             list_idx = itertools.combinations([i for i in range(b)], 2)
#         else:
#             list_idx = [(0, i + 1) for i in range(b - 1)]
#         res = {'images': [], 'depths': []}
#         for idx in list_idx:
#             to_, from_ = idx
#             temp = self.projection(batch, relative_pose, initial_pose, from_, to_)
#             res['images'].append(temp['images'])
#             res['depths'].append(temp['depths'])
#         res['images'] = torch.cat(res['images'], 0)
#         if self.cfg.return_depth:
#             res['depths'] = torch.cat(res['depths'], 0)
#         else:
#             res['depths'] = None
#         return SpatialTransformerOutput(**res)
#
#     def projection(self, batch, relative_pose, initial_pose, from_: int, to_: int):
#         # The grid search algorithm tends to focus on x translation as it's meant for rig mounted on car
#
#         b, f, *_ = batch.videos.shape
#         # INTRINSICS batch for scaling
#         intrinsics1 = focal_lengths_to_scaled_intrinsics(relative_pose.focal_length[to_],
#                                                          batch.image_sizes[batch.cameras[to_].name],
#                                                          center=relative_pose.center[to_] if relative_pose.center is not None else None,
#                                                          shear=relative_pose.shear[to_] if relative_pose.shear is not None else None).squeeze()
#         intrinsics2 = focal_lengths_to_scaled_intrinsics(relative_pose.focal_length[from_],
#                                                          batch.image_sizes[batch.cameras[from_].name],
#                                                          center=relative_pose.center[from_] if relative_pose.center is not None else None,
#                                                          shear=relative_pose.shear[from_] if relative_pose.shear is not None else None).squeeze()
#
#         intrinsics = torch.cat([intrinsics1[None, None], intrinsics2[None, None]], dim=0)
#         relative_pose_ = (initial_pose[to_].inverse() @ initial_pose[from_])[None]
#         *images, idx = projection_frame_to_frame(batch,
#                                                  relative_pose_,
#                                                  intrinsics,
#                                                  return_depths=self.cfg.return_depth,
#                                                  bidirectional=self.cfg.bidirectional,
#                                                  idx='all',
#                                                  size=self.image_shape,
#                                                  from_=from_, to_=to_, from_file=False)
#         if self.cfg.bidirectional:
#             if self.cfg.return_depth:
#                 images1_2, images2_1, depths1_2, depths2_1 = images
#                 return {'images': torch.stack([images1_2, images2_1]), 'depths': torch.stack([depths1_2, depths2_1])}
#             else:
#                 images1_2, images2_1 = images
#                 return {'images': torch.stack([images1_2, images2_1]), 'depths': None}
#         else:
#             if self.cfg.return_depth:
#                 images2_1, depths2_1 = images
#                 return {'images': images2_1[None], 'depths': depths2_1[None]}
#             else:
#                 images2_1 = images[0]
#                 return {'images': images2_1[None], 'depths': None}
