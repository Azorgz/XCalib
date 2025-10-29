import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from kornia import create_meshgrid
from kornia.geometry import transform_points, normalize_pixel_coordinates, compose_transformations, \
    inverse_transformation, depth_to_3d_v2, convert_points_from_homogeneous
from torch import Tensor, nn
from torch.nn import MaxPool2d
from torchvision.transforms.functional import gaussian_blur
from misc.Mytypes import Batch
from model.inverse_flow import max_method


def depth_warp(batch: Batch, camera_target, camera_src, from_=1, to_=0,
               flowCorrector: nn.Module = None, return_flow: bool = False) -> Tensor:
    intrinsics = torch.cat([camera_target.intrinsics, camera_src.intrinsics], dim=0)
    dst_trans_src: Tensor = compose_transformations(inverse_transformation(camera_target.extrinsics),
                                                    camera_src.extrinsics)
    return projection_frame_to_frame(batch, dst_trans_src, intrinsics, from_=from_, to_=to_, return_flow=return_flow)


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

    # apply intrinsics and return
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
                              from_: int = 1, to_: int = 0,
                              return_flow: bool = False) -> Tensor | tuple:
    b, c, h, w = batch.images[to_].shape
    #  extract images from batch
    images = batch.images[from_]
    max_pool = MaxPool2d(kernel_size=(3, 7), stride=1, padding=(1, 3))
    device = images.device
    direct = True
    for i, depth in enumerate(batch.depths):
        if depth is not None:
            depths = depth
            if i == to_:
                direct = True
            else:
                direct = False
            break

    # Camera matrix from intrinsics
    cam_tgt, cam_src = intrinsics[..., :3, :3].split(1, 0)
    src_shape = batch.images[from_].shape[-2:]
    tgt_shape = batch.images[to_].shape[-2:]

    if direct:
        # convert depth to 3d points
        points_3d_tgt: Tensor = depth_to_3d(depths, cam_tgt)[:, 0]  # Bx3xHxW
        # apply transformation to the 3d points
        points_3d_tgt_trans = transform_points(relative_poses, points_3d_tgt)  # BxHxWx3
        # project the transformed 3d points to 2d
        points_2d_tgt_trans: Tensor = torch.cat(
            [project_points(points_3d_tgt_trans[i], cam_src) for i in range(b)])  # BxHxWx2
        # normalize the 2d points to [-1, 1]
        sampling_grid: Tensor = normalize_pixel_coordinates(points_2d_tgt_trans, *src_shape).to(depths.dtype)  # BxHxWx2
        sampling_grid = gaussian_blur(max_pool(sampling_grid.permute(0, 3, 1, 2)), (5, 5), (1.6, 1.6)).permute(0, 2, 3, 1)
        # sample the source image at the projected 2d points
        images_reg = F.grid_sample(images, sampling_grid, align_corners=True)
    else:
        # convert depth to 3d points
        points_3d_src: Tensor = depth_to_3d(depths, cam_src)[:, 0]  # Bx3xHxW
        # apply inverse transformation to the 3d points
        relative_poses_inv = inverse_transformation(relative_poses)
        points_3d_src_trans = transform_points(relative_poses_inv, points_3d_src)  # BxHxWx3
        # project the transformed 3d points to 2d
        points_2d_src_trans: Tensor = torch.cat(
            [project_points(points_3d_src_trans[i], cam_tgt) for i in range(b)])  # BxHxWx2
        # normalize the 2d points to [-1, 1]
        # sampling_grid = normalize_pixel_coordinates(points_2d_src_trans, *tgt_shape).to(depths.dtype)  # BxHxWx2
        sampling_grid = points_2d_src_trans.to(depths.dtype)  # BxHxWx2
        grid_n = create_meshgrid(*src_shape, device=device, dtype=images.dtype, normalized_coordinates=True).repeat(b, 1, 1, 1)  # 1xHxWx2
        grid = torch.stack([(grid_n[..., 0] + 1) * (tgt_shape[1] / 2), (grid_n[..., 1] + 1) * (tgt_shape[0] / 2)], dim=-1)  # BxHxWx2
        inverse_flow, bkw_mask = max_method((grid - sampling_grid).permute(0, 3, 1, 2))
        sampling_grid = normalize_pixel_coordinates(grid - inverse_flow.permute(0, 2, 3, 1), *tgt_shape)  # BxHxWx2

        # flow = 2 * grid - sampling_grid  # Bx1xHxWx2
        # sampling_grid: Tensor = F.interpolate(inverse_flow.permute(0, 3, 1, 2), size=tgt_shape)  # BxH'xW'x2
        # sample the source image at the projected 2d points
        # images_reg = F.grid_sample(images, sampling_grid.permute(0, 2, 3, 1), align_corners=True)
        images_reg = F.grid_sample(images, F.interpolate(sampling_grid.permute(0, 3, 1, 2), tgt_shape).permute(0, 2, 3, 1) , align_corners=True)
    if not return_flow:
        return images_reg
    else:
        return images_reg, sampling_grid
