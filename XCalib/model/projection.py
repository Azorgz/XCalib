from typing import Literal
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Float
from kornia.geometry import depth_to_3d_v2, transform_points, normalize_pixel_coordinates, \
    convert_points_from_homogeneous
from kornia.geometry import project_points as pp
from torch import Tensor, deg2rad
from ..third_party.ImagesCameras import ImageTensor
from ..dataset.types import Batch


def focal_lengths_to_scaled_intrinsics(
    focal_lengths: Float[Tensor, " *batch"],
    image_shape: tuple[int, int],
    center: Float[Tensor, " *batch 2"] = None,
    shear: Float[Tensor, " *batch"] = None,
    fourbyfour: bool = False,
) -> Float[Tensor, "*batch 3 3"]:
    device = focal_lengths.device
    h, w = image_shape
    # HAFOV of 45°
    HAFOV = focal_lengths * deg2rad(torch.tensor(45))
    f = w / (2 * torch.tan(HAFOV/2))
    s = 4 if fourbyfour else 3
    intrinsics = torch.eye(s, dtype=torch.float32, device=device)
    intrinsics = intrinsics.broadcast_to((*focal_lengths.shape, s, s)).contiguous()
    intrinsics[..., 0, 2] = 0.5 * w if center is None else center[..., :1] * w
    intrinsics[..., 1, 2] = 0.5 * h if center is None else center[..., 1:] * h
    intrinsics[..., 0, 1] = 0 if shear is None else shear * w
    intrinsics[..., 0, 0] = f  # fx
    intrinsics[..., 1, 1] = f  # fy
    return intrinsics


def depth_to_3d(depths: Float[Tensor, "*batch h w"],
                intrinsics: Float[Tensor, "*batch 3 3"]):
    *batch, h, w = depths.shape
    b = np.prod(batch)
    depths = depths.reshape(b, h, w)
    intrinsics = intrinsics.reshape(b, 3, 3)
    surfaces = torch.cat([depth_to_3d_v2(depth, intrinsic) for depth, intrinsic in zip(depths, intrinsics)])
    return surfaces.reshape(*batch, h, w, 3)


# def project_points(surfaces: Float[Tensor, "*batch dims 3"],
#                    intrinsics: Float[Tensor, "*batch 3 3"]):
#     *batch, _, _ = intrinsics.shape
#     *dims, xyz = surfaces[*[0 for _ in batch]].shape
#     b = np.prod(batch)
#     surfaces = surfaces.reshape(b, *dims, xyz)
#     intrinsics = intrinsics.reshape(b, 3, 3)
#     point_2d = torch.cat([pp(surface, intrinsic) for surface, intrinsic in zip(surfaces, intrinsics)])
#     return point_2d.reshape(*batch, *dims, 2)

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
                              return_depths=False, bidirectional=False,
                              from_: int = 1, to_: int = 0,
                              size=None, idx: int | None | Tensor | Literal['all'] = None,
                              from_file=False) -> Tensor | tuple:
    def resize(*args) -> list:
        if size is not None:
            return [F.interpolate(x, size=size, mode='bilinear', align_corners=False) for x in args]
        else:
            return [x for x in args]

    b, f, c, h, w = batch.videos.shape
    device = batch.videos.device
    n_pose = relative_poses.shape[0]
    n_scale = intrinsics.shape[1]
    u_ = 1 if from_ == to_ else 0
    #  extract images from batch
    if from_file:
        if idx != 'all':
            idx = torch.randint(0, batch.videos.shape[1] - u_, [1]) \
                if idx is None else torch.tensor(idx.clone().detach() % f).squeeze()[None]
            images_1 = ImageTensor(batch.frame_paths[to_][idx], device=device)
            images_2 = ImageTensor(batch.frame_paths[from_][idx], device=device)
        else:
            idx = torch.linspace(0, f - 1 - u_, f - u_, dtype=torch.int64, device=device)
            images_1 = ImageTensor(batch.frame_paths[to_][0], device=device).batch(
                [ImageTensor(batch.frame_paths[to_][i + 1], device=device) for i in
                 range(len(batch.frame_paths[to_]) - 1)])  # 1 c h w
            images_2 = ImageTensor(batch.frame_paths[from_][0], batched=True, device=device).batch(
                [ImageTensor(batch.frame_paths[from_][i + 1], device=device) for i in
                 range(len(batch.frame_paths[from_]) - 1)])  # 1 c h w
    else:
        if idx != 'all':
            idx = torch.randint(0, batch.videos.shape[1] - u_, [1]) \
                if idx is None else torch.tensor(idx.clone().detach() % f).squeeze()[None]
            images_1 = ImageTensor(batch.videos[to_, idx], device=device).resize(
                batch.image_sizes[batch.cameras[to_].name])
            images_2 = ImageTensor(batch.videos[from_, idx + u_], device=device).resize(
                batch.image_sizes[batch.cameras[from_].name])
        else:
            idx = torch.linspace(0, f - 1 - u_, f - u_, dtype=torch.int64, device=device)
            images_1 = ImageTensor(batch.videos[to_, idx], batched=True, device=device).resize(
                batch.image_sizes[batch.cameras[to_].name])  # 1 c h w
            images_2 = ImageTensor(batch.videos[from_, idx + u_], batched=True, device=device).resize(
                batch.image_sizes[batch.cameras[from_].name])  # 1 c h w

    # relative pose computation from extrinsics
    src_to_dst = relative_poses
    dst_to_src = src_to_dst.inverse()

    # Camera matrix from intrinsics
    cam_dst, cam_src = intrinsics.split(1, 0)
    cam_dst = repeat(cam_dst, ' b n_scale h w -> (b n_scale f) h w', f=idx.shape[0])
    cam_src = repeat(cam_src, ' b n_scale h w -> (b n_scale f) h w', f=idx.shape[0])
    image_shape1 = images_1.image_size
    image_shape2 = images_2.image_size

    # Depth from model
    depth_dst, depth_src = batch.depths[to_, idx][None], batch.depths[from_, idx][None]
    if depth_dst.ndim == 3:
        depth_dst, depth_src = depth_dst[:, None], depth_src[:, None]

    depth_dst = rearrange(F.interpolate(depth_dst.repeat(n_scale, 1, 1, 1), size=image_shape1, mode='bilinear',
                                        align_corners=True), 'n_scale f h w -> (n_scale f) h w', n_scale=n_scale)
    depth_src = rearrange(F.interpolate(depth_src.repeat(n_scale, 1, 1, 1), size=image_shape2, mode='bilinear',
                                        align_corners=True), 'n_scale f h w -> (n_scale f) h w', n_scale=n_scale)

    # convert depth to 3d points
    points_3d_dst: Tensor = repeat(depth_to_3d(depth_dst, cam_dst),
                                   'f h w xyz -> (n f) h w xyz', n=n_pose, f=n_scale * len(idx))  # Bx3xHxW
    # Bx3xHxW
    if bidirectional:
        points_3d_src: Tensor = repeat(depth_to_3d(depth_src, cam_src),
                                       'f h w xyz -> (n f) h w xyz', n=n_pose, f=n_scale * idx.shape[0])  # Bx3xHxW

    # apply transformation to the 3d points
    dst_to_src = repeat(dst_to_src, 'n_pose h w -> (n_pose n) h w', n=n_scale * idx.shape[0])
    points_3d_dst_trans = transform_points(dst_to_src.to(torch.float32), points_3d_dst)  # BxHxWx3
    if bidirectional:
        src_to_dst = repeat(src_to_dst, 'n_pose h w -> (n_pose n) h w', n=n_scale * idx.shape[0])
        points_3d_src_trans = transform_points(src_to_dst.to(torch.float32), points_3d_src)  # BxHxWx3

    # project back to pixels
    points_3d_dst_trans = rearrange(points_3d_dst_trans, '(n_pose n) h w xyz -> n n_pose h w xyz',
                                    n=n_scale * idx.shape[0], n_pose=n_pose)
    points_2d_dst_trans: Tensor = torch.stack(
        [project_points(points_3d_dst_trans[i], cam_src[i]) for i in range(n_scale * idx.shape[0])])  # BxHxWx2
    if bidirectional:
        points_3d_src_trans = rearrange(points_3d_src_trans, '(n_pose n) h w xyz -> n n_pose h w xyz',
                                        n=n_scale * idx.shape[0], n_pose=n_pose)
        points_2d_src_trans: Tensor = torch.stack(
            [project_points(points_3d_src_trans[i], cam_dst[i]) for i in range(n_scale * idx.shape[0])])  # BxHxWx2

    points_2d_dst_trans_norm: Tensor = rearrange((normalize_pixel_coordinates(points_2d_dst_trans, *image_shape2)
                                                  .to(depth_dst.dtype)),
                                                 'n_scale n_pose h w uv -> (n_scale n_pose) h w uv',
                                                 n_scale=n_scale * idx.shape[0], n_pose=n_pose)  # BxHxWx2
    if bidirectional:
        points_2d_src_trans_norm: Tensor = rearrange((normalize_pixel_coordinates(points_2d_src_trans, *image_shape1)
                                                      .to(depth_dst.dtype)),
                                                     'n_scale n_pose h w uv -> (n_scale n_pose) h w uv',
                                                     n_scale=n_scale * idx.shape[0],
                                                     n_pose=n_pose)  # BxHxWx2
    images2_1 = F.grid_sample(repeat(images_2, 'b c h w -> (n_scale n_pose b) c h w',
                                     n_scale=n_scale,
                                     n_pose=n_pose),
                              points_2d_dst_trans_norm,
                              align_corners=True)
    if bidirectional:
        images1_2 = F.grid_sample(
            repeat(images_1, 'b c h w -> (n_scale n_pose b) c h w', n_scale=n_scale, n_pose=n_pose),
            points_2d_src_trans_norm, align_corners=True)
        if return_depths:
            depths1_2 = F.grid_sample(
                repeat(depth_dst[:, None], 'b c h w -> (n_scale n_pose b) c h w', n_scale=n_scale, n_pose=n_pose),
                points_2d_src_trans_norm, align_corners=True)
            depths2_1 = F.grid_sample(
                repeat(depth_src[:, None], 'b c h w -> (n_scale n_pose b) c h w', n_scale=n_scale, n_pose=n_pose),
                points_2d_dst_trans_norm, align_corners=True)
            return *resize(images1_2, images2_1, depths1_2, depths2_1), idx
        else:
            return *resize(images1_2, images2_1), idx
    else:
        if return_depths:
            depths2_1 = F.grid_sample(
                repeat(depth_src[:, None], 'b c h w -> (n_scale n_pose b) c h w', n_scale=n_scale, n_pose=n_pose),
                points_2d_dst_trans_norm, align_corners=True)
            return *resize(images2_1, depths2_1), idx
        else:
            return *resize(images2_1), idx
