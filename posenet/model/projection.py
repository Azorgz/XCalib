from typing import Literal, Tuple
import open3d as o3d
import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from jaxtyping import Bool, Float, Int64
from kornia.geometry import depth_to_3d_v2, transform_points, normalize_pixel_coordinates, \
    convert_points_from_homogeneous
from kornia.geometry import project_points as pp
from torch import Tensor
from matplotlib import colormaps as cm
from ThirdParty.ImagesCameras import ImageTensor
from .procrustes import align_rigid
from ..dataset.types import Batch
from posenet.tracking import Tracks
from ..keypoints.keypoints import Keypoints


def homogenize_points(
        points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogenize_vectors(
        vectors: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Convert batched vectors (xyz) to (xyz0)."""
    return torch.cat([vectors, torch.zeros_like(vectors[..., :1])], dim=-1)


def transform_rigid(
        homogeneous_coordinates: Float[Tensor, "*#batch dim"],
        transformation: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """Apply a rigid-body transformation to points or vectors."""
    return einsum(transformation, homogeneous_coordinates, "... i j, ... j -> ... i")


def transform_cam2world(
        homogeneous_coordinates: Float[Tensor, "*#batch dim"],
        extrinsics: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """Transform points from 3D camera coordinates to 3D world coordinates."""
    return transform_rigid(homogeneous_coordinates, extrinsics)


def transform_world2cam(
        homogeneous_coordinates: Float[Tensor, "*#batch dim"],
        extrinsics: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """Transform points from 3D world coordinates to 3D camera coordinates."""
    return transform_rigid(homogeneous_coordinates, extrinsics.inverse())


def project_camera_space(
        points: Float[Tensor, "*#batch dim"],
        intrinsics: Float[Tensor, "*#batch dim dim"],
        epsilon: float = 1e-5,
        infinity: float = 1e8,
) -> Float[Tensor, "*batch dim-1"]:
    points = points / (points[..., -1:] + epsilon)
    points = points.nan_to_num(posinf=infinity, neginf=-infinity)
    points = einsum(intrinsics, points, "... i j, ... j -> ... i")
    return points[..., :-1]


def project(
        points: Float[Tensor, "*#batch dim"],
        extrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
        intrinsics: Float[Tensor, "*#batch dim dim"],
        epsilon: float = 1e-5,
) -> tuple[
    Float[Tensor, "*batch dim-1"],  # xy coordinates
    Bool[Tensor, " *batch"],  # whether points are in front of the camera
]:
    points = homogenize_points(points)
    points = transform_world2cam(points, extrinsics)[..., :-1]
    in_front_of_camera = points[..., -1] >= 0
    return project_camera_space(points, intrinsics, epsilon=epsilon), in_front_of_camera


def unproject(
        coordinates: Float[Tensor, "*#batch dim"],
        z: Float[Tensor, "*#batch"],
        intrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
) -> Float[Tensor, "*batch dim+1"]:
    """Unproject 2D camera coordinates with the given Z values."""

    # Apply the inverse intrinsics to the coordinates.
    coordinates = homogenize_points(coordinates)
    ray_directions = einsum(
        intrinsics.inverse(), coordinates, "... i j, ... j -> ... i"
    )

    # Apply the supplied depth values.
    return ray_directions * z[..., None]


def sample_image_grid(
        shape: tuple[int, ...],
        device: torch.device = torch.device("cpu"),
) -> tuple[
    Float[Tensor, "*shape dim"],  # float coordinates (xy indexing)
    Int64[Tensor, "*shape dim"],  # integer indices (ij indexing)
]:
    """Get normalized (range 0 to 1) coordinates and integer indices for an image."""

    # Each entry is a pixel-wise integer coordinate. In the 2D case, each entry is a
    # (row, col) coordinate.
    indices = [torch.arange(length, device=device) for length in shape]
    stacked_indices = torch.stack(torch.meshgrid(*indices, indexing="ij"), dim=-1)

    # Each entry is a floating-point coordinate in the range (0, 1). In the 2D case,
    # each entry is an (x, y) coordinate.
    coordinates = [(idx + 0.5) / length for idx, length in zip(indices, shape)]
    coordinates = reversed(coordinates)
    coordinates = torch.stack(torch.meshgrid(*coordinates, indexing="xy"), dim=-1)

    return coordinates, stacked_indices


def reproject_points(
        xyz: Float[Tensor, "*#batch 3"],
        relative_transformations: Float[Tensor, "*#batch 4 4"],
        intrinsics: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*#batch 2"]:
    """Transform the input points using the provided relative transformations, then
    project them using the provided intrinsics. After transformation, the points are
    assumed to be in camera space.
    """

    # Transform the 3D locations into the target view's camera space.
    xyz = einsum(
        relative_transformations,
        homogenize_points(xyz),
        "... i j, ... j -> ... i",
    )[..., :3]

    # Project the 3D locations in the target view's camera space.
    return project_camera_space(xyz, intrinsics)


# Given data with leading (batch, frame) dimensions, these helper functions select the
# earlier and later set of frames, respectively.
earlier = lambda x: x[:, :-1]  # noqa
later = lambda x: x[:, 1:]  # noqa


def compute_forward_flow(
        surfaces: Float[Tensor, "batch frame *grid xyz=3"],
        extrinsics: Float[Tensor, "batch frame 4 4"],
        intrinsics: Float[Tensor, "batch frame 3 3"],
) -> Float[Tensor, "batch frame-1 *grid xy=2"]:
    """Return the positions of all surface points with forward optical flow applied."""

    # Since the poses are camera-to-world and transformations are applied right to
    # left, this can be understood as follows: First, transform from the earlier
    # frame's camera space to world space. Then, transform from world space into the
    # later frame's camera space.
    forward_transformation = later(extrinsics).inverse() @ earlier(extrinsics)

    singletons = " ".join(["()"] * (surfaces.ndim - 3))
    pattern = f"b f i j -> b f {singletons} i j"
    return reproject_points(
        earlier(surfaces),
        rearrange(forward_transformation, pattern),
        rearrange(later(intrinsics), pattern),
    )


# def compute_forward_flow(
#         surfaces: Float[Tensor, "batch frame *grid xyz=3"],
#         extrinsics: Float[Tensor, "batch frame 4 4"],
#         intrinsics: Float[Tensor, "batch frame 3 3"],
#         shape_dst_frame: list[int] = None,
# ) -> Float[Tensor, "batch frame-1 *grid xy=2"]:
#     """Return the positions of all surface points with forward optical flow applied."""
#     batch, frame, h, w, xyz = surfaces.shape
#     # Since the poses are camera-to-world and transformations are applied right to
#     # left, this can be understood as follows: First, transform from the earlier
#     # frame's camera space to world space. Then, transform from world space into the
#     # later frame's camera space.
#     batch, frame, *grid, xyz = surfaces.shape
#     forward_transformation = later(extrinsics).inverse() @ earlier(extrinsics)  # b f-1 4 4
#
#     later_surfaces = later(surfaces)
#     surfaces_trans = transform_points(forward_transformation.to(torch.float32), later_surfaces)  # BxHxWx3
#     surfaces_trans = surfaces_trans.reshape(batch * (frame - 1), np.prod(grid), xyz).permute(1, 0, -1)
#     later_intrinsics = later(intrinsics).reshape(batch * (frame - 1), 3, 3)
#     # project back to pixels
#     points_2d_trans: Tensor = project_points(surfaces_trans, later_intrinsics)
#     if shape_dst_frame is not None:
#         x_flow, y_flow = points_2d_trans.split(1, -1)
#         x_flow, y_flow = x_flow / shape_dst_frame[1], y_flow / shape_dst_frame[0]
#         points_2d_trans = torch.cat([x_flow, y_flow], dim=-1)
#     return points_2d_trans.permute(1, 0, -1)


# def compute_backward_flow(
#         surfaces: Float[Tensor, "batch frame *grid xyz=3"],
#         extrinsics: Float[Tensor, "batch frame 4 4"],
#         intrinsics: Float[Tensor, "batch frame 3 3"],
#         shape_dst_frame: list[int] = None,
# ) -> Float[Tensor, "batch frame-1 *grid xy=2"]:
#     """Return the positions of all surface points with backward optical flow applied."""
#
#     # Since the poses are camera-to-world and transformations are applied right to
#     # left, this can be understood as follows: First, transform from the later
#     # frame's camera space to world space. Then, transform from world space into the
#     # earlier frame's camera space.
#     batch, frame, *grid, xyz = surfaces.shape
#     backward_transformation = earlier(extrinsics).inverse() @ later(extrinsics)
#
#     earlier_surfaces = earlier(surfaces)
#     surfaces_trans = transform_points(backward_transformation.to(torch.float32), earlier_surfaces)  # BxHxWx3
#     surfaces_trans = surfaces_trans.reshape(batch * (frame - 1), np.prod(grid), xyz).permute(1, 0, -1)
#     earlier_intrinsics = earlier(intrinsics).reshape(batch * (frame - 1), 3, 3)
#     # project back to pixels
#     points_2d_trans: Tensor = project_points(surfaces_trans, earlier_intrinsics)
#     if shape_dst_frame is not None:
#         x_flow, y_flow = points_2d_trans.split(1, -1)
#         x_flow, y_flow = x_flow / shape_dst_frame[1], y_flow / shape_dst_frame[0]
#         points_2d_trans = torch.cat([x_flow, y_flow], dim=-1)
#     return points_2d_trans.permute(1, 0, -1)


def depth_to_3d(depths: Float[Tensor, "*batch h w"],
                intrinsics: Float[Tensor, "*batch 3 3"]):
    *batch, h, w = depths.shape
    b = np.prod(batch)
    depths = depths.reshape(b, h, w)
    intrinsics = intrinsics.reshape(b, 3, 3)
    surfaces = torch.cat([depth_to_3d_v2(depth, intrinsic) for depth, intrinsic in zip(depths, intrinsics)])
    return surfaces.reshape(*batch, h, w, 3)


def project_points(surfaces: Float[Tensor, "*batch dims 3"],
                   intrinsics: Float[Tensor, "*batch 3 3"]):
    *batch, _, _ = intrinsics.shape
    *dims, xyz = surfaces[[0 for _ in batch]].shape
    b = np.prod(batch)
    surfaces = surfaces.reshape(b, *dims, xyz)
    intrinsics = intrinsics.reshape(b, 3, 3)
    point_2d = torch.cat([pp(surface, intrinsic) for surface, intrinsic in zip(surfaces, intrinsics)])
    return point_2d.reshape(*batch, *dims, 2)


def compute_backward_flow(
        surfaces: Float[Tensor, "batch frame *grid xyz=3"],
        extrinsics: Float[Tensor, "batch frame 4 4"],
        intrinsics: Float[Tensor, "batch frame 3 3"],
) -> Float[Tensor, "batch frame-1 *grid xy=2"]:
    """Return the positions of all surface points with backward optical flow applied."""

    # Since the poses are camera-to-world and transformations are applied right to
    # left, this can be understood as follows: First, transform from the later
    # frame's camera space to world space. Then, transform from world space into the
    # earlier frame's camera space.
    backward_transformation = earlier(extrinsics).inverse() @ later(extrinsics)

    singletons = " ".join(["()"] * (surfaces.ndim - 3))
    pattern = f"b f i j -> b f {singletons} i j"
    return reproject_points(
        later(surfaces),
        rearrange(backward_transformation, pattern),
        rearrange(earlier(intrinsics), pattern),
    )


def get_extrinsics(
        inverse_relative_transformations: Float[Tensor, "*batch pair 4 4"],
        first_pose: Float[Tensor, "4 4"] = None
) -> Float[Tensor, "*batch pair+1 4 4"]:
    """Convert the inverse relative transformations from ModelOutput to extrinsics.
    Each inverse relative transformation transforms points from frame {i + 1}'s
    camera space to frame i's camera space. Since our extrinsics are in
    camera-to-world format, this means that expressed in terms of camera poses, each
    inverse relative transformation is (P_i^-1 @ P_{i + 1}). If we assume that P_0
    is I (the identity pose), we can thus extract camera poses as follows:

    P_n = (I @ P_1) @ (P_1^-1 @ P_2) @ ... @ (P_{n - 1}^-1 @ P_n)

    This is slightly counterintuitive, since transformations are generally composed
    by right-to-left multiplication.
    """
    *batch, step, _, _ = inverse_relative_transformations.shape
    device = inverse_relative_transformations.device
    if first_pose is not None:
        pose = first_pose
        pose = pose.expand((*batch, 4, 4)).contiguous()
    else:
        pose = torch.eye(4, dtype=torch.float32, device=device)
        pose = pose.expand((*batch, 4, 4)).contiguous()
    result = [pose]
    for i in range(step):
        pose = pose @ inverse_relative_transformations[..., i, :, :]
        result.append(pose)
    return torch.stack(result, dim=-3)


def align_surfaces(
        surfaces: Float[Tensor, "batch frame height width 3"],
        backward_flows: Float[Tensor, "batch frame-1 height width xy=2"],
        backward_weights: Float[Tensor, "batch frame-1 height width"],
        indices: Int64[Tensor, " pixel_index"],
        initial_pose: Float[Tensor, "*batch 4 4"] = None,
        return_transformations: bool = False
) -> Float[Tensor, "batch frame 4 4"] or Float[Tuple, "..."]:
    *original_batch, f, h, w, xyz = surfaces.shape
    # Convert the depth maps into camera-space 3D surfaces (b, f, h, w, xyz).
    xy, _ = sample_image_grid((h, w), device=surfaces.device)
    surfaces = surfaces.flatten(start_dim=0, end_dim=-5)  # Flatten batch
    b, f, h, w, xyz = surfaces.shape
    # Subsample the surfaces to select points for Procrustes alignment. Select the later
    # points from the surfaces using the provided indices.
    xyz_later = rearrange(later(surfaces), "b f h w xyz -> b f (h w) xyz")
    xyz_later = xyz_later[..., indices, :]

    # Flow the grid of XY locations backwards, then select from the flowed XY locations
    # using the provided indices.
    xy_earlier = rearrange(xy + backward_flows, "b f h w xy -> b f (h w) xy")
    xy_earlier = xy_earlier[..., indices, :]

    # Use the earlier XY locations to select from the earlier 3D surfaces.
    xyz_earlier = F.grid_sample(
        rearrange(earlier(surfaces), "b f h w xyz -> (b f) xyz h w"),
        rearrange(xy_earlier * 2 - 1, "b f p xy -> (b f) p () xy"),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    xyz_earlier = rearrange(xyz_earlier, "(b f) xyz p () -> b f p xyz", b=b, f=f - 1)

    # Estimate poses via Procrustes alignment.
    inverse_relative_transformations = align_rigid(
        xyz_later,
        xyz_earlier,
        rearrange(backward_weights, "b f h w -> b f (h w)")[..., indices],
    )
    inverse_relative_transformations = inverse_relative_transformations.reshape(*original_batch, f - 1, 4, 4)
    extrinsics = get_extrinsics(inverse_relative_transformations, initial_pose)
    if return_transformations:
        return extrinsics, inverse_relative_transformations
    else:
        return extrinsics


def align_surfaces_new(
        surfaces: Float[Tensor, "batch frame height width 3"],
        keypoints: Float[Tensor, "batch frame-1 N xy=2"],
        backward_weights: Float[Tensor, "batch frame-1 height width"],
        initial_pose: Float[Tensor, "*batch 4 4"] = None,
        indice=None) -> Float[Tensor, "batch frame 4 4"]:
    *original_batch, f, h, w, xyz = surfaces.shape
    # Convert the depth maps into camera-space 3D surfaces (b, f, h, w, xyz).
    xy, _ = sample_image_grid((h, w), device=surfaces.device)
    surfaces = surfaces.flatten(start_dim=0, end_dim=-5)  # Flatten batch
    b, f, h, w, xyz = surfaces.shape
    # Project the two surface in the same world frame
    # surface_1, surface_2 = surfaces
    surfaces_ = transform_cam2world(homogenize_points(surfaces[0]), initial_pose[0].inverse())[..., :3]

    xyz_later = F.grid_sample(
        rearrange(later(surfaces_), "b f h w xyz -> (b f) xyz h w"),
        rearrange(keypoints.keypoints1[:1] * 2 - 1, "b f p xy -> (b f) p () xy"),
        mode="bilinear",
        padding_mode="border",
        align_corners=False, )
    xyz_earlier = F.grid_sample(
        rearrange(earlier(surfaces_), "b f h w xyz -> (b f) xyz h w"),
        rearrange(keypoints.keypoints0[:1] * 2 - 1, "b f p xy -> (b f) p () xy"),
        mode="bilinear",
        padding_mode="border",
        align_corners=False, )

    # Gather the two later surfaces
    xyz_later = rearrange(xyz_later, "(b f) xyz p () -> f (b p) xyz", b=1, f=f - 1)[None]
    xyz_earlier = rearrange(xyz_earlier, "(b f) xyz p () -> f (b p) xyz", b=1, f=f - 1)[None]
    # Flow the grid of XY locations backwards, then select from the flowed XY locations
    # using the provided indices.

    weights = F.grid_sample(
        rearrange(backward_weights, "b f h w -> (b f) () h w"),
        rearrange(keypoints.keypoints0[:1] * 2 - 1, "b f p xy -> (b f) p () xy"),
        mode="bilinear",
        padding_mode="border",
        align_corners=False, )
    # Estimate poses via Procrustes alignment.
    inverse_relative_transformations = align_rigid(
        xyz_later,
        xyz_earlier,
        rearrange(weights, "(b f) () p () -> () f (b p)", b=1, f=f - 1),
    )
    inverse_relative_transformations = inverse_relative_transformations.reshape(1, f - 1, 4, 4)

    return get_extrinsics(inverse_relative_transformations)


def align_surfaces_joint_keypoints(
        surfaces: Float[Tensor, "batch frame height width 3"],
        keypoints: Keypoints,
        backward_weights: Float[Tensor, "batch frame-1 height width"],
        initial_pose: Float[Tensor, "*batch 4 4"] = None,
        first_pose: Float[Tensor, "4 4"] = None,
        **kwargs) -> Float[Tensor, "batch frame 4 4"]:
    *original_batch, f, h, w, xyz = surfaces.shape
    # Convert the depth maps into camera-space 3D surfaces (b, f, h, w, xyz).
    xy, _ = sample_image_grid((h, w), device=surfaces.device)
    surfaces = surfaces.flatten(start_dim=0, end_dim=-5)  # Flatten batch
    b, f, h, w, xyz = surfaces.shape
    # Project the two surface in the same world frame
    # surface_1, surface_2 = surfaces
    surfaces_ = torch.stack([transform_cam2world(homogenize_points(surface),
                                                 initial_pose[i].inverse())[..., :3] for i, surface in
                             enumerate(surfaces)], dim=0)

    xyz_later = F.grid_sample(
        rearrange(later(surfaces_), "b f h w xyz -> (b f) xyz h w"),
        rearrange(keypoints.keypoints1 * 2 - 1, "b f p xy -> (b f) p () xy"),
        mode="bilinear",
        padding_mode="border",
        align_corners=False, )
    xyz_earlier = F.grid_sample(
        rearrange(earlier(surfaces_), "b f h w xyz -> (b f) xyz h w"),
        rearrange(keypoints.keypoints0 * 2 - 1, "b f p xy -> (b f) p () xy"),
        mode="bilinear",
        padding_mode="border",
        align_corners=False, )
    # pc_visu(xyz_earlier, xyz_later, f)

    # Gather the two later surfaces
    xyz_later = rearrange(xyz_later, "(b f) xyz p () -> f (b p) xyz", b=b, f=f - 1)[None]
    xyz_earlier = rearrange(xyz_earlier, "(b f) xyz p () -> f (b p) xyz", b=b, f=f - 1)[None]
    # Flow the grid of XY locations backwards, then select from the flowed XY locations
    # using the provided indices.

    weights = F.grid_sample(
        rearrange(backward_weights, "b f h w -> (b f) () h w"),
        rearrange(keypoints.keypoints0 * 2 - 1, "b f p xy -> (b f) p () xy"),
        mode="bilinear",
        padding_mode="border",
        align_corners=False, )
    # Estimate poses via Procrustes alignment.
    inverse_relative_transformations = align_rigid(
        xyz_later,
        xyz_earlier,
        rearrange(weights, "(b f) () p () -> () f (b p)", b=b, f=f - 1),
    )
    inverse_relative_transformations = inverse_relative_transformations.reshape(1, f - 1, 4, 4)

    return get_extrinsics(inverse_relative_transformations, first_pose)


def align_surfaces_joint(
        surfaces: Float[Tensor, "batch frame height width 3"],
        backward_flows: Float[Tensor, "batch frame-1 height width xy=2"],
        backward_weights: Float[Tensor, "batch frame-1 height width"],
        initial_pose: Float[Tensor, "*batch 4 4"] = None,
        first_pose: Float[Tensor, "4 4"] = None,
        indices: Int64[Tensor, " pixel_index"] = None) -> Float[Tensor, "batch frame 4 4"]:
    *original_batch, f, h, w, xyz = surfaces.shape
    # Convert the depth maps into camera-space 3D surfaces (b, f, h, w, xyz).
    xy, _ = sample_image_grid((h, w), device=surfaces.device)
    surfaces = surfaces.flatten(start_dim=0, end_dim=-5)  # Flatten batch
    b, f, h, w, xyz = surfaces.shape
    # Project the two surface in the same world frame
    surfaces = torch.stack([transform_cam2world(homogenize_points(surface),
                                                initial_pose[i].inverse())[..., :3] for i, surface in
                            enumerate(surfaces)], dim=0)
    # Subsample the surfaces to select points for Procrustes alignment. Select the later
    # points from the surfaces using the provided indices.
    xyz_later = rearrange(later(surfaces), "b f h w xyz -> b f (h w) xyz")
    # Gather the two later surfaces
    xyz_later = rearrange(xyz_later[..., indices, :], "b f p xyz -> f (b p) xyz")[None]

    # Flow the grid of XY locations backwards, then select from the flowed XY locations
    # using the provided indices.
    xy_earlier = rearrange(xy + backward_flows, "b f h w xy -> b f (h w) xy")
    xy_earlier = xy_earlier[..., indices, :]

    # Use the earlier XY locations to select from the earlier 3D surfaces.
    xyz_earlier = F.grid_sample(
        rearrange(earlier(surfaces), "b f h w xyz -> (b f) xyz h w"),
        rearrange(xy_earlier * 2 - 1, "b f p xy -> (b f) p () xy"),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    # Gather the two later surfaces
    xyz_earlier = rearrange(xyz_earlier, "(b f) xyz p () -> f (b p) xyz", b=b, f=f - 1)[None]
    weights = rearrange(backward_weights, "b f h w -> b f (h w)")[..., indices]
    # Estimate poses via Procrustes alignment.
    inverse_relative_transformations = align_rigid(
        xyz_later,
        xyz_earlier,
        rearrange(weights, "b f p ->  f (b p)")[None],
    )
    inverse_relative_transformations = inverse_relative_transformations.reshape(1, f - 1, 4, 4)

    return get_extrinsics(inverse_relative_transformations, first_pose)


def compute_track_flow(
        surfaces: Float[Tensor, "batch frame height width xyz=3"],
        extrinsics: Float[Tensor, "batch frame 4 4"],
        intrinsics: Float[Tensor, "batch frame 3 3"],
        tracks: Tracks,
) -> tuple[
    Float[Tensor, "batch frame_source frame_target point 2"],  # flow
    Bool[Tensor, "batch frame_source frame_target point"],  # visibility
]:
    # Sample the surfaces at the track locations.
    b, f, _, _, _ = surfaces.shape
    xyz = F.grid_sample(
        rearrange(surfaces, "b f h w xyz -> (b f) xyz h w"),
        rearrange(tracks.xy * 2 - 1, "b f p xy -> (b f) () p xy"),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    xyz = rearrange(xyz, "(b f) xy () p -> b f p xy", b=b, f=f)

    # Add singleton dimensions so that everything broadcasts to the following shape:
    # (b = batch, fs = source frame, ft = target frame, p = point)
    xy_source = rearrange(tracks.xy, "b fs p xy -> b fs () p xy")
    xyz_source = rearrange(xyz, "b fs p xyz -> b fs () p xyz")
    extrinsics_source = rearrange(extrinsics, "b fs i j -> b fs () () i j")
    extrinsics_target = rearrange(extrinsics, "b ft i j -> b () ft () i j")
    intrinsics_target = rearrange(intrinsics, "b ft i j -> b () ft () i j")
    visibility_source = rearrange(tracks.visibility, "b fs p -> b fs () p")
    visibility_target = rearrange(tracks.visibility, "b ft p -> b () ft p")

    # Compute flow and visibility.
    xy_target = reproject_points(
        xyz_source,
        extrinsics_target.inverse() @ extrinsics_source,
        intrinsics_target,
    )
    visibility = visibility_source & visibility_target

    # Filter out points that are not in the frame for either the source or target.
    source_in_frame = (xy_source >= 0).all(dim=-1) & (xy_source < 1).all(dim=-1)
    target_in_frame = (xy_target >= 0).all(dim=-1) & (xy_target < 1).all(dim=-1)
    visibility = visibility & source_in_frame & target_in_frame

    return xy_target, visibility


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

    # normalize points between [-1 / 1]
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


def projection_frame_to_frame_sequential_(batch: Batch,
                                         relative_poses: Float[Tensor, "batch n_pose 4 4"],
                                         intrinsics: Float[Tensor, "batch n_scale 3 3"],
                                         size=(480, 640),
                                         idx: int | None | Tensor | Literal['all'] = None) -> Tensor | tuple:
    b, f, c, h, w = batch.videos.shape
    device = batch.videos.device
    n_pose = relative_poses.shape[1]
    n_scale = intrinsics.shape[1]
    assert n_pose == n_scale
    #  extract images from batch
    if idx != 'all':
        idx = torch.randint(0, batch.videos.shape[1] - 1, [1]) if idx is None else (
            torch.tensor(idx.clone().detach() % f).squeeze())[None]
        images_1 = ImageTensor(batch.frame_paths[0][idx + 1], device=device).resize(size)
        images_2 = ImageTensor(batch.frame_paths[1][idx + 1], device=device).resize(size)
    else:
        idx = torch.linspace(1, f, f, dtype=torch.int64, device=device)
        images_1 = batch.videos[0, 1:].resize(size)  # f c h w
        images_2 = batch.videos[1, 1:].resize(size)  # f c h w

    # relative pose computation from extrinsics
    # cam1_poses, cam2_poses = relative_poses.split(1, 0)
    # Camera matrix from intrinsics
    # cam_0, cam_1 = intrinsics.split(1, 0)
    # cam_0 = repeat(cam_0, ' () n_scale h w -> (n_scale f) h w', f=idx.shape[0])
    # cam_1 = repeat(cam_1, ' () n_scale h w -> (n_scale f) h w', f=idx.shape[0])
    intrinsics = repeat(intrinsics, ' b n_scale h w -> (b n_scale f) h w', f=idx.shape[0])
    relative_poses = repeat(relative_poses, ' b n_scale h w -> (b n_scale f) h w', f=idx.shape[0])
    # Depth from model
    depths = batch.depths[:, idx]
    if depths.ndim == 3:
        depths = depths[:, None]

    depths = repeat(F.interpolate(depths, size=size, mode='bilinear',
                                  align_corners=True), 'b f h w -> (b n_scale f) h w', n_scale=n_scale)
    # convert depth to 3d points
    points_3d: Tensor = depth_to_3d(depths, intrinsics)  # Bx3xHxW

    # apply transformation to the 3d points
    points_3d_trans = transform_points(relative_poses.to(torch.float32), points_3d)  # BxHxWx3

    # project back to pixels
    points_2d_trans: Tensor = project_points(points_3d_trans, intrinsics)  # BxHxWx2

    # normalize points between [-1 / 1]
    points_2d_trans_norm: Tensor = (normalize_pixel_coordinates(points_2d_trans, *size).to(depths.dtype)).split(n_pose,
                                                                                                                0)
    images_1_proj = F.grid_sample(repeat(images_1, 'b c h w -> (n_pose b) c h w', n_pose=n_pose),
                                  points_2d_trans_norm[0],
                                  align_corners=True)
    images_2_proj = F.grid_sample(repeat(images_2, 'b c h w -> (n_pose b) c h w', n_pose=n_pose),
                                  points_2d_trans_norm[1],
                                  align_corners=True)
    return images_1_proj, images_2_proj


def pc_visu(xyz_earlier, xyz_later, frame):
    print(frame)
    n = xyz_later.shape[-2]
    cloud_earlier = xyz_earlier.squeeze().permute(0, 2, 1)
    cloud_later = xyz_later.squeeze().permute(0, 2, 1)
    colors = cm['Blues_r'](torch.linspace(0, 1, frame).numpy())[:, None, :3]
    pcd = []
    for f in range(2):
        color = np.repeat(colors[f], n, 0)
        pcd_ = o3d.t.geometry.PointCloud()
        pcd_.point.positions = o3d.core.Tensor(cloud_earlier[f].cpu().detach().numpy(), o3d.core.Dtype.Float32)
        pcd_.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcd_.point.colors = o3d.core.Tensor(color, o3d.core.Dtype.Float32)
        pcd.append(pcd_)
    o3d.visualization.draw_geometries([p.voxel_down_sample(voxel_size=0.1).to_legacy() for p in pcd])