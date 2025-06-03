import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, deg2rad

#
# def focal_lengths_to_intrinsics(
#     focal_lengths: Float[Tensor, " *batch"],
#     image_shape: tuple[int, int],
# ) -> Float[Tensor, "*batch 3 3"]:
#     device = focal_lengths.device
#     h, w = image_shape
#     focal_lengths = focal_lengths * (h * w) ** 0.5
#
#     intrinsics = torch.eye(3, dtype=torch.float32, device=device)
#     intrinsics[0, 2] = 0.5
#     intrinsics[1, 2] = 0.5
#     intrinsics = intrinsics.broadcast_to((*focal_lengths.shape, 3, 3)).contiguous()
#     intrinsics[..., 0, 0] = focal_lengths / w  # fx
#     intrinsics[..., 1, 1] = focal_lengths / h  # fy
#
#     return intrinsics
#
#
# def focal_lengths_to_scaled_intrinsics(
#     focal_lengths: Float[Tensor, " *batch"],
#     image_shape: tuple[int, int],
# ) -> Float[Tensor, "*batch 3 3"]:
#     device = focal_lengths.device
#     h, w = image_shape
#     focal_lengths = focal_lengths * (h * w) ** 0.5
#
#     intrinsics = torch.eye(3, dtype=torch.float32, device=device)
#     intrinsics[0, 2] = 0.5
#     intrinsics[1, 2] = 0.5
#     intrinsics = intrinsics.broadcast_to((*focal_lengths.shape, 3, 3)).contiguous()
#     intrinsics[..., 0, 0] = focal_lengths  # fx
#     intrinsics[..., 1, 1] = focal_lengths  # fy
#
#     return intrinsics
#
#
# def intrinsics_to_focal_lengths(
#     intrinsics: Float[Tensor, "*batch 3 3"],
#     image_shape: tuple[int, int],
# ) -> Float[Tensor, " *batch"]:
#     h, w = image_shape
#     f = intrinsics[..., 0, 0] * w
#     return f / ((h * w)**0.5)
def focal_lengths_to_intrinsics(
    focal_lengths: Float[Tensor, " *batch"],
    image_shape: tuple[int, int],
    center: Float[Tensor, " *batch 2"] = None,
    shear: Float[Tensor, " *batch"] = None,
    fourbyfour: bool = False
) -> Float[Tensor, "*batch 3 3"]:
    device = focal_lengths.device
    h, w = image_shape
    # HAFOV of 45°
    HAFOV = focal_lengths * deg2rad(torch.tensor(45))
    f = w / (2 * torch.tan(HAFOV/2))
    s = 4 if fourbyfour else 3
    intrinsics = torch.eye(s, dtype=torch.float32, device=device)
    intrinsics = intrinsics.broadcast_to((*focal_lengths.shape, s, s)).contiguous()
    intrinsics[..., 0, 2] = 0.5 if center is None else center[0]
    intrinsics[..., 1, 2] = 0.5 if center is None else center[1]
    intrinsics[..., 0, 1] = 0 if shear is None else shear
    intrinsics[..., 0, 0] = f/w  # fx
    intrinsics[..., 1, 1] = f/h  # fy
    return intrinsics

# def focal_lengths_to_intrinsics(
#     focal_lengths: Float[Tensor, " *batch"],
#     image_shape: tuple[int, int],
#     fourbyfour: bool = False
# ) -> Float[Tensor, "*batch 3 3"]:
#     device = focal_lengths.device
#     h, w = image_shape
#     # HFOV of 45°
#     HFOV = focal_lengths * deg2rad(torch.tensor(45))
#     f = w / (2 * torch.tan(HFOV/2))
#     s = 4 if fourbyfour else 3
#     intrinsics = torch.eye(s, dtype=torch.float32, device=device)
#     intrinsics[0, 2] = 0.5
#     intrinsics[1, 2] = 0.5
#     intrinsics = intrinsics.broadcast_to((*focal_lengths.shape, s, s)).contiguous()
#     intrinsics[..., 0, 0] = f/w  # fx
#     intrinsics[..., 1, 1] = f/h  # fy

    return intrinsics


def intrinsics_to_focal_lengths(
    intrinsics: Float[Tensor, "*batch 3 3"],
    image_shape: tuple[int, int],
) -> Float[Tensor, " *batch"]:
    h, w = image_shape
    f = intrinsics[..., 0, 0] * w
    HAFOV = 2 * torch.arctan(w/(2*f))
    return HAFOV / deg2rad(torch.tensor(45))


def scaled_intrinsics_to_focal_lengths(
    intrinsics: Float[Tensor, "*batch 3 3"],
    image_shape: tuple[int, int],
) -> Float[Tensor, " *batch"]:
    h, w = image_shape
    f = intrinsics[..., 0, 0]
    HFOV = 2 * torch.arctan(w/(2*f))
    return HFOV / deg2rad(torch.tensor(45))


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


def scale_intrinsics(intrinsics: Float[Tensor, "*batch 3 3"], scale: Float[Tensor, "n_scale"]):
    if len(scale.shape) > 1:
        batched_scale = True
    else:
        batched_scale = False
    if intrinsics.ndim > 2:
        *batch, _, _ = intrinsics.shape
        batched = True
    else:
        intrinsics = intrinsics.unsqueeze(0)
        batch = [1]
        batched = False
    c1, c2, c3 = intrinsics.split(1, -1)
    c1 = c1 * scale
    c2 = c2 * scale
    c3 = c3.repeat(*batch, 1, c1.shape[-1])
    if not batched_scale:
        if not batched:
            return torch.stack((c1, c2, c3), dim=-2).movedim(-1, len(batch)).squeeze()
        else:
            return torch.stack((c1, c2, c3), dim=-2).movedim(-1, len(batch)).squeeze(1)
    else:
        if not batched:
            return torch.stack((c1, c2, c3), dim=-2).movedim(-1, len(batch)).squeeze(0)
        else:
            return torch.stack((c1, c2, c3), dim=-2).movedim(-1, len(batch))

