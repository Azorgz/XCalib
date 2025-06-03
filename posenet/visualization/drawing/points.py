from typing import Optional

import numpy as np
import torch
from einops import repeat, rearrange
from jaxtyping import Float
from kornia.geometry import transform_points
from matplotlib import pyplot as plt
from torch import Tensor

from ThirdParty.ImagesCameras import ImageTensor
from .coordinate_conversion import generate_conversions
from .rendering import render_over_image
from .types import Pair, Scalar, Vector, sanitize_scalar, sanitize_vector


def interpolate(p_from, p_to, num):
    direction = (p_to - p_from) / np.linalg.norm(p_to - p_from)
    distance = np.linalg.norm(p_to - p_from) / (num - 1)

    ret_vec = []

    for i in range(0, num):
        ret_vec.append(p_from + direction * distance * i)

    return np.array(ret_vec)


def draw_points(
        image: Float[Tensor, "3 height width"] | Float[Tensor, "4 height width"],
        points: Vector,
        color: Vector = [1, 1, 1],
        radius: Scalar = 1,
        inner_radius: Scalar = 0,
        num_msaa_passes: int = 1,
        x_range: Optional[Pair] = None,
        y_range: Optional[Pair] = None,
) -> Float[Tensor, "3 height width"] | Float[Tensor, "4 height width"]:
    device = image.device
    points = sanitize_vector(points, 2, device)
    color = sanitize_vector(color, 3, device)
    radius = sanitize_scalar(radius, device)
    inner_radius = sanitize_scalar(inner_radius, device)
    (num_points,) = torch.broadcast_shapes(
        points.shape[0],
        color.shape[0],
        radius.shape,
        inner_radius.shape,
    )

    # Convert world-space points to pixel space.
    _, h, w = image.shape
    world_to_pixel, _ = generate_conversions((h, w), device, x_range, y_range)
    points = world_to_pixel(points)

    def color_function(
            xy: Float[Tensor, "point 2"],
    ) -> Float[Tensor, "point 4"]:
        # Define a vector between the start and end points.
        delta = xy[:, None] - points[None]
        delta_norm = delta.norm(dim=-1)
        mask = (delta_norm >= inner_radius[None]) & (delta_norm <= radius[None])

        # Determine the sample's color.
        selectable_color = color.broadcast_to((num_points, 3))
        arrangement = mask * torch.arange(num_points, device=device)
        top_color = selectable_color.gather(
            dim=0,
            index=repeat(arrangement.argmax(dim=1), "s -> s c", c=3),
        )
        rgba = torch.cat((top_color, mask.any(dim=1).float()[:, None]), dim=-1)

        return rgba

    return render_over_image(image, color_function, device, num_passes=num_msaa_passes)


def plotImage3D(axe: plt.axes, img: ImageTensor, transform, scale, img_size=(30, 45)):
    """
        plot image (plane) in 3D with given Pose (R|t) of corner point

        ax      : matplotlib axes to plot on
        R       : Rotation as roation matrix
        t       : translation as np.array (1, 3), left down corner of image in real world coord
        size    : Size as np.array (1, 2), size of image plane in real world
        img_scale: Scale to bring down image, since this solution needs 1 face for every pixel it will become very slow on big images
    """
    img = img.resize((img_size[0], img_size[1]), keep_ratio=True)
    ratio = img.image_size[1]/img.image_size[0]
    x = np.linspace(-scale/2*ratio, scale/2*ratio, img.image_size[1])
    y = np.linspace(-scale/2, scale/2, img.image_size[0])
    xx, yy = np.meshgrid(x, y)
    zz = np.ones_like(xx)*0.5*scale
    srf = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], -1)
    transform = transform.detach().cpu().numpy()
    srf = srf @ transform[:3, :3].T + transform[:3, -1][None, :].repeat(srf.shape[0], 0)
    srf = srf.reshape(img.image_size[0], img.image_size[1], 3)
    xx, zz, yy = srf[:, :, 0], srf[:, :, 1], srf[:, :, 2]
    # xx = xx.reshape(img.image_size[1], img.image_size[0])
    # yy = yy.reshape(img.image_size[1], img.image_size[0])
    # zz = zz.reshape(img.image_size[1], img.image_size[0])

    # xx = np.zeros((img_size[0], img_size[1]))
    # yy = np.zeros((img_size[0], img_size[1]))
    # zz = np.zeros((img_size[0], img_size[1]))
    # l1 = interpolate(corners[0], corners[2], img_size[0])
    # xx[:, 0] = l1[:, 0]
    # yy[:, 0] = l1[:, 1]
    # zz[:, 0] = l1[:, 2]
    # l1 = interpolate(corners[1], corners[3], img_size[0])
    # xx[:, img_size[1] - 1] = l1[:, 0]
    # yy[:, img_size[1] - 1] = l1[:, 1]
    # zz[:, img_size[1] - 1] = l1[:, 2]

    # for idx in range(0, img_size[0]):
    #     p_from = np.array((xx[idx, 0], yy[idx, 0], zz[idx, 0]))
    #     p_to = np.array((xx[idx, img_size[1] - 1], yy[idx, img_size[1] - 1], zz[idx, img_size[1] - 1]))
    #     l1 = interpolate(p_from, p_to, img_size[1])
    #     xx[idx, :] = l1[:, 0]
    #     yy[idx, :] = l1[:, 1]
    #     zz[idx, :] = l1[:, 2]
    img = img.permute('b', 'h', 'w', 'c').to_numpy()[0]
    axe.plot_surface(xx, yy, zz, rstride=1, cstride=1, facecolors=img, shade=False)
    return axe


def draw_simple_cam(axe: plt.axes,
                    scale=0.1,
                    color='red',
                    transforms: [Tensor, np.ndarray] = None,
                    images: list = None,
                    ratio: float = None):
    if transforms is not None:
        if len(transforms.shape) == 2:
            transforms = transforms.unsqueeze(0)
    else:
        transforms = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    if images is not None:
        assert (len(images) == transforms.shape[0]) or len(images) == 1
        if len(images) == 1:
            images = images * transforms.shape[0]
        ratio = images[0].shape[-1] / images[0].shape[-2]
    # Make data
    ratio = ratio if ratio is not None else 90/60
    x = [0, -0.5 * ratio, -0.5 * ratio, 0.5 * ratio, 0.5 * ratio]
    y = [0, -0.5, 0.5, 0.5, -0.5]
    z = [0, 0.5]
    xyz = torch.from_numpy(np.array([[x[0], y[0], z[0]],
                                     [x[1], y[1], z[1]],
                                     [x[2], y[2], z[1]],
                                     [x[3], y[3], z[1]],
                                     [x[4], y[4], z[1]]])).to(transforms.device) * scale

    for i, transform in enumerate(transforms):
        x, y, z = transform_points(transform.unsqueeze(0), xyz.to(torch.float32)).split(1, dim=-1)
        x, z, y = x.squeeze().detach().cpu().numpy(), y.squeeze().detach().cpu().numpy(), z.squeeze().detach().cpu().numpy()
        lines = np.array([[[x[0], x[1]], [y[0], y[1]], [z[0], z[1]]], [[x[0], x[2]], [y[0], y[2]], [z[0], z[2]]],
                          [[x[0], x[3]], [y[0], y[3]], [z[0], z[3]]], [[x[0], x[4]], [y[0], y[4]], [z[0], z[4]]],
                          [[x[1], x[2]], [y[1], y[2]], [z[1], z[2]]], [[x[2], x[3]], [y[2], y[3]], [z[2], z[3]]],
                          [[x[3], x[4]], [y[3], y[4]], [z[3], z[4]]], [[x[4], x[1]], [y[4], y[1]], [z[4], z[1]]]])
        axe.scatter3D(x, y, z, color=color)
        if images is not None:
            image = images[i]
            axe = plotImage3D(axe, image, transform, scale, img_size=(60, 90))

        for line in lines:
            axe.plot(*line, color=color)
    return axe


# def draw_rig(axe: plt.axes, cam1: Float[np.ndarray, "f 3"], cam2: Float[np.ndarray, "f 3"], color='black'):
#     for xyz1, xyz2 in zip(cam1, cam2):
#         x1, y1, z1 = xyz1
#         x2, y2, z2 = xyz2
#         axe.plot3D([x1, x2], [y1, y2], [z1, z2], color=color)

def draw_rig(axe: plt.axes, trajectory: Float[Tensor, "f 4 4"], color='black', scale=0.1):
    xyz1 = torch.tensor([-5, 0, 0]).unsqueeze(0).to(trajectory.device)*scale
    xyz2 = torch.tensor([5, 0, 0]).unsqueeze(0).to(trajectory.device)*scale
    for i, transform in enumerate(trajectory):
        x1, y1, z1 = transform_points(transform.unsqueeze(0), xyz1.to(torch.float32)).split(1, dim=-1)
        x2, y2, z2 = transform_points(transform.unsqueeze(0), xyz2.to(torch.float32)).split(1, dim=-1)
        x1, y1, z1 = x1.squeeze().detach().cpu().numpy(), y1.squeeze().detach().cpu().numpy(), z1.squeeze().detach().cpu().numpy()
        x2, y2, z2 = x2.squeeze().detach().cpu().numpy(), y2.squeeze().detach().cpu().numpy(), z2.squeeze().detach().cpu().numpy()
        axe.plot3D([x1, x2], [z1, z2], [y1, y2], color=color)

def draw_cam(axe: plt.axes, scale=0.1, color='red', transforms=None):
    if transforms is not None:
        if len(transforms.shape) == 2:
            transforms = transforms.unsqueeze(0)
    else:
        transforms = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    # Make data
    nx, ny = (2, 2)
    u = np.linspace(-0.5, 0.5, nx)
    v = np.linspace(-0.5, 0.5, ny)
    w = np.linspace(-1, 0, ny)
    # Face z = -1
    x1, y1 = np.meshgrid(u, v)
    z1 = np.ones_like(x1) * -1
    # Face z = 0
    x2, y2 = np.meshgrid(u, v)
    z2 = np.zeros_like(x2)
    # Face x = -0.5
    y3, z3 = np.meshgrid(v, w)
    x3 = np.ones_like(y3) * -0.5
    # Face x = 0.5
    y4, z4 = np.meshgrid(v, w)
    x4 = np.ones_like(y3) * 0.5
    # Face y = -0.5
    x5, z5 = np.meshgrid(u, w)
    y5 = np.ones_like(x5) * -0.5
    # Face y = 0.5
    x6, z6 = np.meshgrid(u, w)
    y6 = np.ones_like(x5) * 0.5
    # Conic face
    ncone = 25
    u = np.linspace(0, 2 * np.pi, ncone)
    circle_small = np.stack([np.cos(u), np.sin(u)], axis=0) * 0.25
    circle_big = np.stack([np.cos(u), np.sin(u)], axis=0) * 0.5
    x7 = np.stack([circle_small[0, :], circle_big[0, :]], axis=0)
    y7 = np.stack([circle_small[1, :], circle_big[1, :]], axis=0)
    z7 = np.tile([0, 0.5], [ncone, 1]).T
    # Circle face
    x8 = np.stack([circle_small[0, :] * 0, circle_big[0, :]], axis=0)
    y8 = np.stack([circle_small[1, :] * 0, circle_big[1, :]], axis=0)
    z8 = np.ones_like(x8) * 0.5
    # Ready the tensor for tranformation.
    X = torch.from_numpy(np.concatenate([x1, x2, x3, x4, x5, x6, x7, x8], axis=-1))
    Y = torch.from_numpy(np.concatenate([y1, y2, y3, y4, y5, y6, y7, y8], axis=-1))
    Z = torch.from_numpy(np.concatenate([z1, z2, z3, z4, z5, z6, z7, z8], axis=-1))
    n = X.shape[-1]
    xyz = rearrange(torch.stack([X, Y, Z], dim=-1).to(transforms.device), 'c n xyz -> (c n) xyz') * scale

    for transform in transforms:
        X, Y, Z = rearrange(transform_points(transform.unsqueeze(0), xyz.to(torch.float32)), '(c n) xyz -> c n xyz',
                            c=2, n=n).split(1, dim=-1)
        X = X.squeeze().split([nx, nx, nx, nx, nx, nx, ncone, ncone], dim=-1)
        Y = Y.squeeze().split([nx, nx, nx, nx, nx, nx, ncone, ncone], dim=-1)
        Z = Z.squeeze().split([nx, nx, nx, nx, nx, nx, ncone, ncone], dim=-1)

        for x, y, z in zip(X, Y, Z):
            x, y, z = x.detach().cpu().numpy(), y.detach().cpu().numpy(), z.detach().cpu().numpy()
            axe.plot_surface(x, y, z, color=color)
    return axe
