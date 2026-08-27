"""
inverse_optical_flow_torch.py
GPU-accelerated, PyTorch-based implementation of inverse optical flow (drop-in replacement).

Implements:
- max_method(forward_flow)
- average_method(forward_flow)

Input format (same as README):
- forward_flow: numpy array or torch tensor with shape (2, H, W) or (B, 2, H, W)
  channel 0 = horizontal (x), channel 1 = vertical (y)
Outputs:
- backward_flow: same dtype and device as internal tensors; shape (2, H, W) or (B, 2, H, W)
- disocclusion_mask: uint8-like mask (0/1) shape (H, W) or (B, H, W)

Notes:
- Implementation uses vectorized mapping: every source pixel (x_src,y_src) maps to
  a destination (x_dst,y_dst) = (x_src + flow_x, y_src + flow_y).
- For non-integer targets we bucket to integer grid with floor/round rules.
- `max_method`: when multiple sources map to same target, we pick the source with maximum
  "score" defined by magnitude of flow (or by a chosen channel). In original repo they
  compare something—this is a robust fallback: we pick the source with largest absolute x+y magnitude.
- `average_method`: averages all mapped sources (useful when flows overlap).
- Implementation supports batching and works on CUDA if available.
"""

from typing import Tuple
import torch
import numpy as np
import torch.nn.functional as F


def _to_torch(flow, device='cuda'):
    """Utility: convert numpy to torch, move to device if possible."""
    if isinstance(flow, np.ndarray):
        t = torch.from_numpy(flow)
    elif isinstance(flow, torch.Tensor):
        t = flow
    else:
        raise TypeError("flow must be numpy.ndarray or torch.Tensor")
    # ensure float32
    if not torch.is_floating_point(t):
        t = t.float()
    # expected shape: (2, H, W) or (B, 2, H, W)
    if t.dim() == 3:
        t = t.unsqueeze(0)  # make batch dim
    if device == 'cuda' and torch.cuda.is_available():
        t = t.cuda()
    return t


def _unbatch(t):
    """If batch dim was added, remove it for single-batch convenience in return types."""
    if t.shape[0] == 1:
        return t.squeeze(0)
    return t


def _compute_target_coords(flow: torch.Tensor):
    """
    flow: (B, 2, H, W)
    returns:
    - target_x: (B, H*W) long tensor with x indices (0..W-1)
    - target_y: (B, H*W) long tensor with y indices (0..H-1)
    - src_x: (B, H*W) long tensor with source x (0..W-1)
    - src_y: (B, H*W) long tensor with source y (0..H-1)
    - valid_mask: (B, H*W) bool tensor indicating target in bounds and finite
    """
    B, C, H, W = flow.shape
    assert C == 2, "flow must have 2 channels (x,y)"

    # create source coordinates
    ys = torch.arange(0, H, device=flow.device, dtype=flow.dtype)
    xs = torch.arange(0, W, device=flow.device, dtype=flow.dtype)
    grid_y = ys.view(1, H, 1).expand(1, H, W)
    grid_x = xs.view(1, 1, W).expand(1, H, W)
    grid_y = grid_y.expand(B, H, W)
    grid_x = grid_x.expand(B, H, W)

    # flow channels
    flow_x = flow[:, 0, :, :]  # (B,H,W)
    flow_y = flow[:, 1, :, :]  # (B,H,W)

    # floating point destination
    dst_x = grid_x + flow_x
    dst_y = grid_y + flow_y

    # validity (finite)
    valid = torch.isfinite(dst_x) & torch.isfinite(dst_y)

    # round or floor destinations into integer grid indices:
    # we'll use floor (like splatting to NW/NE/SW/SE in original) — simplest is round
    # but using floor keeps consistency with classic implementations. We'll use round here.
    # you can change to floor/ceil or bilinear splatting if desired.
    dst_xi = torch.round(dst_x).long()
    dst_yi = torch.round(dst_y).long()

    # source indices flattened:
    src_y = grid_y.long().reshape(B, -1)  # (B, H*W)
    src_x = grid_x.long().reshape(B, -1)

    dst_xi = dst_xi.reshape(B, -1)
    dst_yi = dst_yi.reshape(B, -1)
    valid = valid.reshape(B, -1)

    return dst_xi, dst_yi, src_x, src_y, valid, H, W


def _flatten_idx(x_idx, y_idx, W):
    """Return flattened index into H*W from x,y indices: idx = y*W + x"""
    return y_idx * W + x_idx


def max_method(forward_flow) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute backward flow (approx.) using 'max' accumulation rule on GPU (PyTorch).

    forward_flow: np.ndarray or torch.Tensor with shape (2, H, W) or (B,2,H,W)
    returns: backward_flow, disocclusion_mask both numpy arrays (if input was numpy)
    """
    was_numpy = isinstance(forward_flow, np.ndarray)
    flow = _to_torch(forward_flow, device='cuda')  # (B,2,H,W)
    B, C, H, W = flow.shape

    dst_xi, dst_yi, src_x, src_y, valid, H, W = _compute_target_coords(flow)

    # flattened indices for destination
    dst_flat = _flatten_idx(dst_xi, dst_yi, W)  # shape (B, H*W)
    src_flat = _flatten_idx(src_x, src_y, W)    # shape (B, H*W)

    # mask in-bounds
    in_bounds = (dst_xi >= 0) & (dst_xi < W) & \
                (dst_yi >= 0) & (dst_yi < H) & valid

    # We'll compute a "score" per source to choose the max contributing source when collisions happen.
    # Use magnitude of flow vector as score (could be replaced by brightness or other metric).
    score = torch.sqrt(flow[:, 0].reshape(B, -1) ** 2 + flow[:, 1].reshape(B, -1) ** 2)  # (B, H*W)
    score = torch.where(in_bounds, score, torch.tensor(float('-inf'), device=flow.device))

    # For each target flattened index, we want the source index with max score.
    # Trick: we can use scatter_max-like behavior using grouping by indices. PyTorch doesn't have a direct scatter_max in pure python,
    # but we can use the following trick with sorting per batch.
    # This approach will be O(N log N) because of the sort, but vectorized on GPU.
    bwd_flow = torch.zeros_like(flow)  # (B,2,H,W)
    mask = torch.zeros(B, H * W, dtype=torch.bool, device=flow.device)

    for b in range(B):
        # gather per-batch arrays
        dst_i = dst_flat[b]        # (N,)
        valid_i = in_bounds[b]     # (N,)
        score_i = score[b]         # (N,)
        src_x_i = src_x[b]         # (N,)
        src_y_i = src_y[b]         # (N,)

        # filter only valid mappings
        if valid_i.any():
            dst_valid = dst_i[valid_i]
            score_valid = score_i[valid_i]
            src_x_valid = src_x_i[valid_i]
            src_y_valid = src_y_i[valid_i]

            # sort by dst index then by score descending
            # create keys for sorting: (dst_index, -score)
            # We'll sort by dst_index primary, score secondary.
            # order = torch.argsort(torch.stack([dst_valid, -score_valid], dim=1), dim=0, stable=True)
            # argsort returned indices per column; we want row ordering by combined keys:
            # simpler and robust: get permutation via lexsort style using tuple keys
            # keys = dst_valid * (score_valid.new_tensor(1) * 0)  # placeholder, we use a two-stage approach

            # Two-stage: sort by dst_index, then within each block pick max score
            dst_unique, inverse_idx = torch.unique(dst_valid, return_inverse=True)
            # For each unique dst, pick the index of max score_valid
            # compute max across groups:
            # group_max_score = scatter_max(score_valid, inverse_idx, dim=0, out_size=len(dst_unique))
            idxs = torch.arange(score_valid.shape[0], device=score_valid.device)
            packed = score_valid * (score_valid.shape[0] + 1) + idxs  # unique pack; max of packed gives max score and idx
            # scatter reduce by dst_valid
            packed_max = torch.zeros(dst_unique.size(0), device=score_valid.device).scatter_reduce(0, inverse_idx, packed, reduce='amax')
            # retrieve source index as packed_max % ... :
            chosen_src_idxs = (packed_max.long() % (score_valid.shape[0])).long()

            # Now write into bwd_flow at dst locations:
            chosen_dst_flat = dst_unique  # these are flattened indices
            chosen_src_x = src_x_valid[chosen_src_idxs]
            chosen_src_y = src_y_valid[chosen_src_idxs]

            # read forward_flow values at chosen src positions:
            # we need to index the original flow channels. Build src_flat index to index into flattened H*W
            # but easiest: convert src_x/y to linear coordinate for gathering
            linear_src = chosen_src_y * W + chosen_src_x  # (K,)
            # flatten flow channels:
            f0 = flow[b, 0].reshape(-1)  # (H*W,)
            f1 = flow[b, 1].reshape(-1)
            picked_f0 = f0[linear_src]
            picked_f1 = f1[linear_src]

            # write into backward flow: at dst positions we want the negative displacements to go back to source.
            # backward_flow(dst) = -forward_flow(src)  (approx)
            # Convert chosen_dst_flat to y,x:
            dst_y = (chosen_dst_flat // W).long()
            dst_x = (chosen_dst_flat % W).long()
            bwd_flow[b, 0, dst_y, dst_x] = -picked_f0
            bwd_flow[b, 1, dst_y, dst_x] = -picked_f1

            # mark mask
            mask_indices = chosen_dst_flat
            mask[b, mask_indices] = True

    disocclusion_mask = mask.reshape(B, H, W).to(dtype=torch.uint8)

    # If original input was non-batched, remove batch dim
    bwd_out = _unbatch(bwd_flow)
    mask_out = _unbatch(disocclusion_mask)

    if was_numpy:
        return bwd_out.cpu().numpy(), mask_out.cpu().numpy()
    else:
        return bwd_out, mask_out


def average_method(forward_flow) -> Tuple[np.ndarray, np.ndarray]:
    """
    Average accumulative method: for targets with multiple sources, average their contributions.
    Vectorized scatter-add approach.
    """
    was_numpy = isinstance(forward_flow, np.ndarray)
    flow = _to_torch(forward_flow, device='cuda')
    B, C, H, W = flow.shape

    dst_xi, dst_yi, src_x, src_y, valid, H, W = _compute_target_coords(flow)
    dst_flat = _flatten_idx(dst_xi, dst_yi, W)  # (B, N)

    in_bounds = (dst_xi >= 0) & (dst_xi < W) & (dst_yi >= 0) & (dst_yi < H) & valid
    N = H * W

    # Prepare output accumulators
    bwd_flow = torch.zeros_like(flow)  # will hold summed contributions
    counts = torch.zeros(B, N, device=flow.device, dtype=flow.dtype)

    for b in range(B):
        maskb = in_bounds[b]
        if not maskb.any():
            continue
        dstb = dst_flat[b][maskb]   # target indices
        src_xb = src_x[b][maskb]
        src_yb = src_y[b][maskb]
        linear_src = src_yb * W + src_xb

        # gather corresponding forward flow vectors
        f0 = flow[b, 0].reshape(-1)
        f1 = flow[b, 1].reshape(-1)
        picked_f0 = f0[linear_src]
        picked_f1 = f1[linear_src]

        # accumulate (-flow) into backward flow at dst positions
        # we create flattened accum buffers per channel
        acc0 = torch.zeros(N, device=flow.device, dtype=flow.dtype)
        acc1 = torch.zeros(N, device=flow.device, dtype=flow.dtype)
        cnt = torch.zeros(N, device=flow.device, dtype=flow.dtype)

        # scatter_add
        acc0 = acc0.scatter_add_(0, dstb, -picked_f0)
        acc1 = acc1.scatter_add_(0, dstb, -picked_f1)
        cnt = cnt.scatter_add_(0, dstb, torch.ones_like(dstb, dtype=flow.dtype))

        # reshape back to H,W
        bwd_flow[b, 0] = acc0.reshape(H, W)
        bwd_flow[b, 1] = acc1.reshape(H, W)
        counts[b] = cnt

    # compute averages where counts>0
    counts = counts.reshape(B, 1, H, W)
    mask_nonzero = (counts > 0)
    # avoid division by zero
    bwd_flow = torch.where(mask_nonzero, bwd_flow / counts, bwd_flow)

    disocclusion_mask = (counts.reshape(B, H, W) > 0).to(dtype=torch.uint8)

    bwd_out = _unbatch(bwd_flow)
    mask_out = _unbatch(disocclusion_mask)

    if was_numpy:
        return bwd_out.cpu().numpy(), mask_out.cpu().numpy()
    else:
        return bwd_out, mask_out

def dense_method(
    flow_A2B: torch.Tensor,
    eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the inverse optical/dense flow field and occlusion mask from a forward flow field.

    Args:
        flow_A2B: Tensor of shape (B, 2, H, W) or (2, H, W) where channel 0 is dx and channel 1 is dy.
        eps: Small threshold to determine valid/unoccluded regions.

    Returns:
        flow_B2A: Tensor of same shape as flow_A2B representing flow from B to A.
        occlusion_mask: Tensor of shape (B, 1, H, W) or (1, H, W) where 1 indicates an
                        occluded/unmapped pixel in frame B, and 0 indicates a valid pixel.
    """
    is_unbatched = flow_A2B.dim() == 3
    if is_unbatched:
        flow_A2B = flow_A2B.unsqueeze(0)

    N, _, H, W = flow_A2B.shape
    device = flow_A2B.device

    # 1. Create target coordinate grid for Frame A
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=flow_A2B.dtype),
        torch.arange(W, device=device, dtype=flow_A2B.dtype),
        indexing="ij"
    )
    grid_A = torch.stack([x_grid, y_grid], dim=0).unsqueeze(0).repeat(N, 1, 1, 1) # (N, 2, H, W)

    # 2. Map coordinates from Frame A to Frame B
    target_coords_B = grid_A + flow_A2B # (N, 2, H, W)
    u_B = target_coords_B[:, 0]
    v_B = target_coords_B[:, 1]

    # 3. Identify valid in-bounds target coordinates
    in_bounds = (u_B >= 0) & (u_B <= W - 1) & (v_B >= 0) & (v_B <= H - 1)

    # 4. Prepare for forward splatting onto Grid B
    flow_B2A_flat = torch.zeros((N, 2, H * W), device=device, dtype=flow_A2B.dtype)
    weight_accum = torch.zeros((N, 1, H * W), device=device, dtype=flow_A2B.dtype)

    # Bilinear corner indices and weights
    u0 = torch.floor(u_B).long()
    v0 = torch.floor(v_B).long()
    u1 = u0 + 1
    v1 = v0 + 1

    u0_c = torch.clamp(u0, 0, W - 1)
    u1_c = torch.clamp(u1, 0, W - 1)
    v0_c = torch.clamp(v0, 0, H - 1)
    v1_c = torch.clamp(v1, 0, H - 1)

    w_u1 = u_B - u0.float()
    w_u0 = 1.0 - w_u1
    w_v1 = v_B - v0.float()
    w_v0 = 1.0 - w_v1

    # Inverted flow value vector from A to B (-flow_A2B)
    inv_val = -flow_A2B.reshape(N, 2, -1)

    corners = [
        (w_u0 * w_v0, v0_c, u0_c),
        (w_u1 * w_v0, v0_c, u1_c),
        (w_u0 * w_v1, v1_c, u0_c),
        (w_u1 * w_v1, v1_c, u1_c),
    ]

    # 5. Scatter forward flow vectors onto Frame B grid
    for b in range(N):
        mask_b = in_bounds[b]
        for w, v_idx, u_idx in corners:
            weight = (w[b] * mask_b).reshape(-1)
            flat_idx = (v_idx[b] * W + u_idx[b]).reshape(-1)

            for c in range(2):
                flow_B2A_flat[b, c].scatter_add_(0, flat_idx, inv_val[b, c] * weight)
            weight_accum[b, 0].scatter_add_(0, flat_idx, weight)

    # 6. Normalize splatting weights
    valid_mask = weight_accum > eps
    flow_B2A_flat = torch.where(valid_mask, flow_B2A_flat / (weight_accum + eps), 0.0)

    flow_B2A = flow_B2A_flat.reshape(N, 2, H, W)
    # Occlusion mask: 1 where no flow vectors mapped (hole/occluded), 0 where valid
    occlusion_mask = (~valid_mask).reshape(N, 1, H, W).float()

    if is_unbatched:
        flow_B2A = flow_B2A.squeeze(0)
        occlusion_mask = occlusion_mask.squeeze(0)

    return flow_B2A, occlusion_mask


def forward_warp_and_occlusion(
    img1: torch.Tensor,
    flow: torch.Tensor,
    target_shape: tuple[int, int],
    threshold_collision: float = 1.25,
    bg_color: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward-warps img1 into target shape (H2, W2) using normalized [-1, 1] canonical coordinates.

    Args:
        img1: (B, C, H1, W1) Source image tensor.
        flow: (B, 2, H1, W1) Forward optical flow in pixel units of target domain (W2, H2).
        target_shape: (H2, W2) Desired target output frame resolution.
        threshold_collision: Accumulated density threshold for collision/occlusion.
        bg_color: Fill value for unmapped/occluded pixels.

    Returns:
        warped_img2: (B, C, H2, W2) Warped target image.
        occ_mask: (B, 1, H1, W1) Occlusion mask on source grid (1 = Occluded).
    """
    B, C, H1, W1 = img1.shape
    H2, W2 = target_shape
    device = flow.device

    # 1. Normalized [-1, 1] Canonical Coordinates
    xs = torch.linspace(-1.0, 1.0, W1, device=device)
    ys = torch.linspace(-1.0, 1.0, H1, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

    # Convert flow to normalized [-1, 1] coordinate shifts
    norm_flow_x = 2.0 * flow[:, 0, :, :] / max(1, W2 - 1)
    norm_flow_y = 2.0 * flow[:, 1, :, :] / max(1, H2 - 1)

    norm_target_x = grid_x + norm_flow_x
    norm_target_y = grid_y + norm_flow_y

    # Target pixel coordinates (continuous float)
    target_px_x = (norm_target_x + 1.0) * 0.5 * (W2 - 1)
    target_px_y = (norm_target_y + 1.0) * 0.5 * (H2 - 1)

    # 2. Continuous Bilinear Splatting Neighbors (4 corners per source pixel)
    x0 = torch.floor(target_px_x)
    x1 = x0 + 1.0
    y0 = torch.floor(target_px_y)
    y1 = y0 + 1.0

    # Bilinear Interpolation Weights (Fully Differentiable w.r.t target_px_x and y)
    w_top_left = (x1 - target_px_x) * (y1 - target_px_y)
    w_top_right = (target_px_x - x0) * (y1 - target_px_y)
    w_bot_left = (x1 - target_px_x) * (target_px_y - y0)
    w_bot_right = (target_px_x - x0) * (target_px_y - y0)

    # Clip coordinate indices for accumulation boundaries
    x0_idx = torch.clamp(x0.long(), 0, W2 - 1)
    x1_idx = torch.clamp(x1.long(), 0, W2 - 1)
    y0_idx = torch.clamp(y0.long(), 0, H2 - 1)
    y1_idx = torch.clamp(y1.long(), 0, H2 - 1)

    neighbors = [
        (y0_idx, x0_idx, w_top_left),
        (y0_idx, x1_idx, w_top_right),
        (y1_idx, x0_idx, w_bot_left),
        (y1_idx, x1_idx, w_bot_right),
    ]

    # Accumulation buffers
    density = torch.zeros((B, 1, H2, W2), device=device)
    warped_acc = torch.zeros((B, C, H2, W2), device=device)

    # 3. Soft Differentiable Accumulation Loop
    for y_idx, x_idx, weight in neighbors:
        weight_exp = weight.unsqueeze(1)  # (B, 1, H1, W1)
        flat_idx = (y_idx * W2 + x_idx).view(B, 1, -1)

        # Accumulate Density using torch.view / scatter (maintains autograd graph)
        w_flat = weight.view(B, 1, -1)
        density_flat = density.view(B, 1, H2 * W2)
        density = density_flat.scatter_add(2, flat_idx, w_flat).view(B, 1, H2, W2)

        # Accumulate Weighted Colors
        weighted_colors_flat = (img1 * weight_exp).view(B, C, -1)
        warped_acc_flat = warped_acc.view(B, C, H2 * W2)
        flat_idx_c = flat_idx.expand(-1, C, -1)
        warped_acc = warped_acc_flat.scatter_add(
            2, flat_idx_c, weighted_colors_flat
        ).view(B, C, H2, W2)

    # 4. Normalize Warped Image
    valid_mask = density > 1e-4
    warped_img2 = torch.where(
        valid_mask,
        warped_acc / (density + 1e-8),
        torch.tensor(bg_color, device=device),
    )

    # 5. Differentiable Occlusion & Collision Masking
    grid_canonical = torch.stack([norm_target_x, norm_target_y], dim=-1)

    # Sample density back to source grid
    sampled_density = F.grid_sample(
        density, grid_canonical, mode="bilinear", align_corners=True
    ).squeeze(1)

    # 5a. Out-of-bounds Mask (Triggered when target coordinate leaves [-1, 1])
    oob_x = torch.relu(torch.abs(norm_target_x) - 1.0)
    oob_y = torch.relu(torch.abs(norm_target_y) - 1.0)
    oob_mask = torch.sigmoid((oob_x + oob_y) * 20.0)

    # 5b. Collision / Compression Mask
    # In normalized splatting, standard density is ~1.0 regardless of target scale.
    # Collisions occur when multiple source points map to the same target area (density > threshold).
    collision_mask = torch.sigmoid((sampled_density - threshold_collision) * 10.0)

    # 5c. Folding / Inversion Mask via Jacobian Determinant
    # Compute spatial derivatives of flow in source pixel coordinates
    du_dx = torch.gradient(flow[:, 0, :, :], dim=2)[0]
    du_dy = torch.gradient(flow[:, 0, :, :], dim=1)[0]
    dv_dx = torch.gradient(flow[:, 1, :, :], dim=2)[0]
    dv_dy = torch.gradient(flow[:, 1, :, :], dim=1)[0]

    # Local area element ratio (det J <= 0 indicates surface folding/inversion)
    det_J = (1.0 + du_dx) * (1.0 + dv_dy) - (du_dy * dv_dx)
    folding_mask = torch.sigmoid((0.0 - det_J) * 10.0)

    # Combine masks using soft logical OR: 1 - (1 - A)(1 - B)(1 - C)
    # This prevents artificial saturation caused by simple addition
    occ_mask = (
            1.0 - (1.0 - oob_mask) * (1.0 - collision_mask) * (1.0 - folding_mask)
    ).unsqueeze(1)

    return warped_img2, occ_mask

def forward_warp(
    img1: torch.Tensor,
    flow: torch.Tensor,
    target_shape: tuple[int, int],
    threshold_collision: float = 1.25,
    bg_color: float = 0.0,
    return_occlusion_mask: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """Fully differentiable forward-warping and occlusion masking with scale-adaptive footprint splatting.

    Spans over all target pixels falling under the projected extent of each source pixel.
    """
    B, C, H1, W1 = img1.shape
    H2, W2 = target_shape
    device = flow.device

    # 1. Canonical Coordinate Setup [-1, 1]
    xs = torch.linspace(-1.0, 1.0, W1, device=device)
    ys = torch.linspace(-1.0, 1.0, H1, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

    # Convert flow to normalized [-1, 1] shifts
    norm_flow_x = 2.0 * flow[:, 0, :, :] / max(1, W2 - 1)
    norm_flow_y = 2.0 * flow[:, 1, :, :] / max(1, H2 - 1)

    norm_target_x = grid_x + norm_flow_x
    norm_target_y = grid_y + norm_flow_y

    # Target continuous pixel coordinates (float)
    target_px_x = (norm_target_x + 1.0) * 0.5 * (W2 - 1)
    target_px_y = (norm_target_y + 1.0) * 0.5 * (H2 - 1)

    # 2. Scale-Adaptive Radius Definition
    # Footprint half-width in target pixel units (minimum 0.5 to prevent sub-pixel collapsing)
    rx = max(0.5, (W2 / W1) / 2.0)
    ry = max(0.5, (H2 / H1) / 2.0)

    # Integer search window bounding box
    rad_x_int = int(torch.ceil(torch.tensor(2.0 * rx)).item())
    rad_y_int = int(torch.ceil(torch.tensor(2.0 * ry)).item())

    # Pre-compute target center anchor for local window offsetting
    center_x = torch.round(target_px_x)
    center_y = torch.round(target_px_y)

    # 3. Compute Footprint Normalization Factor (Ensures sum of kernel weights = 1.0 per pixel)
    # We evaluate unnormalized weights over the full footprint window
    weight_sum = torch.zeros_like(target_px_x)
    for dy in range(-rad_y_int, rad_y_int + 1):
        for dx in range(-rad_x_int, rad_x_int + 1):
            qx_curr = center_x + dx
            qy_curr = center_y + dy
            dist_sq_x = (qx_curr - target_px_x) ** 2
            dist_sq_y = (qy_curr - target_px_y) ** 2
            w_raw = torch.exp(-dist_sq_x / (2.0 * rx**2) - dist_sq_y / (2.0 * ry**2))
            weight_sum = weight_sum + w_raw

    weight_sum = torch.clamp(weight_sum, min=1e-8)

    # Accumulation buffers on target grid (H2, W2)
    density = torch.zeros((B, 1, H2, W2), device=device)
    warped_acc = torch.zeros((B, C, H2, W2), device=device)

    # 4. Multi-Pixel Area Splatting
    for dy in range(-rad_y_int, rad_y_int + 1):
        for dx in range(-rad_x_int, rad_x_int + 1):
            qx_curr = center_x + dx
            qy_curr = center_y + dy

            # Differentiable Spatial Weight w.r.t continuous target_px_x and y
            dist_sq_x = (qx_curr - target_px_x) ** 2
            dist_sq_y = (qy_curr - target_px_y) ** 2
            w_raw = torch.exp(-dist_sq_x / (2.0 * rx**2) - dist_sq_y / (2.0 * ry**2))
            w = w_raw / weight_sum  # Differentiable normalized spatial weight

            # Clamped integer coordinates for scatter accumulation
            qx_idx = torch.clamp(qx_curr.long(), 0, W2 - 1)
            qy_idx = torch.clamp(qy_curr.long(), 0, H2 - 1)

            flat_idx = (qy_idx * W2 + qx_idx).view(B, 1, -1)

            # Accumulate Density (autograd-tracked scatter_add)
            w_flat = w.view(B, 1, -1)
            density_flat = density.view(B, 1, H2 * W2)
            density = density_flat.scatter_add(2, flat_idx, w_flat).view(B, 1, H2, W2)

            # Accumulate Weighted Colors
            w_expanded = w.unsqueeze(1)
            weighted_colors_flat = (img1 * w_expanded).view(B, C, -1)
            warped_acc_flat = warped_acc.view(B, C, H2 * W2)
            flat_idx_c = flat_idx.expand(-1, C, -1)
            warped_acc = warped_acc_flat.scatter_add(
                2, flat_idx_c, weighted_colors_flat
            ).view(B, C, H2, W2)

    # 5. Normalize Target Image
    valid_mask = density > 1e-4
    warped_img2 = torch.where(
        valid_mask,
        warped_acc / (density + 1e-8),
        torch.tensor(bg_color, device=device),
    )
    if return_occlusion_mask:
        # 6. Differentiable Occlusion Masking
        grid_canonical = torch.stack([norm_target_x, norm_target_y], dim=-1)

        sampled_density = F.grid_sample(
            density, grid_canonical, mode="bilinear", align_corners=True
        ).squeeze(1)

        # Out-of-bounds Mask
        oob_x = torch.relu(torch.abs(norm_target_x) - 1.0)
        oob_y = torch.relu(torch.abs(norm_target_y) - 1.0)
        oob_mask = torch.sigmoid((oob_x + oob_y) * 20.0)

        # Soft Collision Masking
        collision_mask = torch.sigmoid((sampled_density - threshold_collision) * 10.0)

        # Surface Folding Mask (Jacobian Determinant)
        du_dx = torch.gradient(flow[:, 0, :, :], dim=2)[0]
        du_dy = torch.gradient(flow[:, 0, :, :], dim=1)[0]
        dv_dx = torch.gradient(flow[:, 1, :, :], dim=2)[0]
        dv_dy = torch.gradient(flow[:, 1, :, :], dim=1)[0]

        det_J = (1.0 + du_dx) * (1.0 + dv_dy) - (du_dy * dv_dx)
        folding_mask = torch.sigmoid((0.0 - det_J) * 10.0)

        # Soft Union
        occ_mask = (
            1.0 - (1.0 - oob_mask) * (1.0 - collision_mask) * (1.0 - folding_mask)
        ).unsqueeze(1)

        return warped_img2, occ_mask
    return warped_img2


def inverse_flow_maxpool(
    flow_A2B: torch.Tensor,
    eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the inverse flow field using MAX-MAGNITUDE pooling to preserve
    the largest displacements (regardless of sign) into target areas.

    Args:
        flow_A2B: Tensor of shape (B, 2, H, W) or (2, H, W) (dx, dy).
        eps: Threshold for valid flow/occlusion determination.

    Returns:
        flow_B2A: Tensor of same shape as flow_A2B containing inverted max-magnitude flow.
        occlusion_mask: Tensor (B, 1, H, W) or (1, H, W) where 1 indicates an
                        occluded/unmapped region in frame B.
    """
    is_unbatched = flow_A2B.dim() == 3
    if is_unbatched:
        flow_A2B = flow_A2B.unsqueeze(0)

    N, _, H, W = flow_A2B.shape
    device = flow_A2B.device

    # 1. Target coordinate grid in Frame A
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=flow_A2B.dtype),
        torch.arange(W, device=device, dtype=flow_A2B.dtype),
        indexing="ij"
    )
    grid_A = torch.stack([x_grid, y_grid], dim=0).unsqueeze(0).repeat(N, 1, 1, 1)

    # 2. Forward projection to Frame B
    inv_flow = -flow_A2B  # Inverted vectors (B -> A)
    target_coords_B = grid_A + flow_A2B

    u_B = target_coords_B[:, 0]
    v_B = target_coords_B[:, 1]

    # In-bounds mask
    in_bounds = (u_B >= 0) & (u_B <= W - 1) & (v_B >= 0) & (v_B <= H - 1)

    # 3. Calculate MAGNITUDE (length of displacement vector, always >= 0)
    magnitude = torch.linalg.norm(inv_flow, dim=1, keepdim=True) # (N, 1, H, W)

    # Round to target pixel grid B
    u_idx = torch.round(u_B).long().clamp(0, W - 1)
    v_idx = torch.round(v_B).long().clamp(0, W - 1)

    max_mag_flat = torch.zeros((N, 1, H * W), device=device, dtype=flow_A2B.dtype)
    flow_B2A_flat = torch.zeros((N, 2, H * W), device=device, dtype=flow_A2B.dtype)

    # 4. Perform Max-Magnitude Scatter
    for b in range(N):
        valid = in_bounds[b].reshape(-1)
        flat_idx = (v_idx[b] * W + u_idx[b]).reshape(-1)[valid]

        mag_vals = magnitude[b, 0].reshape(-1)[valid]
        flow_x_vals = inv_flow[b, 0].reshape(-1)[valid]
        flow_y_vals = inv_flow[b, 1].reshape(-1)[valid]

        # Step A: Find the max MAGNITUDE arriving at each pixel cell in B
        max_mag_flat[b, 0].scatter_reduce_(0, flat_idx, mag_vals, reduce="amax", include_self=True)

        # Step B: Identify points matching/exceeding stored max magnitude
        target_max_mag = max_mag_flat[b, 0, flat_idx]
        is_max = mag_vals >= (target_max_mag - eps)

        # Step C: Write the flow vector (preserving negative/positive sign)
        flat_idx_max = flat_idx[is_max]
        flow_B2A_flat[b, 0].scatter_(0, flat_idx_max, flow_x_vals[is_max])
        flow_B2A_flat[b, 1].scatter_(0, flat_idx_max, flow_y_vals[is_max])

    # 5. Reshape and generate outputs
    flow_B2A = flow_B2A_flat.reshape(N, 2, H, W)
    valid_mask = max_mag_flat > eps
    occlusion_mask = (~valid_mask).reshape(N, 1, H, W).float()

    if is_unbatched:
        flow_B2A = flow_B2A.squeeze(0)
        occlusion_mask = occlusion_mask.squeeze(0)

    return flow_B2A, occlusion_mask

# --- quick test against the repo README example (small) ---
if __name__ == "__main__":
    # small example from README to sanity-check behavior
    forward_flow = np.array([
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]],

        [[0, 2, 0],
         [0, 1, 0],
         [0, 0, 0]],
    ], dtype=np.float32)

    bwd_max, mask_max = max_method(forward_flow)
    bwd_avg, mask_avg = average_method(forward_flow)
    print("max backward:\n", bwd_max)
    print("mask max:\n", mask_max)
    print("avg backward:\n", bwd_avg)
    print("mask avg:\n", mask_avg)
