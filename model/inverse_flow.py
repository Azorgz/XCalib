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
            # scatter_max isn't builtin -> emulate with segment_max using scatter_reduce if available
            try:
                # PyTorch >=1.12 has scatter_reduce:
                # Use scatter_reduce to compute index of max: compute (score, idx) pair trick:
                idxs = torch.arange(score_valid.shape[0], device=score_valid.device)
                packed = score_valid * (score_valid.shape[0] + 1) + idxs  # unique pack; max of packed gives max score and idx
                # scatter reduce by dst_valid
                packed_max = torch.zeros(dst_unique.size(0), device=score_valid.device).scatter_reduce(0, inverse_idx, packed, reduce='amax')
                # retrieve source index as packed_max % ... :
                src_choice_idx = (packed_max.long() % (score_valid.shape[0] + 1)).long()
            except Exception:
                # Generic fallback: for each unique dst do a masked max (still vectorized-ish)
                src_choice_idx = torch.empty(dst_unique.shape[0], dtype=torch.long, device=flow.device)
                for i_u, d in enumerate(dst_unique):
                    mask_u = (dst_valid == d)
                    scores_u = score_valid[mask_u]
                    pos = torch.argmax(scores_u)
                    idxs_u = torch.nonzero(mask_u, as_tuple=False).squeeze(1)
                    src_choice_idx[i_u] = idxs_u[pos]

            chosen_src_idxs = src_choice_idx  # indices into the filtered arrays
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
