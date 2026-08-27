from dataclasses import dataclass, field
from typing import Literal, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float

from misc.Mytypes import Batch
from .loss import Loss, LossCfgCommon


@dataclass
class LossSemanticScaleCfg(LossCfgCommon):
    name: Literal["semantic_scale"]
    # Map COCO class IDs (int) or class names to average height (meters)
    # COCO Class 0 = Person (1.70m), Class 2 = Car (1.50m)
    class_heights: dict
    confidence_threshold: float = 0.7


class LossSemanticScale(Loss[LossSemanticScaleCfg]):
    def __init__(self, cfg: LossSemanticScaleCfg, targets: int, *args) -> None:
        super().__init__(cfg, targets)
        self.class_heights = cfg.class_heights
        self.weight = getattr(cfg, 'weight', 1.0)

    def compute_unweighted_loss(
            self,
            batch: Batch,
            global_step: int,
            cameras,
    ) -> Float[Tensor, ""]:
        """
        Computes semantic scale anchoring loss using detected objects stored in batch.objects.
        batch.objects structure: [camera_idx][batch_img_idx] -> [(cls_id, [x1, y1, x2, y2]), ...]
        """
        losses = torch.tensor(0.0, device=batch.images[0].device)

        # Check if objects exist for the target camera index
        if not hasattr(batch, 'objects') or batch.objects is None or batch.objects[self.targets] is None:
            return losses

        target_cam_objects = batch.objects[self.targets]  # List of detections per batch item
        target_img = batch.images[self.targets]  # [B, C, H, W]
        img_height_px = target_img.shape[-2]

        # Regressed vertical FOV for target camera
        pred_fy = cameras.cameras[self.targets].fy  # in pixels

        # Retrieve depth map for depth-based distance estimation (e.g., from projections or depth attribute)
        target_depth = batch.depths[self.targets]

        # Process each item in the batch
        batch_size = target_img.shape[0]
        for b_idx in range(batch_size):
            detections = target_cam_objects[b_idx]  # List of tuples: (cls_id, [x1, y1, x2, y2])
            if not detections:
                continue

            item_depth = target_depth[b_idx] if target_depth is not None else None

            loss = self.compute_loss(
                pred_fy=pred_fy,
                detected_objects=detections,
                img_height_px=img_height_px,
                depth_map=item_depth
            )
            losses += loss

        return losses / max(batch_size, 1)

    def compute_loss(
            self,
            pred_fy: Tensor,
            detected_objects: list,
            img_height_px: int,
            depth_map: Tensor = None
    ) -> Tensor:
        """
        Calculates focal length discrepancy from 2D bounding boxes and depth maps.
        f_implied = (h_px * Z) / H_real
        """
        device = pred_fy.device
        scale_errors = []

        for cls_id, bbox in detected_objects:
            if cls_id not in self.class_heights:
                continue

            x1, y1, x2, y2 = bbox
            h_px = y2 - y1  # Bounding box height in pixels

            if h_px <= 0:
                continue

            h_real_prior = self.class_heights[cls_id]  # Physical prior (meters)

            # Estimate depth Z from the depth map at the object center if available
            if depth_map is not None:
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                # Clamp coordinates within valid image bounds
                cy = min(max(cy, 0), depth_map.shape[-2] - 1)
                cx = min(max(cx, 0), depth_map.shape[-1] - 1)
                z_depth = depth_map[..., cy, cx]
            else:
                # Fallback unit depth if depth map is unattached
                z_depth = torch.tensor(1.0, device=device)

            # Implied focal length from pinhole projection formula
            f_implied = (h_px * z_depth) / h_real_prior

            # Scale difference between regressed and object-implied focal length
            scale_errors.append(torch.abs(pred_fy - f_implied)/f_implied)

        if len(scale_errors) == 0:
            return torch.tensor(0.0, device=device)

        loss_tot = torch.stack(scale_errors).mean() * self.weight
        return loss_tot