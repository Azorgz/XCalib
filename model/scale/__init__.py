from typing import List
from ultralytics import YOLO

# Lightweight detector registry
DETECTORS = {
    "yolo": "yolo26s.pt"
}

# COCO Dataset class indices for person (0) and car (2)
TARGET_CLASSES: List[int] = [0, 2]


def get_detector() -> YOLO:
    detector_path = DETECTORS["yolo"]
    model = YOLO(detector_path)

    # Pre-configure model overrides to only predict person and car classes
    model.overrides["classes"] = TARGET_CLASSES

    return model