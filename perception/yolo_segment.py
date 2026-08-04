"""YOLO-based 2D detection + color-masked depth sampling to a 3D centroid.

A direct port of experiments/yolo_precision/evaluate.py's
_yolo_bbox_center_to_3d (commit 567297c, the version that took the
validation experiment from NO-GO to GO -- see design spec
docs/superpowers/specs/2026-08-02-yolo-detector-precision-validation-design.md,
Section 7). The detector supplies the 2D pixel location (bounding box
center); the same color threshold perception/segment.py already uses
gates which pixels inside that box count for depth -- a zero-match is a
genuine miss (None), not a fallback estimate. That exact zero-fallback
semantic is what fixed the experiment's original full-box-mean approach,
and must not be simplified away here.
"""

from pathlib import Path

import numpy as np

from perception.camera import CameraIntrinsics

MODEL_PATH = Path(__file__).parent / "models" / "cube_detector.pt"


def yolo_centroid(
    rgb: np.ndarray,
    depth: np.ndarray,
    model,
    intrinsics: CameraIntrinsics,
    color_lower: tuple[int, int, int],
    color_upper: tuple[int, int, int],
    depth_bias: float = 0.0,
) -> np.ndarray | None:
    results = model.predict(source=rgb, verbose=False)[0]
    if len(results.boxes) == 0:
        return None
    confs = results.boxes.conf.cpu().numpy()
    best = int(np.argmax(confs))
    x_min, y_min, x_max, y_max = results.boxes.xyxy[best].cpu().numpy()
    u = (x_min + x_max) / 2.0
    v = (y_min + y_max) / 2.0
    y0, y1 = int(max(0, y_min)), int(min(depth.shape[0], y_max + 1))
    x0, x1 = int(max(0, x_min)), int(min(depth.shape[1], x_max + 1))
    region_rgb = rgb[y0:y1, x0:x1]
    region_depth = depth[y0:y1, x0:x1]
    lower = np.array(color_lower, dtype=np.uint8)
    upper = np.array(color_upper, dtype=np.uint8)
    mask = np.all((region_rgb >= lower) & (region_rgb <= upper), axis=-1)
    if mask.sum() == 0:
        return None
    z = float(region_depth[mask].mean()) + depth_bias
    return intrinsics.deproject(u, v, z)
