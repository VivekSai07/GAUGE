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

Extended per design spec docs/superpowers/specs/2026-08-04-rendezvous-grasp-design.md
(Section 2.2/2.3) with three further miss/correction cases, each also a
genuine "don't guess" response to a measured failure mode, not a
simplification of the above:

- Border-clipped detections are rejected (None), since the object's true
  centre projects off-image and the measured centroid is clamped inward --
  a known-bad measurement that dead-reckoning handles strictly better than
  integrating (measured max residual 11.2mm vs 2.5mm once rejected).
- The centroid is computed from the top face only (color-mask pixels within
  `_TOP_FACE_DEPTH_TOL_M` of the nearest depth), not the whole box, so it
  isn't pulled toward the optical axis by a visible side face.
- `depth_bias` is subtracted along world Z after deprojection, not added to
  the measured depth before it -- the sensed point is on the top face, and
  its centre is half a cube-height below that in world Z, not along the
  camera ray. This makes the function's contract world-frame, so it now
  requires the camera pose (`cam_pos`/`cam_mat`).
"""

from pathlib import Path

import numpy as np

from perception.camera import CameraIntrinsics, camera_point_to_world

MODEL_PATH = Path(__file__).parent / "models" / "cube_detector.pt"

# Colour-mask pixels within this many meters of the nearest (top-face) depth
# are kept for the centroid; pixels on a visible side face sit farther away
# and are excluded. See module docstring and design spec Section 2.2.
_TOP_FACE_DEPTH_TOL_M = 0.010


def yolo_centroid(
    rgb: np.ndarray,
    depth: np.ndarray,
    model,
    intrinsics: CameraIntrinsics,
    color_lower: tuple[int, int, int],
    color_upper: tuple[int, int, int],
    cam_pos: np.ndarray,
    cam_mat: np.ndarray,
    depth_bias: float = 0.0,
    reject_border: bool = True,
) -> np.ndarray | None:
    results = model.predict(source=rgb, verbose=False)[0]
    if len(results.boxes) == 0:
        return None
    confs = results.boxes.conf.cpu().numpy()
    best = int(np.argmax(confs))
    x_min, y_min, x_max, y_max = results.boxes.xyxy[best].cpu().numpy()

    height_px, width_px = depth.shape
    if reject_border and (
        x_min <= 1 or y_min <= 1 or x_max >= width_px - 1 or y_max >= height_px - 1
    ):
        return None

    y0, y1 = int(max(0, y_min)), int(min(height_px, y_max + 1))
    x0, x1 = int(max(0, x_min)), int(min(width_px, x_max + 1))
    region_rgb, region_depth = rgb[y0:y1, x0:x1], depth[y0:y1, x0:x1]
    lower = np.array(color_lower, dtype=np.uint8)
    upper = np.array(color_upper, dtype=np.uint8)
    mask = np.all((region_rgb >= lower) & (region_rgb <= upper), axis=-1)
    if mask.sum() == 0:
        return None

    # The top face is nearest the camera; restrict to it so the centroid is
    # the face's centre (whose x/y equal the cube's) rather than a silhouette
    # centre pulled toward the optical axis by a visible side face.
    top_face = mask & (
        region_depth <= float(region_depth[mask].min()) + _TOP_FACE_DEPTH_TOL_M
    )
    if top_face.sum() == 0:
        return None
    ys, xs = np.nonzero(top_face)
    u = x0 + float(xs.mean())
    v = y0 + float(ys.mean())
    z = float(region_depth[top_face].mean())

    point_cam = intrinsics.deproject(u, v, z)
    point_world = camera_point_to_world(point_cam, cam_pos, cam_mat)
    # The measured point is on the TOP FACE; the centre is depth_bias below it
    # in WORLD Z -- not along the camera ray (design spec 2.3).
    return point_world - np.array([0.0, 0.0, depth_bias])
