"""Classical color+depth segmentation to a 3D centroid."""

import numpy as np

from perception.camera import CameraIntrinsics


def segment_object_centroid(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    color_lower: tuple[int, int, int],
    color_upper: tuple[int, int, int],
    depth_bias: float = 0.0,
) -> np.ndarray | None:
    """Segment by color, then deproject the masked pixels' centroid to 3D.

    ``depth_bias`` (additive, default 0.0 -- backward-compatible, existing
    callers/tests are unaffected): a top-down-viewing camera measures depth
    to the *visible top face* of a box-shaped object, not its volumetric
    center, which is `depth_bias` further along the camera's viewing ray.
    This is a systematic, geometry-derived offset (not noise) -- see design
    spec Section 12 -- and is corrected here, before deprojection, so it
    also proportionally corrects the deprojected x/y (which scale with
    depth), not just z. Callers that know the object's true half-height
    (e.g. `run_conveyor_demo.py`, via `sim.conveyor_scene.OBJECT_HALF_HEIGHT_M`)
    should pass it here; callers that don't care (e.g. unit tests using a
    synthetic flat-depth image) can leave it at the default 0.0.
    """
    lower = np.array(color_lower, dtype=np.uint8)
    upper = np.array(color_upper, dtype=np.uint8)
    mask = np.all((rgb >= lower) & (rgb <= upper), axis=-1)

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None

    u = float(xs.mean())
    v = float(ys.mean())
    z = float(depth[ys, xs].mean()) + depth_bias
    return intrinsics.deproject(u, v, z)
