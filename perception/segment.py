"""Classical color+depth segmentation to a 3D centroid."""
import numpy as np

from perception.camera import CameraIntrinsics


def segment_object_centroid(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    color_lower: tuple[int, int, int],
    color_upper: tuple[int, int, int],
) -> np.ndarray | None:
    lower = np.array(color_lower, dtype=np.uint8)
    upper = np.array(color_upper, dtype=np.uint8)
    mask = np.all((rgb >= lower) & (rgb <= upper), axis=-1)

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None

    u = float(xs.mean())
    v = float(ys.mean())
    z = float(depth[ys, xs].mean())
    return intrinsics.deproject(u, v, z)
