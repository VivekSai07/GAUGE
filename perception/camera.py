"""Pinhole camera intrinsics and depth back-projection."""

import numpy as np


class CameraIntrinsics:
    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy

    def deproject(self, u: int, v: int, depth: float) -> np.ndarray:
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth
        return np.array([x, y, z], dtype=np.float64)


def camera_point_to_world(
    point_cam: np.ndarray, cam_pos: np.ndarray, cam_mat: np.ndarray
) -> np.ndarray:
    """Transform a point from the pinhole camera frame (x=right, y=down,
    z=forward-depth) into world coordinates, given the camera's world
    position and its local-axes-in-world rotation matrix (MuJoCo's
    `cam_xmat`, reshaped to 3x3, columns = local x/y/z in world frame).
    """
    point_local = np.array([point_cam[0], -point_cam[1], -point_cam[2]])
    return cam_pos + cam_mat @ point_local
