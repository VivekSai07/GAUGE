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
