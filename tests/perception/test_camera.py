import numpy as np
from perception.camera import CameraIntrinsics


def test_deproject_center_pixel():
    intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=32.0, cy=32.0)
    point = intr.deproject(u=32, v=32, depth=2.0)
    np.testing.assert_allclose(point, [0.0, 0.0, 2.0], atol=1e-9)


def test_deproject_off_center_pixel():
    intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=32.0, cy=32.0)
    point = intr.deproject(u=42, v=32, depth=2.0)
    # x = (u - cx) * depth / fx
    expected_x = (42 - 32) * 2.0 / 100.0
    np.testing.assert_allclose(point, [expected_x, 0.0, 2.0], atol=1e-9)
