import numpy as np
from perception.camera import CameraIntrinsics
from perception.segment import segment_object_centroid


def test_segment_finds_red_square():
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[20:30, 20:30] = [200, 20, 20]  # red square, rows=v, cols=u
    depth = np.full((64, 64), 1.5, dtype=np.float32)
    intr = CameraIntrinsics(fx=64.0, fy=64.0, cx=32.0, cy=32.0)

    centroid = segment_object_centroid(
        rgb, depth, intr, color_lower=(150, 0, 0), color_upper=(255, 80, 80)
    )

    assert centroid is not None
    expected_u, expected_v = 24.5, 24.5  # center of rows/cols 20..29
    expected = intr.deproject(expected_u, expected_v, 1.5)
    np.testing.assert_allclose(centroid, expected, atol=0.05)


def test_segment_returns_none_when_no_match():
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    depth = np.full((64, 64), 1.5, dtype=np.float32)
    intr = CameraIntrinsics(fx=64.0, fy=64.0, cx=32.0, cy=32.0)

    centroid = segment_object_centroid(
        rgb, depth, intr, color_lower=(150, 0, 0), color_upper=(255, 80, 80)
    )
    assert centroid is None
