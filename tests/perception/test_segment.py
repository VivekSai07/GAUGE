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


def test_segment_depth_bias_shifts_centroid_along_camera_ray():
    """depth_bias (added post-Task-14 to correct the top-face-vs-volumetric-
    center systematic Z bias, see design spec Section 12) must be applied to
    the raw depth *before* deprojection, so it also proportionally corrects
    x/y for off-center pixels, not just z. Default 0.0 must reproduce the
    original (pre-bias-fix) behavior exactly, for backward compatibility.
    """
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[20:30, 20:30] = [200, 20, 20]
    depth = np.full((64, 64), 1.5, dtype=np.float32)
    intr = CameraIntrinsics(fx=64.0, fy=64.0, cx=32.0, cy=32.0)

    baseline = segment_object_centroid(
        rgb, depth, intr, color_lower=(150, 0, 0), color_upper=(255, 80, 80)
    )
    biased = segment_object_centroid(
        rgb,
        depth,
        intr,
        color_lower=(150, 0, 0),
        color_upper=(255, 80, 80),
        depth_bias=0.02,
    )

    # Default depth_bias=0.0 is unchanged from the original, pre-fix behavior.
    expected_u, expected_v = 24.5, 24.5
    expected_baseline = intr.deproject(expected_u, expected_v, 1.5)
    np.testing.assert_allclose(baseline, expected_baseline, atol=0.05)

    # A nonzero depth_bias is applied to depth *before* deprojection, so both
    # z (directly) and x/y (proportionally, since they scale with depth)
    # shift -- not just z.
    expected_biased = intr.deproject(expected_u, expected_v, 1.5 + 0.02)
    np.testing.assert_allclose(biased, expected_biased, atol=0.05)
    assert biased[2] > baseline[2]
    assert not np.allclose(biased[:2], baseline[:2])


def test_segment_returns_none_when_no_match():
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    depth = np.full((64, 64), 1.5, dtype=np.float32)
    intr = CameraIntrinsics(fx=64.0, fy=64.0, cx=32.0, cy=32.0)

    centroid = segment_object_centroid(
        rgb, depth, intr, color_lower=(150, 0, 0), color_upper=(255, 80, 80)
    )
    assert centroid is None
