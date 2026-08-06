import numpy as np

from perception.camera import CameraIntrinsics
from perception.yolo_segment import MODEL_PATH, yolo_centroid
from sim.conveyor_scene import OBJECT_HALF_HEIGHT_M, ConveyorSceneEnv


def _place_cube_in_view(env: ConveyorSceneEnv, x: float, y: float, z: float) -> None:
    """Teleport the conveyor object's true center to (x, y, z), within the
    same envelope experiments/yolo_precision/generate_dataset.py trained on,
    and re-derive the physics state. No settling needed -- this is a
    single-frame render, not a dynamics test."""
    env.set_object_pose([x, y, z])


def test_yolo_centroid_finds_cube_near_ground_truth():
    from ultralytics import YOLO

    env = ConveyorSceneEnv(conveyor_velocity=np.array([0.0, 0.08, 0.0]))
    env.reset()
    _place_cube_in_view(env, x=0.5, y=-0.1, z=0.05)

    model = YOLO(str(MODEL_PATH))
    rgb, depth = env.get_rgbd(64, 64)
    fx, fy, cx, cy = env.camera_intrinsics(64, 64)
    intrinsics = CameraIntrinsics(fx, fy, cx, cy)
    cam_id = env.model.camera("wrist_cam").id
    cam_pos = env.data.cam_xpos[cam_id].copy()
    cam_mat = env.data.cam_xmat[cam_id].reshape(3, 3).copy()

    # yolo_centroid, unlike perception.segment.segment_object_centroid,
    # returns a point already in world coordinates -- it needs the camera
    # pose internally to apply the world-Z depth_bias offset (design spec
    # 2.3), so it takes cam_pos/cam_mat and does the camera->world
    # transform itself instead of leaving it to the caller.
    result = yolo_centroid(
        rgb,
        depth,
        model,
        intrinsics,
        color_lower=(150, 0, 0),
        color_upper=(255, 80, 80),
        cam_pos=cam_pos,
        cam_mat=cam_mat,
        depth_bias=OBJECT_HALF_HEIGHT_M,
    )

    assert result is not None
    truth = env.get_object_ground_truth()
    np.testing.assert_allclose(result, truth, atol=0.05)


def test_yolo_centroid_returns_none_when_no_cube_in_view():
    from ultralytics import YOLO

    model = YOLO(str(MODEL_PATH))
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    depth = np.full((64, 64), 1.5, dtype=np.float32)
    intrinsics = CameraIntrinsics(fx=64.0, fy=64.0, cx=32.0, cy=32.0)

    # No detection means the function returns before touching cam_pos/cam_mat
    # at all, so their values here are irrelevant -- dummies are fine.
    result = yolo_centroid(
        rgb,
        depth,
        model,
        intrinsics,
        color_lower=(150, 0, 0),
        color_upper=(255, 80, 80),
        cam_pos=np.zeros(3),
        cam_mat=np.eye(3),
    )
    assert result is None


def test_yolo_centroid_returns_none_when_box_found_but_color_mask_empty():
    """A box exists but zero pixels inside it match the color threshold --
    this is the case that took the earlier validation experiment
    (experiments/yolo_precision/evaluate.py's _yolo_bbox_center_to_3d) from
    NO-GO to GO (see perception/yolo_segment.py's module docstring). It's
    distinct from the zero-boxes case already covered above, and must not
    be simplified into a full-box-mean fallback."""
    from ultralytics import YOLO

    env = ConveyorSceneEnv(conveyor_velocity=np.array([0.0, 0.08, 0.0]))
    env.reset()
    _place_cube_in_view(env, x=0.5, y=-0.1, z=0.05)

    model = YOLO(str(MODEL_PATH))
    rgb, depth = env.get_rgbd(64, 64)
    fx, fy, cx, cy = env.camera_intrinsics(64, 64)
    intrinsics = CameraIntrinsics(fx, fy, cx, cy)
    cam_id = env.model.camera("wrist_cam").id
    cam_pos = env.data.cam_xpos[cam_id].copy()
    cam_mat = env.data.cam_xmat[cam_id].reshape(3, 3).copy()

    # Prove the detector actually found something, so the None checked
    # below can only have come from the color-mask branch, not this one.
    results = model.predict(source=rgb, verbose=False)[0]
    assert len(results.boxes) >= 1

    result = yolo_centroid(
        rgb,
        depth,
        model,
        intrinsics,
        color_lower=(0, 200, 0),
        color_upper=(0, 255, 0),
        cam_pos=cam_pos,
        cam_mat=cam_mat,
    )
    assert result is None


def test_yolo_centroid_rejects_border_clipped_detection():
    """A detection touching the image edge has a provably unreliable
    centroid: the object's true centre projects off-image and the measured
    centroid is clamped inward (design spec 2.2 -- measured max residual
    11.2mm vs 2.5mm once rejected). Treat it as a miss.

    y = -0.15 is far enough into the frame that every other branch (box
    found, color mask non-empty) still succeeds -- confirmed below by the
    reject_border=False call returning a real result -- so the None from
    reject_border=True below can only have come from the border-rejection
    branch under test. (At the more extreme y=-0.20 used previously,
    reject_border=False ALSO returns None because the detection is clipped
    badly enough to fail earlier branches too, which made that version of
    this test pass even with border rejection deleted entirely.)"""
    from ultralytics import YOLO

    env = ConveyorSceneEnv(conveyor_velocity=np.array([0.0, 0.08, 0.0]))
    env.reset()
    # y = -0.15 puts the cube near the edge of the wrist camera's view --
    # clipped enough to trigger border rejection, not clipped so badly that
    # other branches would already return None on their own.
    _place_cube_in_view(env, x=0.5, y=-0.15, z=0.05)
    model = YOLO(str(MODEL_PATH))
    rgb, depth = env.get_rgbd(64, 64)
    fx, fy, cx, cy = env.camera_intrinsics(64, 64)
    intrinsics = CameraIntrinsics(fx, fy, cx, cy)
    cam_id = env.model.camera("wrist_cam").id
    cam_pos = env.data.cam_xpos[cam_id].copy()
    cam_mat = env.data.cam_xmat[cam_id].reshape(3, 3).copy()
    args = (
        rgb,
        depth,
        model,
        intrinsics,
        (150, 0, 0),
        (255, 80, 80),
        cam_pos,
        cam_mat,
    )

    # Prove the detector genuinely finds and localizes the cube at this
    # pose, so the None below can only have come from the border-rejection
    # branch, not from the detection or color-mask branches failing first.
    unrejected = yolo_centroid(
        *args, depth_bias=OBJECT_HALF_HEIGHT_M, reject_border=False
    )
    assert unrejected is not None

    assert (
        yolo_centroid(*args, depth_bias=OBJECT_HALF_HEIGHT_M, reject_border=True)
        is None
    )
