import mujoco
import numpy as np

from perception.camera import CameraIntrinsics
from perception.yolo_segment import MODEL_PATH, yolo_centroid
from run_conveyor_demo import _camera_point_to_world
from sim.conveyor_scene import OBJECT_HALF_HEIGHT_M, ConveyorSceneEnv


def _place_cube_in_view(env: ConveyorSceneEnv, x: float, y: float, z: float) -> None:
    """Teleport the conveyor object to a known pose within the same
    x/y/z envelope experiments/yolo_precision/generate_dataset.py trained
    on, and re-derive the physics state. No settling needed -- this is a
    single-frame render, not a dynamics test."""
    obj_jid = env.model.body("conveyor_object").jntadr[0]
    obj_qpos_addr = env.model.jnt_qposadr[obj_jid]
    env.data.qpos[obj_qpos_addr : obj_qpos_addr + 3] = [x, y, z]
    env.data.qpos[obj_qpos_addr + 3 : obj_qpos_addr + 7] = [1.0, 0.0, 0.0, 0.0]
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)


def test_yolo_centroid_finds_cube_near_ground_truth():
    from ultralytics import YOLO

    env = ConveyorSceneEnv(conveyor_velocity=np.array([0.0, 0.08, 0.0]))
    env.reset()
    _place_cube_in_view(env, x=0.5, y=-0.1, z=0.05)

    model = YOLO(str(MODEL_PATH))
    rgb, depth = env.get_rgbd(64, 64)
    fx, fy, cx, cy = env.camera_intrinsics(64, 64)
    intrinsics = CameraIntrinsics(fx, fy, cx, cy)

    result_cam = yolo_centroid(
        rgb,
        depth,
        model,
        intrinsics,
        color_lower=(150, 0, 0),
        color_upper=(255, 80, 80),
        depth_bias=OBJECT_HALF_HEIGHT_M,
    )

    assert result_cam is not None
    # yolo_centroid, like perception.segment.segment_object_centroid, returns
    # a point in the pinhole camera frame -- every existing caller
    # (experiments/yolo_precision/evaluate.py, run_conveyor_demo.py) converts
    # to world coordinates via _camera_point_to_world before comparing
    # against ground truth, since wrist_cam moves with the arm and isn't at
    # the world origin.
    cam_id = env.model.camera("wrist_cam").id
    cam_pos = env.data.cam_xpos[cam_id].copy()
    cam_mat = env.data.cam_xmat[cam_id].reshape(3, 3).copy()
    result = _camera_point_to_world(result_cam, cam_pos, cam_mat)

    truth = env.get_object_ground_truth()
    np.testing.assert_allclose(result, truth, atol=0.05)


def test_yolo_centroid_returns_none_when_no_cube_in_view():
    from ultralytics import YOLO

    model = YOLO(str(MODEL_PATH))
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    depth = np.full((64, 64), 1.5, dtype=np.float32)
    intrinsics = CameraIntrinsics(fx=64.0, fy=64.0, cx=32.0, cy=32.0)

    result = yolo_centroid(
        rgb, depth, model, intrinsics, color_lower=(150, 0, 0), color_upper=(255, 80, 80)
    )
    assert result is None
