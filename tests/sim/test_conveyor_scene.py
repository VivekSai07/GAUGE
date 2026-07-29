import mujoco
import numpy as np
from sim.conveyor_scene import ConveyorSceneEnv


def test_scene_loads_and_steps():
    env = ConveyorSceneEnv(conveyor_velocity=np.array([0.0, 0.1, 0.0]))
    env.reset()
    initial_pos = env.get_object_ground_truth().copy()
    for _ in range(50):
        env.step(qdot_cmd=np.zeros(7))
    moved_pos = env.get_object_ground_truth()
    assert moved_pos[1] > initial_pos[1]  # object moved along +y as scripted
    assert env.get_joint_positions().shape == (7,)


def test_rgbd_shapes():
    env = ConveyorSceneEnv(conveyor_velocity=np.array([0.0, 0.1, 0.0]))
    env.reset()
    rgb, depth = env.get_rgbd(width=64, height=64)
    assert rgb.shape == (64, 64, 3)
    assert depth.shape == (64, 64)
    assert depth.dtype == np.float32


def _camera_forward_dot_finger_direction(env: ConveyorSceneEnv) -> float:
    """dot(camera boresight, direction from camera to the finger midpoint).

    Should be close to +1 if the eye-in-hand camera actually looks toward
    the gripper's workspace, and close to -1 if it looks back into the arm
    (a regression of the fixed local-frame mismatch found in code review:
    the camera was mounted with no compensating rotation, so its boresight
    was exactly antiparallel to the direction to the fingers at every pose).
    """
    cam_id = env.model.camera("wrist_cam").id
    cam_pos = env.data.cam_xpos[cam_id]
    cam_mat = env.data.cam_xmat[cam_id].reshape(3, 3)
    cam_forward = -cam_mat[:, 2]  # MuJoCo cameras look along local -z
    left_finger_id = env.model.body("left_finger").id
    right_finger_id = env.model.body("right_finger").id
    finger_mid = 0.5 * (env.data.xpos[left_finger_id] + env.data.xpos[right_finger_id])
    to_fingers = finger_mid - cam_pos
    to_fingers = to_fingers / np.linalg.norm(to_fingers)
    return float(np.dot(cam_forward, to_fingers))


def test_wrist_camera_faces_gripper_workspace():
    env = ConveyorSceneEnv(conveyor_velocity=np.array([0.0, 0.1, 0.0]))
    env.reset()
    assert _camera_forward_dot_finger_direction(env) > 0.9

    # Not pose-dependent: re-check after moving the arm to an arbitrary bent
    # pose, since a fixed local-frame mismatch (the actual regression seen
    # in review) would stay wrong at every pose, not just qpos=0.
    env.data.qpos[:7] = np.array([0.5, -0.3, 0.2, -1.8, 0.1, 1.4, 0.4])
    mujoco.mj_forward(env.model, env.data)
    assert _camera_forward_dot_finger_direction(env) > 0.9


def test_nonzero_qdot_cmd_produces_sustained_joint_motion():
    """A nonzero qdot_cmd must keep moving the joint, not just nudge it to a
    one-time position target. The Panda's arm actuators are position-PD
    servos (ctrl = target angle), not velocity actuators, so step() has to
    integrate qdot_cmd into an internal position setpoint over time; this
    is exactly the behavior later tasks' MPC controller depends on.
    """
    env = ConveyorSceneEnv(conveyor_velocity=np.array([0.0, 0.1, 0.0]))
    env.reset()
    qdot_cmd = np.zeros(7)
    qdot_cmd[0] = 0.3  # rad/s on joint 1

    checkpoints = {}
    for i in range(1, 201):
        env.step(qdot_cmd=qdot_cmd)
        if i in (10, 50, 100, 200):
            checkpoints[i] = env.get_joint_positions()[0]

    # Strictly increasing across widely spaced checkpoints -- a one-time
    # position-target bug would plateau almost immediately instead.
    assert checkpoints[10] < checkpoints[50] < checkpoints[100] < checkpoints[200]
    # After 0.4s at 0.3 rad/s the ideal (lag-free) displacement is 0.12 rad;
    # allow generous PD-tracking lag but require a substantial, clearly
    # nonzero displacement (measured ~0.091 rad).
    assert checkpoints[200] > 0.03


def test_reset_pose_within_joint_limits_with_safety_margin():
    """Task 13's `_RESET_QPOS` is claimed (in its own module docstring and
    task-13-report.md) to sit with >=1.0 rad of margin from every joint's
    real `env.model.jnt_range` limit in both directions (measured minimum:
    1.070 rad, on joint4's lower side). Verify that claim directly against
    the actual compiled model after reset(), rather than trusting the sweep
    narrative alone -- this is the one piece of Task 13's verification that
    had no executable regression test.
    """
    env = ConveyorSceneEnv(conveyor_velocity=np.array([0.0, 0.1, 0.0]))
    env.reset()
    q_reset = env.get_joint_positions()
    jnt_range = env.model.jnt_range[:7]

    margin_to_lower = q_reset - jnt_range[:, 0]
    margin_to_upper = jnt_range[:, 1] - q_reset

    assert np.all(margin_to_lower >= 1.0), margin_to_lower
    assert np.all(margin_to_upper >= 1.0), margin_to_upper


def test_reset_pose_is_stable_under_zero_velocity_command():
    """qpos=0 is outside joint4's own range ([-3.0718, -0.0698]), so
    resetting to all-zeros left the arm drifting under constraint-recovery
    forces even with qdot_cmd=0 every step (a regression found in code
    review: joint4 moved to -0.0389 rad and joint6 to 0.155 rad after just
    20 steps). reset() now loads the model's "home" keyframe, a valid
    resting pose, so zero commanded velocity should hold the arm still
    (up to a small, bounded gravity-sag tracking offset).
    """
    env = ConveyorSceneEnv(conveyor_velocity=np.array([0.0, 0.1, 0.0]))
    env.reset()
    q_before = env.get_joint_positions().copy()
    for _ in range(20):
        env.step(qdot_cmd=np.zeros(7))
    q_after = env.get_joint_positions()
    assert np.max(np.abs(q_after - q_before)) < 0.01
