import mujoco
import numpy as np

from control.panda_kinematics import (
    camera_pose_numpy,
    camera_pose_symbolic,
    panda_fk_numpy,
    panda_fk_symbolic,
    panda_tcp_numpy,
    panda_tcp_pose_numpy,
    panda_tcp_pose_symbolic,
    panda_tcp_symbolic,
)
from sim.conveyor_scene import ConveyorSceneEnv


def test_casadi_fk_matches_independent_numpy_fk_at_zero_config():
    fk = panda_fk_symbolic()
    q = np.zeros(7)
    casadi_result = np.array(fk(q)).flatten()
    numpy_result = panda_fk_numpy(q)
    np.testing.assert_allclose(casadi_result, numpy_result, atol=1e-9)


def test_casadi_fk_matches_independent_numpy_fk_at_random_config():
    rng = np.random.default_rng(7)
    fk = panda_fk_symbolic()
    for _ in range(5):
        q = rng.uniform(-1.5, 1.5, size=7)
        casadi_result = np.array(fk(q)).flatten()
        numpy_result = panda_fk_numpy(q)
        np.testing.assert_allclose(casadi_result, numpy_result, atol=1e-6)


def test_fk_output_is_finite_and_reasonable_reach():
    fk = panda_fk_symbolic()
    q = np.zeros(7)
    pos = np.array(fk(q)).flatten()
    assert np.all(np.isfinite(pos))
    # Panda's reach is roughly within 1m of the base.
    assert np.linalg.norm(pos) < 1.5


def test_fk_numpy_matches_mujoco_conveyor_scene_hand_body():
    """Ground-truth cross-check against the real simulated Panda (Task 2's
    MuJoCo scene): panda_fk_numpy must reproduce the compiled model's "hand"
    body position, not just agree with panda_fk_symbolic.

    This guards against the Task 12 integration finding that the DH `_A`
    array was off by one index -- a bug that agrees with itself (casadi vs
    numpy) and even reproduces the correct position at q=0, but diverges by
    up to 0.245 m at realistic bent poses such as the conveyor scene's home
    keyframe. Only a check against the independently-authored MJCF model
    catches that class of bug.
    """
    env = ConveyorSceneEnv(conveyor_velocity=np.array([0.0, 0.08, 0.0]))
    env.reset()
    hand_id = env.model.body("hand").id

    configs = [np.zeros(7), env.get_joint_positions().copy()]
    rng = np.random.default_rng(3)
    configs += [rng.uniform(-1.2, 1.2, size=7) for _ in range(5)]

    for q in configs:
        env.data.qpos[:7] = q
        mujoco.mj_forward(env.model, env.data)
        mj_pos = env.data.xpos[hand_id].copy()
        fk_pos = panda_fk_numpy(q)
        np.testing.assert_allclose(fk_pos, mj_pos, atol=1e-6)


def test_casadi_tcp_matches_independent_numpy_tcp():
    tcp = panda_tcp_symbolic()
    rng = np.random.default_rng(11)
    for q in [np.zeros(7)] + [rng.uniform(-1.5, 1.5, size=7) for _ in range(5)]:
        casadi_result = np.array(tcp(q)).flatten()
        numpy_result = panda_tcp_numpy(q)
        np.testing.assert_allclose(casadi_result, numpy_result, atol=1e-6)


def test_tcp_numpy_matches_mujoco_fingertip_pad_midpoint():
    """Ground-truth cross-check: panda_tcp_numpy must reproduce the real
    compiled model's fingertip-pad midpoint (average of both fingers' small
    pad collision geoms), not just an assumed offset -- same discipline as
    the flange check above, since a wrong _TCP_OFFSET_Z would agree with
    itself but not with the real geometry.

    This Menagerie revision's finger pad geoms are unnamed. They were
    identified by inspecting every geom attached to the left_finger/
    right_finger bodies and finding the small (~0.0085m half-size) geom at
    local pos [0, 0.0055, 0.0445] on each -- the fingertip contact pad, as
    opposed to the larger surrounding finger-body collision geoms. Geom
    indices 69 (left) and 77 (right) confirmed by direct model inspection
    (see control/panda_kinematics.py's _TCP_OFFSET_Z derivation).
    """
    env = ConveyorSceneEnv(conveyor_velocity=np.array([0.0, 0.08, 0.0]))
    env.reset()
    lf_pad_geom_id = 69
    rf_pad_geom_id = 77
    # Guard against a future Menagerie update silently changing which geoms
    # these indices refer to.
    assert env.model.geom_bodyid[lf_pad_geom_id] == env.model.body("left_finger").id
    assert env.model.geom_bodyid[rf_pad_geom_id] == env.model.body("right_finger").id
    assert np.allclose(
        env.model.geom_pos[lf_pad_geom_id], [0.0, 0.0055, 0.0445], atol=1e-4
    )
    assert np.allclose(
        env.model.geom_pos[rf_pad_geom_id], [0.0, 0.0055, 0.0445], atol=1e-4
    )

    configs = [np.zeros(7), env.get_joint_positions().copy()]
    rng = np.random.default_rng(13)
    configs += [rng.uniform(-1.2, 1.2, size=7) for _ in range(5)]

    for q in configs:
        env.data.qpos[:7] = q
        mujoco.mj_forward(env.model, env.data)
        mj_tcp = (
            env.data.geom_xpos[lf_pad_geom_id] + env.data.geom_xpos[rf_pad_geom_id]
        ) / 2.0
        tcp_pos = panda_tcp_numpy(q)
        np.testing.assert_allclose(tcp_pos, mj_tcp, atol=1e-6)


def test_casadi_tcp_pose_matches_independent_numpy_tcp_pose():
    pose_fn = panda_tcp_pose_symbolic()
    rng = np.random.default_rng(17)
    for q in [np.zeros(7)] + [rng.uniform(-1.5, 1.5, size=7) for _ in range(5)]:
        casadi_pos, casadi_rot = pose_fn(q)
        numpy_pos, numpy_rot = panda_tcp_pose_numpy(q)
        np.testing.assert_allclose(np.array(casadi_pos).flatten(), numpy_pos, atol=1e-6)
        np.testing.assert_allclose(np.array(casadi_rot), numpy_rot, atol=1e-6)


def test_tcp_pose_numpy_matches_mujoco_hand_orientation():
    """Ground-truth cross-check: panda_tcp_pose_numpy's rotation must
    reproduce MuJoCo's real compiled "hand" body orientation (env.data.xmat),
    not just an assumed correction -- same discipline as the position
    checks above. Also verifies the empirical claim `panda_tcp_pose_symbolic`
    documents: the finger-closing direction (right_finger body - left_finger
    body, in world coordinates) is exactly the local Y axis (rotation[:, 1])
    of the returned rotation, at every configuration tested."""
    env = ConveyorSceneEnv(conveyor_velocity=np.array([0.0, 0.08, 0.0]))
    env.reset()
    hand_id = env.model.body("hand").id
    lf_id = env.model.body("left_finger").id
    rf_id = env.model.body("right_finger").id

    configs = [np.zeros(7), env.get_joint_positions().copy()]
    rng = np.random.default_rng(19)
    configs += [rng.uniform(-1.2, 1.2, size=7) for _ in range(5)]

    for q in configs:
        env.data.qpos[:7] = q
        mujoco.mj_forward(env.model, env.data)
        mj_rot = env.data.xmat[hand_id].reshape(3, 3).copy()
        _, rot = panda_tcp_pose_numpy(q)
        np.testing.assert_allclose(rot, mj_rot, atol=1e-6)

        closing_dir_world = env.data.xpos[rf_id] - env.data.xpos[lf_id]
        closing_dir_world /= np.linalg.norm(closing_dir_world)
        local_y = rot[:, 1]
        # Anti-parallel is fine (left/right labeling is arbitrary); only the
        # axis, not the sign, matters for the lateral-offset cost.
        assert abs(abs(np.dot(closing_dir_world, local_y)) - 1.0) < 1e-4


def test_casadi_camera_pose_matches_independent_numpy_camera_pose():
    pose_fn = camera_pose_symbolic()
    rng = np.random.default_rng(23)
    for q in [np.zeros(7)] + [rng.uniform(-1.5, 1.5, size=7) for _ in range(5)]:
        casadi_pos, casadi_fwd = pose_fn(q)
        numpy_pos, numpy_fwd = camera_pose_numpy(q)
        np.testing.assert_allclose(np.array(casadi_pos).flatten(), numpy_pos, atol=1e-6)
        np.testing.assert_allclose(np.array(casadi_fwd).flatten(), numpy_fwd, atol=1e-6)


def test_camera_forward_is_unit_length():
    rng = np.random.default_rng(29)
    for q in [np.zeros(7)] + [rng.uniform(-1.5, 1.5, size=7) for _ in range(5)]:
        _, fwd = camera_pose_numpy(q)
        assert abs(np.linalg.norm(fwd) - 1.0) < 1e-9


def test_camera_pose_numpy_matches_mujoco_wrist_cam():
    """Ground-truth cross-check against the real compiled MuJoCo camera --
    same discipline as panda_tcp_pose_numpy's hand-orientation check.
    camera_pos must match data.cam_xpos; camera_forward (this module's
    boresight convention) must match MuJoCo's own camera-forward
    convention, -cam_xmat[:, 2] (a MuJoCo camera looks down its local -Z
    axis; cam_xmat's columns are the camera's local axes in world frame)."""
    env = ConveyorSceneEnv(conveyor_velocity=np.array([0.0, 0.08, 0.0]))
    env.reset()
    cam_id = env.model.camera("wrist_cam").id

    configs = [np.zeros(7), env.get_joint_positions().copy()]
    rng = np.random.default_rng(31)
    configs += [rng.uniform(-1.2, 1.2, size=7) for _ in range(5)]

    for q in configs:
        env.data.qpos[:7] = q
        mujoco.mj_forward(env.model, env.data)
        mj_cam_pos = env.data.cam_xpos[cam_id].copy()
        mj_cam_mat = env.data.cam_xmat[cam_id].reshape(3, 3).copy()
        mj_cam_forward = -mj_cam_mat[:, 2]

        cam_pos, cam_forward = camera_pose_numpy(q)
        np.testing.assert_allclose(cam_pos, mj_cam_pos, atol=1e-6)
        np.testing.assert_allclose(cam_forward, mj_cam_forward, atol=1e-6)
