import mujoco
import numpy as np
from control.panda_kinematics import panda_fk_symbolic, panda_fk_numpy
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
