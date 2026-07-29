import numpy as np
from control.panda_kinematics import panda_fk_symbolic, panda_fk_numpy
from control.mpc import KinematicMPC


def test_mpc_step_reduces_distance_to_reachable_target():
    fk = panda_fk_symbolic()
    q_current = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.7])
    current_pos = panda_fk_numpy(q_current)

    # Nudge the target slightly from the current EE position -- reachable.
    target = current_pos + np.array([0.03, 0.0, 0.0])

    mpc = KinematicMPC(
        fk_func=fk,
        horizon=5,
        dt=0.05,
        q_min=np.full(7, -2.8),
        q_max=np.full(7, 2.8),
        qdot_max=np.full(7, 1.5),
    )

    qdot_cmd = mpc.solve(q_current, target)
    assert qdot_cmd.shape == (7,)
    assert np.all(np.abs(qdot_cmd) <= 1.5 + 1e-6)

    q_next = q_current + qdot_cmd * 0.05
    next_pos = panda_fk_numpy(q_next)

    dist_before = np.linalg.norm(current_pos - target)
    dist_after = np.linalg.norm(next_pos - target)
    assert dist_after < dist_before


def test_mpc_respects_joint_velocity_limits():
    fk = panda_fk_symbolic()
    q_current = np.zeros(7)
    target = np.array([0.5, 0.5, 0.5])  # a far, possibly unreachable target

    mpc = KinematicMPC(
        fk_func=fk,
        horizon=5,
        dt=0.05,
        q_min=np.full(7, -2.8),
        q_max=np.full(7, 2.8),
        qdot_max=np.full(7, 0.2),
    )

    qdot_cmd = mpc.solve(q_current, target)
    assert np.all(np.abs(qdot_cmd) <= 0.2 + 1e-6)


def test_solve_called_twice_reuses_warm_start_and_keeps_converging():
    """Task 12 added warm-start reuse (shift-by-one, last-knot hold,
    q_init[:, 0] overwrite) to solve() -- but both pre-existing tests above
    call .solve() only once per instance, so that whole code path
    (`self._prev_Q_sol is not None`) never ran under test. This calls
    .solve() twice in a row on the same instance (the second call is the
    only way to exercise the warm-start-reuse branch) and checks the second
    call doesn't crash or produce garbage, and that the arm keeps making
    real progress toward the target using the reused warm start.
    """
    fk = panda_fk_symbolic()
    q_current = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.7])
    current_pos = panda_fk_numpy(q_current)
    # A displacement too large to fully close in one 0.05s step at
    # qdot_max=1.5, so there is real remaining distance for the second call's
    # warm start to work with.
    target = current_pos + np.array([0.08, 0.05, -0.05])

    mpc = KinematicMPC(
        fk_func=fk,
        horizon=5,
        dt=0.05,
        q_min=np.full(7, -2.8),
        q_max=np.full(7, 2.8),
        qdot_max=np.full(7, 1.5),
    )

    qdot_1 = mpc.solve(q_current, target)
    q_after_1 = q_current + qdot_1 * 0.05
    dist_after_1 = np.linalg.norm(panda_fk_numpy(q_after_1) - target)

    # Second call: exercises the warm-start-reuse branch (self._prev_Q_sol is
    # no longer None here).
    qdot_2 = mpc.solve(q_after_1, target)
    assert qdot_2.shape == (7,)
    assert np.all(np.isfinite(qdot_2))
    assert np.all(np.abs(qdot_2) <= 1.5 + 1e-6)

    # The warm-start trajectory buffers must reflect the second solve, be the
    # right shape, finite, and satisfy the Q[:, 0] == q0_param equality
    # constraint exactly (a direct check that q_init[:, 0]'s overwrite -- and
    # the solve itself -- behaved sensibly, not just "didn't crash").
    assert mpc._prev_Q_sol.shape == (7, 6)
    assert mpc._prev_Qdot_sol.shape == (7, 5)
    assert np.all(np.isfinite(mpc._prev_Q_sol))
    np.testing.assert_allclose(mpc._prev_Q_sol[:, 0], q_after_1, atol=1e-6)

    q_after_2 = q_after_1 + qdot_2 * 0.05
    dist_after_2 = np.linalg.norm(panda_fk_numpy(q_after_2) - target)
    # Continues converging on the second, warm-started call rather than
    # stalling or diverging.
    assert dist_after_2 < dist_after_1


def test_posture_weight_biases_solution_toward_posture_target():
    """posture_target/posture_weight (Task 12) are supposed to pull the
    redundant nullspace toward a reference posture. Verify this actually
    happens: with a target displacement small enough to leave the 4-D
    self-motion nullspace free, a solve with posture_weight > 0 should move
    the joints measurably closer to posture_target than an otherwise
    identical solve with posture_weight = 0.
    """
    fk = panda_fk_symbolic()
    q_current = np.array([0.3, -0.2, 0.1, -1.6, 0.05, 1.4, 0.6])
    current_pos = panda_fk_numpy(q_current)
    # Tiny Cartesian displacement -- easily reachable, leaving the redundant
    # nullspace free for the posture cost to act in.
    target = current_pos + np.array([0.01, 0.0, 0.0])
    # Panda's own "home" configuration -- far from q_current in joint space.
    posture_target = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

    common = dict(
        fk_func=fk,
        horizon=5,
        dt=0.05,
        q_min=np.full(7, -2.8),
        q_max=np.full(7, 2.8),
        qdot_max=np.full(7, 1.5),
        posture_target=posture_target,
    )
    mpc_no_posture = KinematicMPC(**common, posture_weight=0.0)
    mpc_with_posture = KinematicMPC(**common, posture_weight=0.5)

    qdot_no = mpc_no_posture.solve(q_current, target)
    qdot_with = mpc_with_posture.solve(q_current, target)

    q_next_no = q_current + qdot_no * 0.05
    q_next_with = q_current + qdot_with * 0.05

    dist_no = np.linalg.norm(q_next_no - posture_target)
    dist_with = np.linalg.norm(q_next_with - posture_target)
    assert dist_with < dist_no - 1e-4


def test_terminal_weight_reduces_terminal_step_tracking_error():
    """terminal_weight (Task 12) adds an extra, heavily-weighted copy of the
    position cost on just the last horizon step. Verify it actually changes
    the solution: with a high effort_weight (discouraging fast motion) and a
    reachable-but-nontrivial target, a solve with terminal_weight=0 should
    leave more residual position error at the final horizon step than an
    otherwise identical solve with a high terminal_weight.
    """
    fk = panda_fk_symbolic()
    q_current = np.zeros(7)
    current_pos = panda_fk_numpy(q_current)
    target = current_pos + np.array([0.15, 0.1, -0.1])

    common = dict(
        fk_func=fk,
        horizon=5,
        dt=0.05,
        q_min=np.full(7, -2.8),
        q_max=np.full(7, 2.8),
        qdot_max=np.full(7, 1.0),
        effort_weight=1.0,  # heavy, so running cost alone trades off speed
    )
    mpc_no_terminal = KinematicMPC(**common, terminal_weight=0.0)
    mpc_with_terminal = KinematicMPC(**common, terminal_weight=50.0)

    mpc_no_terminal.solve(q_current, target)
    mpc_with_terminal.solve(q_current, target)

    q_final_no = mpc_no_terminal._prev_Q_sol[:, -1]
    q_final_with = mpc_with_terminal._prev_Q_sol[:, -1]

    err_no = np.linalg.norm(panda_fk_numpy(q_final_no) - target)
    err_with = np.linalg.norm(panda_fk_numpy(q_final_with) - target)

    assert err_with < err_no - 1e-4
