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
