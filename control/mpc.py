"""Kinematic (joint-velocity) MPC tracking a moving Cartesian target."""
import casadi as ca
import numpy as np


class KinematicMPC:
    """Joint-velocity MPC that steers a Panda's end-effector toward a
    Cartesian target over a short receding horizon.

    The Opti NLP is built once in ``__init__`` and reused across ``.solve()``
    calls via CasADi parameters (``q0_param``, ``target_param``) -- this
    avoids re-deriving the NLP's symbolic structure and derivatives on every
    control step. Only the parameter values and the decision-variable warm
    start change between calls; the constraint/cost structure never does, so
    no state leaks between solves.
    """

    def __init__(
        self,
        fk_func: ca.Function,
        horizon: int,
        dt: float,
        q_min: np.ndarray,
        q_max: np.ndarray,
        qdot_max: np.ndarray,
        effort_weight: float = 0.01,
        posture_target: np.ndarray | None = None,
        posture_weight: float = 0.0,
        terminal_weight: float = 0.0,
        pose_fk_func: ca.Function | None = None,
        lateral_axis_weight: float = 0.0,
    ):
        """
        ``posture_target``/``posture_weight`` (Task 12 integration finding,
        additive and backward-compatible: both default to "off", so existing
        callers/tests are unaffected): a position-only Cartesian cost leaves
        a 4-dimensional self-motion nullspace completely unconstrained for
        this 7-DOF arm. In the conveyor integration this let the solver
        drift toward joint6 ~= 0 -- a documented Franka Panda wrist
        singularity (joints 5 and 7's axes become co-linear there) -- at
        which point tiny position corrections demanded huge joint5/7
        velocities (observed: joint5 jumping from 0.03 to 1.16 rad across
        three ticks) and the trajectory diverged. A small quadratic bias
        toward a reference posture (e.g. the arm's home configuration, safely
        away from that singularity) resolves the nullspace toward a known-
        good region instead of leaving it to numerical drift, at the cost of
        a small, tunable pull away from the pure minimum-position-error
        solution.

        ``terminal_weight`` (Task 12 integration finding, additive and
        backward-compatible: 0.0 keeps the original behavior): the running
        cost sums a *soft* position penalty over every horizon step, which
        for a target moving at constant velocity settles into a nonzero
        steady-state tracking gap (a classic type-1-system offset against a
        ramp input) -- observed in the conveyor integration as the
        end-effector plateauing 7-15 cm from a live, continuously-updated
        target no matter how the effort weight was tuned. Adding an extra,
        much more heavily weighted copy of the position cost on just the
        *last* horizon step pushes the solver to actually close the gap by
        the end of the horizon (subject to the same qdot_max/joint-limit
        constraints), which is what removed the steady-state offset in
        practice.

        ``pose_fk_func``/``lateral_axis_weight`` (found via a user-reported
        visual grasp failure, additive and backward-compatible: `None`/0.0
        keeps the original behavior): a *position-only* Cartesian cost gets
        the TCP close to the target in aggregate 3D distance, but does
        nothing to ensure the object is *centered along the gripper's
        finger-closing axis* -- the fingers only move along one local axis
        (empirically, local Y; see `panda_tcp_pose_symbolic`), and closing
        them cannot correct any offset along the perpendicular local X axis.
        Direct instrumentation of a full episode found exactly this: the
        object sat ~3cm offset along local X even as the fingers closed
        fully, so nothing was actually captured between them. `pose_fk_func`
        (e.g. `panda_tcp_pose_symbolic()`) supplies both TCP position and
        the gripper's orientation; when `lateral_axis_weight > 0`, an extra
        cost term penalizes the target's offset along the gripper's local X
        axis specifically, biasing the solver toward a wrist orientation
        that actually centers the object between the fingers, not just
        near the TCP in aggregate distance.
        """
        self.horizon = horizon
        self.dt = dt
        self.n_joints = q_min.shape[0]

        opti = ca.Opti()
        Q = opti.variable(self.n_joints, horizon + 1)  # joint trajectory
        Qdot = opti.variable(self.n_joints, horizon)  # joint velocity commands
        q0_param = opti.parameter(self.n_joints)
        target_param = opti.parameter(3)

        opti.subject_to(Q[:, 0] == q0_param)
        cost = 0
        for k in range(horizon):
            opti.subject_to(Q[:, k + 1] == Q[:, k] + Qdot[:, k] * dt)
            opti.subject_to(opti.bounded(q_min, Q[:, k + 1], q_max))
            opti.subject_to(opti.bounded(-qdot_max, Qdot[:, k], qdot_max))
            ee_pos = fk_func(Q[:, k + 1])
            cost += ca.sumsqr(ee_pos - target_param) + effort_weight * ca.sumsqr(Qdot[:, k])
            if posture_target is not None and posture_weight > 0.0:
                cost += posture_weight * ca.sumsqr(Q[:, k + 1] - posture_target)
            if k == horizon - 1 and terminal_weight > 0.0:
                cost += terminal_weight * ca.sumsqr(ee_pos - target_param)
            if pose_fk_func is not None and lateral_axis_weight > 0.0:
                pose_pos, pose_rot = pose_fk_func(Q[:, k + 1])
                lateral_axis = pose_rot[:, 0]
                lateral_offset = ca.dot(lateral_axis, pose_pos - target_param)
                cost += lateral_axis_weight * lateral_offset**2

        opti.minimize(cost)
        # print_level=0 + sb="yes" suppresses IPOPT's banner/iteration log;
        # print_time=False suppresses CasADi's own solver-timing printout.
        opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "sb": "yes"})

        self._opti = opti
        self._Q = Q
        self._Qdot = Qdot
        self._q0_param = q0_param
        self._target_param = target_param
        # Warm-start memory across .solve() calls -- see solve()'s docstring
        # for why this matters for a receding-horizon controller called
        # repeatedly against a moving target.
        self._prev_Q_sol: np.ndarray | None = None
        self._prev_Qdot_sol: np.ndarray | None = None

    def solve(self, q_current: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        """Solve one receding-horizon step and return the first joint-velocity
        command.

        Task 12 integration finding: the first version of this method reset
        the initial guess to ``q_current`` tiled across the whole horizon
        with zero velocity on *every* call, discarding the previous solve's
        trajectory. For a 7-DOF arm tracking a 3D position target there is a
        4-dimensional self-motion nullspace with no cost gradient in it, so
        IPOPT (a local NLP solver) has no reason to stay in the same
        solution basin between consecutive calls -- in the conveyor
        integration test this caused the arm to visibly oscillate between
        two different wrist configurations tick to tick (confirmed by
        logging the eye-in-hand camera's boresight direction, which
        flip-flopped between roughly +0.5 and -0.8 aligned with the target
        every other control tick), stalling convergence and spinning the
        wrist-mounted camera away from the target it needed to keep
        tracking. Standard receding-horizon MPC practice is to warm-start
        each solve from the previous solution's trajectory shifted by one
        step (holding the last knot point) rather than from scratch; doing
        so keeps consecutive solves in the same local-minimum basin and
        removed the oscillation in practice. The first call (no previous
        solution yet) falls back to the original tile/zero scheme.
        """
        self._opti.set_value(self._q0_param, q_current)
        self._opti.set_value(self._target_param, target_pos)

        if self._prev_Q_sol is None:
            q_init = np.tile(q_current.reshape(-1, 1), (1, self.horizon + 1))
            qdot_init = np.zeros((self.n_joints, self.horizon))
        else:
            q_init = np.empty_like(self._prev_Q_sol)
            q_init[:, :-1] = self._prev_Q_sol[:, 1:]
            q_init[:, -1] = self._prev_Q_sol[:, -1]
            q_init[:, 0] = q_current
            qdot_init = np.empty_like(self._prev_Qdot_sol)
            qdot_init[:, :-1] = self._prev_Qdot_sol[:, 1:]
            qdot_init[:, -1] = self._prev_Qdot_sol[:, -1]

        self._opti.set_initial(self._Q, q_init)
        self._opti.set_initial(self._Qdot, qdot_init)
        sol = self._opti.solve()
        self._prev_Q_sol = np.asarray(sol.value(self._Q)).reshape(self.n_joints, self.horizon + 1)
        self._prev_Qdot_sol = np.asarray(sol.value(self._Qdot)).reshape(self.n_joints, self.horizon)
        qdot_all = self._prev_Qdot_sol
        return np.asarray(qdot_all[:, 0]).flatten()
