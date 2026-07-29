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
    ):
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

        opti.minimize(cost)
        # print_level=0 + sb="yes" suppresses IPOPT's banner/iteration log;
        # print_time=False suppresses CasADi's own solver-timing printout.
        opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "sb": "yes"})

        self._opti = opti
        self._Q = Q
        self._Qdot = Qdot
        self._q0_param = q0_param
        self._target_param = target_param

    def solve(self, q_current: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        self._opti.set_value(self._q0_param, q_current)
        self._opti.set_value(self._target_param, target_pos)
        self._opti.set_initial(self._Q, np.tile(q_current.reshape(-1, 1), (1, self.horizon + 1)))
        self._opti.set_initial(self._Qdot, np.zeros((self.n_joints, self.horizon)))
        sol = self._opti.solve()
        qdot_all = sol.value(self._Qdot)
        return np.asarray(qdot_all[:, 0]).flatten()
