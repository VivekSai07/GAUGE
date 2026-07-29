"""Franka Panda forward kinematics (modified/Craig DH convention).

DH parameters below are the widely-published Franka Panda modified-DH
link parameters (7 revolute joints + fixed flange offset).

Two independent implementations are provided on purpose:
  - `panda_fk_symbolic`: builds a CasADi `Function` for use in symbolic
    optimization (e.g. the kinematic MPC in Task 10).
  - `panda_fk_numpy`: a plain-numpy reference implementation, used only to
    cross-check the CasADi version in tests.

They are written as two separate, self-contained functions (each with its
own private per-link transform helper) rather than sharing one
backend-parameterized helper, so that the two never call into each other and
an agreement between them is a genuine independent cross-check.
"""
import casadi as ca
import numpy as np

# a_{i-1} (m), alpha_{i-1} (rad), d_i (m) for i = 1..7, then the fixed flange offset.
_A = [0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088, 0.0]
_ALPHA = [0.0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2]
_D = [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.0]
_FLANGE_A, _FLANGE_ALPHA, _FLANGE_D = 0.0, 0.0, 0.107


def _dh_transform_ca(a: float, alpha: float, d: float, theta: ca.SX) -> ca.SX:
    """Modified-DH homogeneous transform, CasADi symbolic version."""
    ct, st = ca.cos(theta), ca.sin(theta)
    calpha, salpha = np.cos(alpha), np.sin(alpha)  # alpha is a fixed float
    return ca.vertcat(
        ca.horzcat(ct, -st, 0, a),
        ca.horzcat(st * calpha, ct * calpha, -salpha, -salpha * d),
        ca.horzcat(st * salpha, ct * salpha, calpha, calpha * d),
        ca.horzcat(0, 0, 0, 1),
    )


def panda_fk_symbolic() -> ca.Function:
    """Return a CasADi Function mapping q (7,) to end-effector position (3,)."""
    q = ca.SX.sym("q", 7)
    T = ca.SX.eye(4)
    for i in range(7):
        T = T @ _dh_transform_ca(_A[i], _ALPHA[i], _D[i], q[i])
    T = T @ _dh_transform_ca(_FLANGE_A, _FLANGE_ALPHA, _FLANGE_D, 0.0)
    ee_pos = T[0:3, 3]
    return ca.Function("panda_fk", [q], [ee_pos])


def _dh_transform_np(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Modified-DH homogeneous transform, pure-numpy version."""
    ct, st = np.cos(theta), np.sin(theta)
    calpha, salpha = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * calpha, ct * calpha, -salpha, -salpha * d],
            [st * salpha, ct * salpha, calpha, calpha * d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def panda_fk_numpy(q: np.ndarray) -> np.ndarray:
    """Pure-numpy reference FK: q (7,) -> end-effector position (3,)."""
    T = np.eye(4)
    for i in range(7):
        T = T @ _dh_transform_np(_A[i], _ALPHA[i], _D[i], q[i])
    T = T @ _dh_transform_np(_FLANGE_A, _FLANGE_ALPHA, _FLANGE_D, 0.0)
    return T[0:3, 3]
