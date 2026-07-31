"""Franka Panda forward kinematics (modified/Craig DH convention).

DH parameters below are the widely-published Franka Panda modified-DH
link parameters (7 revolute joints + fixed flange offset).

Two implementations are provided:
  - `panda_fk_symbolic`: builds a CasADi `Function` for use in symbolic
    optimization (e.g. the kinematic MPC in Task 10).
  - `panda_fk_numpy`: a plain-numpy reference implementation, used to
    cross-check the CasADi version in tests.

They are written as two separate, self-contained functions (each with its
own private per-link transform helper) so the transform-matrix construction
logic is never shared -- but both read from the same `_A`/`_ALPHA`/`_D`
parameter table below, so agreement between them cannot catch a wrong
literal value in that shared table (only a bug in the per-implementation
transform math). That exact bug class was found once already (see
`task-12-report.md`) and slipped past both implementations' agreement at
`q=0`, where the error happens to cancel out. The actual independent
cross-check against ground truth is
`tests/control/test_panda_kinematics.py::test_fk_numpy_matches_mujoco_conveyor_scene_hand_body`,
which compares `panda_fk_numpy` against the real, independently-authored
MuJoCo model's compiled body position.
"""
import casadi as ca
import numpy as np

# a_{i-1} (m), alpha_{i-1} (rad), d_i (m) for i = 1..7, then the fixed flange offset.
#
# Task 12 integration finding: this array was originally
# [0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088, 0.0] -- shifted one index early
# relative to the alpha/d/theta it's paired with in the loop below (each
# entry held a_{i} instead of a_{i-1}). This coincidentally reproduces the
# correct end-effector position at q = 0 (where sin(theta) terms vanish and
# most a-dependent offsets don't propagate), which is why
# test_casadi_fk_matches_independent_numpy_fk_at_zero_config passed despite
# the bug -- but it diverges sharply at nonzero joint angles: cross-checked
# against the actual Menagerie Panda MJCF's compiled "hand" body position
# (ground truth from mujoco.mj_forward) in
# tests/control/test_panda_kinematics.py, the old array gave a 0.245 m
# position error at the conveyor scene's home keyframe pose. The corrected
# array below reproduces the MuJoCo "hand" body xpos exactly (<1e-9 m) at
# q=0, at the home keyframe, and across 5 random configurations.
_A = [0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088]
_ALPHA = [0.0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2]
_D = [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.0]
_FLANGE_A, _FLANGE_ALPHA, _FLANGE_D = 0.0, 0.0, 0.107
# Fixed offset from the flange ("hand" body) origin to the midpoint between
# the two fingertip pads, along the flange's local Z axis. Both fingers open
#/close symmetrically about this axis, so the midpoint is invariant to the
# gripper's opening width -- this offset does not depend on finger joint
# state, only on the flange's pose. Measured directly against the compiled
# MuJoCo model (average of both fingertip pad geoms' world positions,
# expressed in the flange's local frame) and confirmed identical to
# floating-point precision (<1e-16 m) across two very different arm
# configurations, so it is safe to treat as a fixed constant here.
_TCP_OFFSET_Z = 0.1029
# Fixed rotation about the flange's local Z axis needed to make this DH
# chain's rotation MATCH MuJoCo's "hand" body orientation convention exactly
# (a well-known Franka Panda quirk: the DH-derived flange/K frame and the
# "hand" frame used by the gripper differ by a fixed 45-degree twist about
# the approach axis). Verified empirically: R_dh.T @ hand_xmat equals
# exactly this rotation (to <1e-7) at two very different arm configurations,
# so it is safe to treat as a fixed constant. Needed for any orientation-
# aware use of this FK (e.g. `panda_tcp_pose_symbolic` below) -- position
# alone (`panda_fk_*`/`panda_tcp_*` above) does not need this correction,
# since it doesn't depend on the frame's rotational convention at all.
_HAND_FRAME_Z_ROTATION = -np.pi / 4


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
    """Return a CasADi Function mapping q (7,) to flange position (3,)."""
    q = ca.SX.sym("q", 7)
    T = ca.SX.eye(4)
    for i in range(7):
        T = T @ _dh_transform_ca(_A[i], _ALPHA[i], _D[i], q[i])
    T = T @ _dh_transform_ca(_FLANGE_A, _FLANGE_ALPHA, _FLANGE_D, 0.0)
    ee_pos = T[0:3, 3]
    return ca.Function("panda_fk", [q], [ee_pos])


def panda_tcp_symbolic() -> ca.Function:
    """Return a CasADi Function mapping q (7,) to the fingertip-pad midpoint
    (tool-center-point) position (3,) -- the flange position plus the fixed
    `_TCP_OFFSET_Z` translation along the flange's local Z axis."""
    q = ca.SX.sym("q", 7)
    T = ca.SX.eye(4)
    for i in range(7):
        T = T @ _dh_transform_ca(_A[i], _ALPHA[i], _D[i], q[i])
    T = T @ _dh_transform_ca(_FLANGE_A, _FLANGE_ALPHA, _FLANGE_D, 0.0)
    T = T @ _dh_transform_ca(0.0, 0.0, _TCP_OFFSET_Z, 0.0)
    tcp_pos = T[0:3, 3]
    return ca.Function("panda_tcp", [q], [tcp_pos])


def panda_tcp_pose_symbolic() -> ca.Function:
    """Return a CasADi Function mapping q (7,) to (tcp_pos(3,), rotation(3,3)),
    where `rotation`'s columns are the gripper's local X/Y/Z axes expressed
    in world coordinates, in MuJoCo's "hand"-frame convention (see
    `_HAND_FRAME_Z_ROTATION`). Empirically, the gripper's fingers close
    along the local **Y** axis (rotation[:, 1]) and open/close motion does
    not move the object's offset along the local **X** axis (rotation[:, 0])
    at all -- so an object offset along local X is one the gripper can never
    correct by closing, regardless of how close the TCP position is
    overall. `control/mpc.py` uses `rotation[:, 0]` to penalize exactly that
    offset.
    """
    q = ca.SX.sym("q", 7)
    T = ca.SX.eye(4)
    for i in range(7):
        T = T @ _dh_transform_ca(_A[i], _ALPHA[i], _D[i], q[i])
    T = T @ _dh_transform_ca(_FLANGE_A, _FLANGE_ALPHA, _FLANGE_D, 0.0)
    T = T @ _dh_transform_ca(0.0, 0.0, 0.0, _HAND_FRAME_Z_ROTATION)
    rotation = T[0:3, 0:3]
    T = T @ _dh_transform_ca(0.0, 0.0, _TCP_OFFSET_Z, 0.0)
    tcp_pos = T[0:3, 3]
    return ca.Function("panda_tcp_pose", [q], [tcp_pos, rotation])


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
    """Pure-numpy reference FK: q (7,) -> flange position (3,)."""
    T = np.eye(4)
    for i in range(7):
        T = T @ _dh_transform_np(_A[i], _ALPHA[i], _D[i], q[i])
    T = T @ _dh_transform_np(_FLANGE_A, _FLANGE_ALPHA, _FLANGE_D, 0.0)
    return T[0:3, 3]


def panda_tcp_numpy(q: np.ndarray) -> np.ndarray:
    """Pure-numpy reference: q (7,) -> fingertip-pad midpoint (TCP) position
    (3,) -- see `panda_tcp_symbolic` for the offset this adds."""
    T = np.eye(4)
    for i in range(7):
        T = T @ _dh_transform_np(_A[i], _ALPHA[i], _D[i], q[i])
    T = T @ _dh_transform_np(_FLANGE_A, _FLANGE_ALPHA, _FLANGE_D, 0.0)
    T = T @ _dh_transform_np(0.0, 0.0, _TCP_OFFSET_Z, 0.0)
    return T[0:3, 3]


def panda_tcp_pose_numpy(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pure-numpy reference: q (7,) -> (tcp_pos(3,), rotation(3,3)), matching
    MuJoCo's "hand"-frame rotation convention -- see `panda_tcp_pose_symbolic`."""
    T = np.eye(4)
    for i in range(7):
        T = T @ _dh_transform_np(_A[i], _ALPHA[i], _D[i], q[i])
    T = T @ _dh_transform_np(_FLANGE_A, _FLANGE_ALPHA, _FLANGE_D, 0.0)
    T = T @ _dh_transform_np(0.0, 0.0, 0.0, _HAND_FRAME_Z_ROTATION)
    rotation = T[0:3, 0:3].copy()
    T = T @ _dh_transform_np(0.0, 0.0, _TCP_OFFSET_Z, 0.0)
    return T[0:3, 3], rotation
