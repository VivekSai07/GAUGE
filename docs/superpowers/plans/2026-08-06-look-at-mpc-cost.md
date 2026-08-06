# Look-At MPC Cost Term Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the wrist camera from losing the tracked object during GOTO by adding an optional MPC cost term that keeps the camera's boresight pointed at the object's live position while the arm drives to the rendezvous point.

**Architecture:** A new `camera_pose_symbolic()`/`camera_pose_numpy()` pair in `control/panda_kinematics.py` computes the wrist camera's world position and boresight direction from the same DH chain the TCP functions already use. `control/mpc.py::KinematicMPC` gains two new optional constructor parameters (`camera_fk_func`, `look_at_weight`) and a new `look_at_target` parameter on `.solve()`, following the exact additive/backward-compatible pattern `lateral_axis_weight` already established (default off, zero behavior change for existing callers). `run_conveyor_demo.py` passes the live KF object estimate as `look_at_target` only during the GOTO phase.

**Tech Stack:** Python, CasADi (Opti NLP), NumPy, MuJoCo, pytest.

## Global Constraints

- Every new constructor parameter must default to `None`/`0.0` and leave existing callers' behavior byte-identical when omitted — same pattern as `lateral_axis_weight` (spec Section 3).
- Do not modify `tests/test_integration_conveyor.py` (project-wide constraint, unrelated to this feature).
- Do not touch WAIT/CLOSE phase logic in `run_conveyor_demo.py` — `look_at_weight` is wired into GOTO only (spec Section 4).
- `uv run pytest -v` must stay at 57+/57 (57 today, plus whatever new tests this plan adds) after every task.
- The camera mount offset used (`pos="0 0 0.05"`, `euler="{pi} 0 0"` relative to the hand frame) must match `sim/conveyor_scene.py`'s real MJCF camera element exactly — this is a cross-check invariant, not a free parameter.

---

## Task 1: Camera pose forward kinematics

**Files:**
- Modify: `control/panda_kinematics.py`
- Test: `tests/control/test_panda_kinematics.py`

**Interfaces:**
- Produces: `camera_pose_symbolic() -> ca.Function` mapping `q (7,)` to `(camera_pos (3,), camera_forward (3,))`, where `camera_forward` is a unit vector — the camera's boresight direction (MuJoCo convention: local −Z axis) expressed in world coordinates.
- Produces: `camera_pose_numpy(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]`, the plain-numpy reference implementation with the identical contract, following the existing `panda_tcp_pose_symbolic`/`panda_tcp_pose_numpy` pairing pattern.

The wrist camera's boresight direction, in world coordinates, works out to exactly the hand frame's own local +Z axis — i.e. `rotation[:, 2]` from the existing `panda_tcp_pose_symbolic`/`_numpy` rotation output — because the camera's `euler="{pi} 0 0"` (180° about local X) mount rotation flips the sign twice: once for the camera's local −Z-is-forward convention, once for the 180° flip itself. This is derived, not assumed, and Step 5 below cross-checks it against the real compiled MuJoCo camera to confirm.

- [ ] **Step 1: Write the failing tests**

Append to `tests/control/test_panda_kinematics.py` (add these imports to the existing `from control.panda_kinematics import (...)` block: `camera_pose_numpy`, `camera_pose_symbolic`):

```python
def test_casadi_camera_pose_matches_independent_numpy_camera_pose():
    pose_fn = camera_pose_symbolic()
    rng = np.random.default_rng(23)
    for q in [np.zeros(7)] + [rng.uniform(-1.5, 1.5, size=7) for _ in range(5)]:
        casadi_pos, casadi_fwd = pose_fn(q)
        numpy_pos, numpy_fwd = camera_pose_numpy(q)
        np.testing.assert_allclose(
            np.array(casadi_pos).flatten(), numpy_pos, atol=1e-6
        )
        np.testing.assert_allclose(
            np.array(casadi_fwd).flatten(), numpy_fwd, atol=1e-6
        )


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. uv run pytest tests/control/test_panda_kinematics.py -v -k camera`
Expected: FAIL (or ERROR) with `ImportError: cannot import name 'camera_pose_symbolic'` (and `camera_pose_numpy`) — they don't exist yet.

- [ ] **Step 3: Add the camera mount constant and both implementations**

In `control/panda_kinematics.py`, add this constant near `_TCP_OFFSET_Z`/`_HAND_FRAME_Z_ROTATION` (after `_HAND_FRAME_Z_ROTATION = -np.pi / 4`):

```python
# Fixed offset from the hand-frame origin to the wrist camera's mount
# point, along the hand frame's local Z axis -- matches
# sim/conveyor_scene.py's real MJCF camera element (pos="0 0 0.05"). The
# camera's own euler="{pi} 0 0" mount rotation (180 degrees about local X)
# is not applied separately here: composing it with a MuJoCo camera's
# local -Z-is-forward convention algebraically cancels back to the hand
# frame's own +Z axis, so `camera_forward` below is simply
# `rotation[:, 2]` -- verified against the real compiled MuJoCo camera in
# tests/control/test_panda_kinematics.py::test_camera_pose_numpy_matches_mujoco_wrist_cam.
_CAMERA_OFFSET_Z = 0.05
```

Add `camera_pose_symbolic` after `panda_tcp_pose_symbolic`:

```python
def camera_pose_symbolic() -> ca.Function:
    """Return a CasADi Function mapping q (7,) to (camera_pos(3,),
    camera_forward(3,)) -- the wrist camera's world position and its
    boresight (viewing) direction, a unit vector. See `_CAMERA_OFFSET_Z`
    for why `camera_forward` reduces to the hand frame's local Z axis.
    """
    q = ca.SX.sym("q", 7)
    T = ca.SX.eye(4)
    for i in range(7):
        T = T @ _dh_transform_ca(_A[i], _ALPHA[i], _D[i], q[i])
    T = T @ _dh_transform_ca(_FLANGE_A, _FLANGE_ALPHA, _FLANGE_D, 0.0)
    T = T @ _dh_transform_ca(0.0, 0.0, 0.0, _HAND_FRAME_Z_ROTATION)
    rotation = T[0:3, 0:3]
    hand_pos = T[0:3, 3]
    camera_forward = rotation[:, 2]
    camera_pos = hand_pos + rotation[:, 2] * _CAMERA_OFFSET_Z
    return ca.Function("camera_pose", [q], [camera_pos, camera_forward])
```

Add `camera_pose_numpy` after `panda_tcp_pose_numpy`:

```python
def camera_pose_numpy(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pure-numpy reference: q (7,) -> (camera_pos(3,), camera_forward(3,)),
    matching `camera_pose_symbolic`."""
    T = np.eye(4)
    for i in range(7):
        T = T @ _dh_transform_np(_A[i], _ALPHA[i], _D[i], q[i])
    T = T @ _dh_transform_np(_FLANGE_A, _FLANGE_ALPHA, _FLANGE_D, 0.0)
    T = T @ _dh_transform_np(0.0, 0.0, 0.0, _HAND_FRAME_Z_ROTATION)
    rotation = T[0:3, 0:3].copy()
    hand_pos = T[0:3, 3].copy()
    camera_forward = rotation[:, 2].copy()
    camera_pos = hand_pos + rotation[:, 2] * _CAMERA_OFFSET_Z
    return camera_pos, camera_forward
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. uv run pytest tests/control/test_panda_kinematics.py -v`
Expected: all PASS, including the three new tests and every pre-existing test in this file (unaffected).

- [ ] **Step 5: Run the full suite**

Run: `PYTHONPATH=. uv run pytest -v`
Expected: 60/60 passed (57 existing + 3 new).

- [ ] **Step 6: Commit**

```bash
git add control/panda_kinematics.py tests/control/test_panda_kinematics.py
git commit -m "feat: add camera pose FK for the look-at MPC cost term (#36)"
```

---

## Task 2: Look-at cost term in KinematicMPC

**Files:**
- Modify: `control/mpc.py`
- Test: `tests/control/test_mpc.py`

**Interfaces:**
- Consumes: `camera_pose_symbolic()` from Task 1 (`(camera_pos, camera_forward) = camera_fk_func(Q[:, k+1])`).
- Produces: `KinematicMPC(..., camera_fk_func: ca.Function | None = None, look_at_weight: float = 0.0)` — new constructor parameters, both defaulting to off. `KinematicMPC.solve(q_current, target_pos, look_at_target: np.ndarray | None = None)` — new optional third parameter on the existing method. When the instance was constructed with `look_at_weight > 0.0` and a call omits `look_at_target` (`None`), `solve()` falls back to using `target_pos` as the look-at target too, so the method never raises for a missing look-at value.

- [ ] **Step 1: Read the existing test file to match its style**

Run: `PYTHONPATH=. uv run pytest tests/control/test_mpc.py -v --collect-only` to see current test names before adding to the same file (no code changes this step).

- [ ] **Step 2: Write the failing tests**

Append to `tests/control/test_mpc.py`. The existing `from control.panda_kinematics import (...)` block there currently imports `panda_fk_numpy, panda_fk_symbolic, panda_tcp_pose_numpy, panda_tcp_pose_symbolic` — extend it to also import `camera_pose_numpy, camera_pose_symbolic, panda_tcp_symbolic` (all three are needed by the new tests below):

```python
def test_look_at_weight_off_by_default_is_byte_identical():
    """camera_fk_func/look_at_weight default to None/0.0 -- omitting them
    must reproduce the exact qdot a pre-existing caller would get."""
    fk = panda_tcp_symbolic()
    q_min = np.full(7, -2.8)
    q_max = np.full(7, 2.8)
    qdot_max = np.full(7, 1.5)
    q0 = np.zeros(7)
    target = np.array([0.4, 0.1, 0.4])

    mpc_without_param = KinematicMPC(
        fk_func=fk, horizon=5, dt=0.05, q_min=q_min, q_max=q_max, qdot_max=qdot_max
    )
    mpc_with_default_param = KinematicMPC(
        fk_func=fk,
        horizon=5,
        dt=0.05,
        q_min=q_min,
        q_max=q_max,
        qdot_max=qdot_max,
        camera_fk_func=None,
        look_at_weight=0.0,
    )
    qdot_a = mpc_without_param.solve(q0, target)
    qdot_b = mpc_with_default_param.solve(q0, target)
    np.testing.assert_allclose(qdot_a, qdot_b, atol=1e-9)


def test_look_at_weight_biases_solution_toward_facing_the_target():
    """With look_at_weight active, comparing the camera's boresight
    alignment to look_at_target across two solves -- one with the term on,
    one off, both driving to the same Cartesian target -- the aligned
    solve's final angular deviation must be smaller. Uses a target that is
    reachable by more than one wrist orientation (a real nullspace),
    so the two solves are not forced to agree."""
    fk = panda_tcp_symbolic()
    camera_fk = camera_pose_symbolic()
    q_min = np.full(7, -2.8)
    q_max = np.full(7, 2.8)
    qdot_max = np.full(7, 1.5)
    q0 = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.8, 0.0])
    target = np.array([0.45, 0.15, 0.35])
    look_at_target = np.array([0.45, 0.15, 0.05])

    mpc_off = KinematicMPC(
        fk_func=fk, horizon=5, dt=0.05, q_min=q_min, q_max=q_max, qdot_max=qdot_max
    )
    mpc_on = KinematicMPC(
        fk_func=fk,
        horizon=5,
        dt=0.05,
        q_min=q_min,
        q_max=q_max,
        qdot_max=qdot_max,
        camera_fk_func=camera_fk,
        look_at_weight=50.0,
    )

    def angular_deviation(qdot_cmd):
        q_next = q0 + qdot_cmd * 0.05
        cam_pos, cam_fwd = camera_pose_numpy(q_next)
        direction = look_at_target - cam_pos
        direction /= np.linalg.norm(direction)
        cos_angle = np.dot(cam_fwd, direction)
        return 1.0 - cos_angle

    qdot_off = mpc_off.solve(q0, target)
    qdot_on = mpc_on.solve(q0, target, look_at_target=look_at_target)

    assert angular_deviation(qdot_on) < angular_deviation(qdot_off)


def test_solve_without_look_at_target_falls_back_to_position_target():
    """Constructed with look_at_weight > 0 but called without
    look_at_target -- must not raise, and must fall back to using
    target_pos as the look-at target (verified by equivalence with an
    explicit call passing target_pos as look_at_target)."""
    fk = panda_tcp_symbolic()
    camera_fk = camera_pose_symbolic()
    q_min = np.full(7, -2.8)
    q_max = np.full(7, 2.8)
    qdot_max = np.full(7, 1.5)
    q0 = np.zeros(7)
    target = np.array([0.4, 0.1, 0.4])

    mpc = KinematicMPC(
        fk_func=fk,
        horizon=5,
        dt=0.05,
        q_min=q_min,
        q_max=q_max,
        qdot_max=qdot_max,
        camera_fk_func=camera_fk,
        look_at_weight=10.0,
    )
    qdot_implicit = mpc.solve(q0, target)
    qdot_explicit = mpc.solve(q0, target, look_at_target=target)
    np.testing.assert_allclose(qdot_implicit, qdot_explicit, atol=1e-9)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `PYTHONPATH=. uv run pytest tests/control/test_mpc.py -v -k look_at`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'camera_fk_func'`.

- [ ] **Step 4: Implement the cost term**

In `control/mpc.py`, update the `__init__` signature (add after the existing `lateral_axis_weight: float = 0.0,` parameter):

```python
        camera_fk_func: ca.Function | None = None,
        look_at_weight: float = 0.0,
```

Add this paragraph to the docstring, after the `lateral_axis_weight` paragraph (before the closing `"""`):

```
        ``camera_fk_func``/``look_at_weight`` (#36 -- additive and
        backward-compatible: `None`/0.0 keeps the original behavior):
        measured directly (see the Round 7 look-at MPC cost design spec),
        the wrist camera loses the tracked object the instant GOTO starts
        moving the arm, well before the object is actually out of range --
        because the position-only Cartesian cost leaves the wrist's
        orientation nullspace free to rotate the camera away from the
        object with no penalty for doing so. `camera_fk_func` (e.g.
        `camera_pose_symbolic()`) supplies the camera's world position and
        boresight direction; when `look_at_weight > 0.0`, an extra cost
        term penalizes `1 - cos(angle)` between the boresight and the
        direction to `look_at_target` (a new `.solve()` parameter,
        typically the object's live tracked position) at every horizon
        step, biasing the solver toward wrist orientations that keep the
        object in view while still reaching the position target.
```

Inside `__init__`, add a new CasADi parameter alongside `target_param` (after `target_param = opti.parameter(3)`):

```python
        look_at_target_param = opti.parameter(3)
```

Inside the `for k in range(horizon):` loop, after the existing `lateral_axis_weight` block:

```python
            if camera_fk_func is not None and look_at_weight > 0.0:
                cam_pos, cam_forward = camera_fk_func(Q[:, k + 1])
                direction = look_at_target_param - cam_pos
                # Small epsilon keeps the gradient finite if direction's
                # norm is ever near zero (not expected in practice, since
                # the tracked object is never at the camera's own
                # position, but cheap to guard against).
                direction_norm = ca.sqrt(ca.sumsqr(direction) + 1e-9)
                cos_angle = ca.dot(cam_forward, direction) / direction_norm
                cost += look_at_weight * (1.0 - cos_angle)
```

Store the new parameter alongside the existing ones (after `self._target_param = target_param`):

```python
        self._look_at_target_param = look_at_target_param
        self._look_at_active = camera_fk_func is not None and look_at_weight > 0.0
```

- [ ] **Step 5: Update `.solve()` to accept and set the new parameter**

Change the `solve` method signature:

```python
    def solve(
        self,
        q_current: np.ndarray,
        target_pos: np.ndarray,
        look_at_target: np.ndarray | None = None,
    ) -> np.ndarray:
```

Add, right after the existing `self._opti.set_value(self._target_param, target_pos)` line:

```python
        if self._look_at_active:
            self._opti.set_value(
                self._look_at_target_param,
                target_pos if look_at_target is None else look_at_target,
            )
```

Note: `opti.parameter(3)` is always part of the Opti problem structure regardless of whether the look-at term is active (CasADi has no conditional-parameter construct), but `look_at_target_param` is only ever read by the cost expression when `camera_fk_func is not None and look_at_weight > 0.0` — so leaving it at CasADi's default (`0`) when inactive is harmless: it's built into the graph but multiplied into nothing, since the `if camera_fk_func is not None and look_at_weight > 0.0:` block in `__init__` never adds the cost term reading it at all when the feature is off. The `set_value` call above is guarded by `self._look_at_active` purely to avoid the wasted call, not for correctness.

- [ ] **Step 6: Run tests to verify they pass**

Run: `PYTHONPATH=. uv run pytest tests/control/test_mpc.py -v`
Expected: all PASS, including the three new tests and every pre-existing test in this file.

- [ ] **Step 7: Run the full suite**

Run: `PYTHONPATH=. uv run pytest -v`
Expected: 63/63 passed (60 from Task 1 + 3 new).

- [ ] **Step 8: Commit**

```bash
git add control/mpc.py tests/control/test_mpc.py
git commit -m "feat: add optional look-at cost term to KinematicMPC (#36)"
```

---

## Task 3: Wire look-at into GOTO, measure, and tune

**Files:**
- Modify: `run_conveyor_demo.py`
- Modify: `configs/conveyor.yaml`
- Test: existing `tests/test_integration_conveyor.py` (read-only — do not modify, per Global Constraints; this task only needs it to keep passing)

**Interfaces:**
- Consumes: `camera_pose_symbolic()` (Task 1), `KinematicMPC(camera_fk_func=..., look_at_weight=...)` and `.solve(..., look_at_target=...)` (Task 2).
- Produces: no new public interface — this task is integration + tuning, not new API surface.

- [ ] **Step 1: Wire `camera_fk_func` into the MPC construction**

In `run_conveyor_demo.py`, add `camera_pose_symbolic` to the existing `from control.panda_kinematics import (...)` block. Change the `fk = panda_tcp_symbolic()` / `pose_fk = panda_tcp_pose_symbolic()` lines to also build:

```python
    camera_fk = camera_pose_symbolic()
```

Update the `KinematicMPC(...)` construction to add:

```python
        camera_fk_func=camera_fk,
        look_at_weight=mpc_cfg.get("look_at_weight", 0.0),
```

Add `look_at_weight: 0.0` to `configs/conveyor.yaml`'s `mpc:` section for now (Step 5 below replaces `0.0` with a tuned value once measured) — placed after the existing `lateral_axis_weight: 25.0` line, with a placeholder comment noting Step 5 will fill in the real value and rationale:

```yaml
  # look_at_weight: tuned in the Round 7 look-at-MPC-cost design spec's
  # verification sweep (see docs/superpowers/specs/2026-08-06-look-at-mpc-
  # cost-design.md, Section 5, item 5) -- value filled in below once swept.
  look_at_weight: 0.0
```

- [ ] **Step 2: Pass `look_at_target` during GOTO**

In the `if phase == "GOTO":` block, change:

```python
        qdot_cmd = mpc.solve(q_current, rendezvous)
```

to:

```python
        qdot_cmd = mpc.solve(q_current, rendezvous, look_at_target=obj_est)
```

(`obj_est` is already computed earlier in the loop body, above the `if phase == "TRACK":` branch — this is the same live KF position estimate the rendezvous-point calculation already uses.)

- [ ] **Step 3: Run the existing test suite to confirm nothing broke with `look_at_weight=0.0`**

Run: `PYTHONPATH=. uv run pytest -v`
Expected: 63/63 passed — `look_at_weight: 0.0` in the config must be behaviorally identical to Task 2's off-by-default guarantee, so this is a regression check, not a new feature check.

- [ ] **Step 4: Measure the GOTO-blindness trace with the term active, before choosing a weight**

Create a throwaway measurement script (not committed — this is a manual verification step, matching how the spec's own Section 2 table was produced) at `scratchpad/measure_goto_blindness.py`:

```python
"""Throwaway measurement script -- not part of the test suite. Traces
detection liveness through TRACK/GOTO with a given mpc.look_at_weight
override, to compare against the spec's Section 2 baseline table (which
was measured at look_at_weight=0.0, i.e. today's shipped behavior)."""

import sys

import numpy as np
import yaml

import run_conveyor_demo as rcd
from control.mpc import KinematicMPC
from control.panda_kinematics import (
    camera_pose_symbolic,
    panda_tcp_numpy,
    panda_tcp_pose_symbolic,
    panda_tcp_symbolic,
)
from perception.camera import CameraIntrinsics
from perception.yolo_segment import yolo_centroid
from sim.conveyor_scene import ConveyorSceneEnv
from tracking.kf import ConstantVelocityKF
from tracking.track import Track, TrackStatus
from ultralytics import YOLO

look_at_weight = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0

with open("configs/conveyor.yaml") as f:
    config = yaml.safe_load(f)
cam_cfg = config["camera"]
env = ConveyorSceneEnv(
    conveyor_velocity=np.array(config["conveyor_velocity"]), dt=config["dt"]
)
env.reset()
fx, fy, cx, cy = env.camera_intrinsics(cam_cfg["width"], cam_cfg["height"])
intrinsics = CameraIntrinsics(fx, fy, cx, cy)
cam_id = env.model.camera("wrist_cam").id
detector = YOLO(str(rcd.MODEL_PATH))
kf_cfg, track_cfg = config["kf"], config["track"]
track = None
q_min = env.model.jnt_range[:7, 0].copy()
q_max = env.model.jnt_range[:7, 1].copy()
q_home = env.get_joint_positions().copy()
fk = panda_tcp_symbolic()
pose_fk = panda_tcp_pose_symbolic()
camera_fk = camera_pose_symbolic()
mpc_cfg = config["mpc"]
mpc = KinematicMPC(
    fk_func=fk,
    horizon=mpc_cfg["horizon"],
    dt=1.0 / config["control_hz"],
    q_min=q_min,
    q_max=q_max,
    qdot_max=np.full(7, mpc_cfg["qdot_max"]),
    posture_target=q_home,
    posture_weight=mpc_cfg.get("posture_weight", 0.0),
    terminal_weight=mpc_cfg.get("terminal_weight", 0.0),
    pose_fk_func=pose_fk,
    lateral_axis_weight=mpc_cfg.get("lateral_axis_weight", 0.0),
    camera_fk_func=camera_fk,
    look_at_weight=look_at_weight,
)
sim_steps_per_control = max(1, round((1.0 / config["control_hz"]) / config["dt"]))
qdot_cmd = np.zeros(7)
phase = "TRACK"
rendezvous = None
goto_ticks = 0
tick_dt = 1.0 / config["control_hz"]
from run_conveyor_demo import (
    _GOTO_STALL_DIST_M,
    _GOTO_STALL_QDOT,
    _GOTO_TIMEOUT_S,
    _GRASP_Z,
    _RENDEZVOUS_TIME_BUDGET_S,
    _SETTLE_TOL_M,
)

last_live_step = None
first_blind_step = None
for step in range(3500):
    env.step(qdot_cmd)
    if step % sim_steps_per_control != 0:
        continue
    rgb, depth = env.get_rgbd(cam_cfg["width"], cam_cfg["height"])
    cam_pos = env.data.cam_xpos[cam_id]
    cam_mat = env.data.cam_xmat[cam_id].reshape(3, 3)
    measurement = yolo_centroid(
        rgb,
        depth,
        detector,
        intrinsics,
        tuple(cam_cfg["color_lower"]),
        tuple(cam_cfg["color_upper"]),
        cam_pos,
        cam_mat,
        depth_bias=rcd.OBJECT_HALF_HEIGHT_M,
    )
    if track is None and measurement is not None:
        kf = ConstantVelocityKF(
            dt=1.0 / config["control_hz"],
            process_var=kf_cfg["process_var"],
            meas_var=kf_cfg["meas_var"],
            init_state=np.array([*measurement, 0.0, 0.0, 0.0]),
            init_cov=np.eye(6) * kf_cfg["init_cov_scale"],
        )
        track = Track(
            kf=kf,
            gate_threshold=track_cfg["gate_threshold"],
            m=track_cfg["m"],
            n=track_cfg["n"],
            max_consecutive_misses=track_cfg["max_consecutive_misses"],
        )
        qdot_cmd = np.zeros(7)
        continue
    if track is None:
        qdot_cmd = np.zeros(7)
        continue
    status = track.step(measurement)
    if status == TrackStatus.LOST:
        track = None
        qdot_cmd = np.zeros(7)
        continue
    q_current = env.get_joint_positions()
    ee_pos = panda_tcp_numpy(q_current)
    obj_est = track.kf.x[:3].copy()
    obj_vel = track.kf.x[3:].copy()
    vel_h = obj_vel.copy()
    vel_h[2] = 0.0
    speed = float(np.linalg.norm(vel_h))
    if phase == "TRACK":
        if status == TrackStatus.CONFIRMED and speed > 1e-3:
            rendezvous = obj_est + (vel_h / speed) * (speed * _RENDEZVOUS_TIME_BUDGET_S)
            rendezvous[2] = _GRASP_Z
            phase, goto_ticks = "GOTO", 0
        qdot_cmd = np.zeros(7)
        continue
    if phase == "GOTO":
        qdot_cmd = mpc.solve(q_current, rendezvous, look_at_target=obj_est)
        goto_ticks += 1
        if measurement is not None:
            last_live_step = step
        elif first_blind_step is None:
            first_blind_step = step
        dist = float(np.linalg.norm(ee_pos - rendezvous))
        stalled = (
            float(np.linalg.norm(qdot_cmd)) < _GOTO_STALL_QDOT
            and dist < _GOTO_STALL_DIST_M
        )
        if dist < _SETTLE_TOL_M or stalled or goto_ticks * tick_dt > _GOTO_TIMEOUT_S:
            print(f"look_at_weight={look_at_weight}: ENTER WAIT at step {step}, "
                  f"first_blind_step={first_blind_step}, last_live_step={last_live_step}")
            break
        continue
```

Run it at a few candidate weights and compare `first_blind_step` against the spec's baseline of 2350:

```bash
PYTHONPATH=. uv run python scratchpad/measure_goto_blindness.py 0.0
PYTHONPATH=. uv run python scratchpad/measure_goto_blindness.py 10.0
PYTHONPATH=. uv run python scratchpad/measure_goto_blindness.py 50.0
PYTHONPATH=. uv run python scratchpad/measure_goto_blindness.py 200.0
```

Record each weight's `first_blind_step` (higher = better, i.e. later blindness) in a comment in your work (this becomes part of the commit message or a scratch note — not a permanent file).

- [ ] **Step 5: Run the full closed-loop episode and 6-speed sweep at each candidate weight**

For each weight tested in Step 4, run:

```bash
PYTHONPATH=. uv run python -c "
import yaml
import run_conveyor_demo as rcd
with open('configs/conveyor.yaml') as f: cfg=yaml.safe_load(f)
cfg['mpc']['look_at_weight'] = <WEIGHT>
for v in [0.04, 0.05, 0.06, 0.08, 0.10, 0.12]:
    c = dict(cfg); c['conveyor_velocity'] = [0.0, v]
    result = rcd.run_one_episode(c)
    print(v, '->', {k: (round(x,4) if isinstance(x,float) else x) for k,x in result.items()})
"
```

Pick the smallest weight that (a) meaningfully delays `first_blind_step` from Step 4, and (b) does not regress `contact_verified` at any of the 6 speeds, and (c) does not visibly worsen `grasp_error_m` relative to the `look_at_weight=0.0` baseline (0.0084–0.0119 across the 6 speeds, per the merged Round 7 result). If every nonzero weight tested regresses any speed's `contact_verified`, that is a real, reportable result — do not force a weight that breaks the suite; stop and report the finding instead of picking a broken value.

- [ ] **Step 6: Set the chosen weight in the config**

Replace the `look_at_weight: 0.0` placeholder in `configs/conveyor.yaml` (added in Step 1) with the chosen value from Step 5, and rewrite its comment to state the measured result (matching the style of the existing `position_tolerance` comment's sweep table above it in the same file): the weight chosen, the `first_blind_step` it achieved vs. the 2350 baseline, and the 6-speed `contact_verified`/`grasp_error_m` table.

- [ ] **Step 7: Run the full suite and the acceptance test one more time**

Run: `PYTHONPATH=. uv run pytest -v`
Expected: 63/63 passed, including `tests/test_integration_conveyor.py::test_conveyor_episode_grasps_within_tolerance` (unmodified, must still pass with the chosen weight now live in `configs/conveyor.yaml`).

- [ ] **Step 8: Remove the throwaway measurement script**

```bash
rm scratchpad/measure_goto_blindness.py
```

- [ ] **Step 9: Commit**

```bash
git add run_conveyor_demo.py configs/conveyor.yaml
git commit -m "feat: keep wrist camera on the object during GOTO via look-at cost (#36)"
```

---

## Final steps (not a task — do after Task 3)

- Update `README.md` and `docs/PROJECT_METRICS.md` with the measured before/after (`first_blind_step`, and whether/how much `grasp_error_m` or the WAIT-phase blind duration improved), matching every prior round's documentation practice.
- Update issue #36 with the real result: either it's substantially mitigated (state the numbers) or the sweep found no safe improvement (state that honestly too, per the design spec's stated "if it doesn't work, that's a real, reportable result" framing already established this session).
- Land via branch + PR (branch `feat/look-at-mpc-cost` already exists with the spec committed) — `main` is protected, matching every prior round.
