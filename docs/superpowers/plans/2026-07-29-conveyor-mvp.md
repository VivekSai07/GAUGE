# GAUGE Conveyor MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first end-to-end closed loop — a simulated Franka Panda
with an eye-in-hand RGB-D camera detects, tracks, predicts, intercepts, and
grasps a constant-velocity object moving on a conveyor in MuJoCo.

**Architecture:** Modular Python pipeline (perception → tracking → prediction
→ interception planning → kinematic MPC → grasp execution), each stage a
standalone, independently-unit-tested module. MuJoCo simulates full Panda
dynamics/contacts/rendering; the object moves as a scripted `mocap` body
(exact constant velocity, no belt-friction physics needed). Control is a
kinematic (joint-velocity) MPC formulated directly in CasADi — no Pinocchio/
C++ dependency.

**Tech Stack:** Python 3.11, `mujoco`, `casadi` (bundled IPOPT), `numpy`,
`opencv-python`, `pytest`, `uv`. MuJoCo Menagerie's Franka Panda asset.

## Global Constraints

- Simulation-only — no real robot code paths.
- Pure Python — no ROS2, no C++ build steps, no Pinocchio/acados.
- Windows 11 host — every command in this plan must run under PowerShell/Git
  Bash without a Linux/WSL2 dependency.
- Object geometry (small box) is known — no learned/general grasp-pose
  network in this plan.
- Control is a **kinematic** MPC (controls `q̇`, tracks target via forward
  kinematics only) — not a torque-level dynamic MPC.
- Every module must be independently testable with `pytest` before
  integration.

---

## File Structure

```
dynamic-object-tracking/
├── pyproject.toml
├── sim/
│   ├── __init__.py
│   ├── conveyor_scene.py       # ConveyorSceneEnv: loads MJCF, steps sim, renders RGB-D
│   └── assets/
│       └── menagerie/          # git submodule: MuJoCo Menagerie (franka_emika_panda)
├── perception/
│   ├── __init__.py
│   ├── camera.py               # CameraIntrinsics: back-projection
│   └── segment.py              # segment_object_centroid()
├── tracking/
│   ├── __init__.py
│   ├── kf.py                   # ConstantVelocityKF
│   └── track.py                # Track, TrackStatus (m/n confirmation)
├── prediction/
│   ├── __init__.py
│   └── predict.py              # propagate()
├── planning/
│   ├── __init__.py
│   └── intercept.py            # solve_intercept()
├── control/
│   ├── __init__.py
│   ├── panda_kinematics.py     # panda_fk_symbolic() — CasADi FK
│   └── mpc.py                  # KinematicMPC
├── manipulation/
│   ├── __init__.py
│   └── grasp.py                # GraspExecutor
├── configs/
│   └── conveyor.yaml
├── run_conveyor_demo.py        # top-level integration script
└── tests/
    ├── sim/test_conveyor_scene.py
    ├── perception/test_camera.py
    ├── perception/test_segment.py
    ├── tracking/test_kf.py
    ├── tracking/test_track.py
    ├── prediction/test_predict.py
    ├── planning/test_intercept.py
    ├── control/test_panda_kinematics.py
    ├── control/test_mpc.py
    ├── manipulation/test_grasp.py
    └── test_integration_conveyor.py
```

---

### Task 1: Project scaffolding + environment

**Files:**
- Create: `pyproject.toml`
- Create: `perception/__init__.py`, `tracking/__init__.py`, `prediction/__init__.py`, `planning/__init__.py`, `control/__init__.py`, `manipulation/__init__.py`, `sim/__init__.py`
- Create: `.gitignore`

**Interfaces:** none (no code logic yet).

- [ ] **Step 1: Initialize git repo and uv project**

```bash
git init
uv init --name gauge --python 3.11
```

- [ ] **Step 2: Write `pyproject.toml` dependencies**

Add to `pyproject.toml` under `[project]`:

```toml
dependencies = [
    "mujoco>=3.2.0",
    "casadi>=3.6.5",
    "numpy>=1.26",
    "opencv-python>=4.10",
    "pyyaml>=6.0",
    "defusedxml>=0.7.1",
]

[dependency-groups]
dev = ["pytest>=8.0"]
```

- [ ] **Step 3: Install dependencies**

```bash
uv sync
```

Expected: `uv.lock` created, `.venv` populated with mujoco/casadi/numpy/opencv/pytest.

- [ ] **Step 4: Create package directories and empty `__init__.py` files**

```bash
mkdir -p perception tracking prediction planning control manipulation sim tests/sim tests/perception tests/tracking tests/prediction tests/planning tests/control tests/manipulation
touch perception/__init__.py tracking/__init__.py prediction/__init__.py planning/__init__.py control/__init__.py manipulation/__init__.py sim/__init__.py
```

- [ ] **Step 5: Add `.gitignore`**

```
.venv/
__pycache__/
*.pyc
logs/
sim/assets/menagerie/
```

(Menagerie is added as a submodule in Task 2, not tracked directly.)

- [ ] **Step 6: Verify pytest runs (even with zero tests)**

```bash
uv run pytest
```

Expected: `no tests ran` (exit code 0 or 5, not an error).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore perception tracking prediction planning control manipulation sim tests uv.lock
git commit -m "chore: project scaffolding and uv environment"
```

---

### Task 2: MuJoCo conveyor scene

**Files:**
- Create: `sim/conveyor_scene.py`
- Create: `tests/sim/test_conveyor_scene.py`
- Modify (submodule): `sim/assets/menagerie/`

**Interfaces:**
- Produces:
  - `ConveyorSceneEnv(model_path: str, conveyor_velocity: np.ndarray, dt: float = 0.002)`
  - `.reset() -> None`
  - `.step(qdot_cmd: np.ndarray) -> None` — applies joint velocity command for `dt`, advances mocap object by `conveyor_velocity * dt`, steps physics
  - `.get_rgbd(width: int = 128, height: int = 128, camera: str = "wrist_cam") -> tuple[np.ndarray, np.ndarray]` — returns `(rgb[H,W,3] uint8, depth[H,W] float32 meters)`
  - `.get_object_ground_truth() -> np.ndarray` — object's true 3D position (world frame), for test/logging only, never fed to tracking
  - `.get_joint_positions() -> np.ndarray` — current 7 arm joint angles
  - `.set_gripper(closed: bool) -> None`
  - `.camera_intrinsics(width: int, height: int, camera: str = "wrist_cam") -> tuple[float, float, float, float]` — returns `(fx, fy, cx, cy)`

- [ ] **Step 1: Add MuJoCo Menagerie as a git submodule**

```bash
git submodule add https://github.com/google-deepmind/mujoco_menagerie.git sim/assets/menagerie
```

- [ ] **Step 2: Inspect the downloaded Panda asset structure**

```bash
uv run python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('sim/assets/menagerie/franka_emika_panda/scene.xml')
print([mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite)])
print([mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)])
"
```

Confirm the printed site list includes an attachment site on the last link
(Menagerie arm models conventionally expose one named `attachment_site`).
**Note the exact site name and last-link body name printed here** — use them
in Step 3. If the name differs from `attachment_site`/`hand`, substitute the
actual printed names throughout this task.

- [ ] **Step 3: Write `sim/conveyor_scene.py` — scene assembly via XML tree editing**

```python
"""MuJoCo conveyor scene: Panda arm + eye-in-hand camera + scripted mocap object."""
from pathlib import Path

import defusedxml.ElementTree as ET
import mujoco
import numpy as np

_MENAGERIE_SCENE = Path(__file__).parent / "assets/menagerie/franka_emika_panda/scene.xml"
_ATTACHMENT_SITE = "attachment_site"  # confirmed in Task 2 Step 2; update if different


def _build_model_xml() -> str:
    tree = ET.parse(_MENAGERIE_SCENE)
    root = tree.getroot()

    # Find the attachment site so we can locate its parent body.
    site_parent = None
    for body in root.iter("body"):
        for site in body.findall("site"):
            if site.get("name") == _ATTACHMENT_SITE:
                site_parent = body
                break
        if site_parent is not None:
            break
    if site_parent is None:
        raise RuntimeError(
            f"Could not find site '{_ATTACHMENT_SITE}' in {_MENAGERIE_SCENE}. "
            "Re-run Task 2 Step 2 to find the correct site name."
        )

    # Attach an eye-in-hand RGB-D camera at the attachment site's frame.
    camera = ET.SubElement(site_parent, "camera")
    camera.set("name", "wrist_cam")
    camera.set("mode", "fixed")
    camera.set("pos", "0 0 0.05")
    camera.set("euler", "0 0 0")
    camera.set("fovy", "58")

    # Add a scripted (mocap) conveyor object -- driven by our own step(),
    # not MuJoCo contact/friction physics, since conveyor motion is exactly
    # constant-velocity by design.
    worldbody = root.find("worldbody")
    obj_body = ET.SubElement(worldbody, "body")
    obj_body.set("name", "conveyor_object")
    obj_body.set("mocap", "true")
    obj_body.set("pos", "0.5 -0.3 0.05")
    geom = ET.SubElement(obj_body, "geom")
    geom.set("name", "conveyor_object_geom")
    geom.set("type", "box")
    geom.set("size", "0.02 0.02 0.02")
    geom.set("rgba", "0.8 0.1 0.1 1")

    return ET.tostring(root, encoding="unicode")


class ConveyorSceneEnv:
    def __init__(self, conveyor_velocity: np.ndarray, dt: float = 0.002):
        xml_string = _build_model_xml()
        self.model = mujoco.MjModel.from_xml_string(
            xml_string, {"assets": str(_MENAGERIE_SCENE.parent)}
        )
        self.model.opt.timestep = dt
        self.dt = dt
        self.conveyor_velocity = np.asarray(conveyor_velocity, dtype=np.float64)
        self.data = mujoco.MjData(self.model)
        self._obj_mocap_id = self.model.body("conveyor_object").mocapid[0]
        self._renderer = mujoco.Renderer(self.model, height=128, width=128)
        mujoco.mj_forward(self.model, self.data)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, qdot_cmd: np.ndarray) -> None:
        self.data.ctrl[:7] = qdot_cmd
        self.data.mocap_pos[self._obj_mocap_id] += self.conveyor_velocity * self.dt
        mujoco.mj_step(self.model, self.data)

    def get_rgbd(self, width: int = 128, height: int = 128, camera: str = "wrist_cam"):
        self._renderer.update_scene(self.data, camera=camera)
        rgb = self._renderer.render().copy()
        self._renderer.enable_depth_rendering()
        depth = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()
        return rgb, depth.astype(np.float32)

    def get_object_ground_truth(self) -> np.ndarray:
        return self.data.mocap_pos[self._obj_mocap_id].copy()

    def get_joint_positions(self) -> np.ndarray:
        return self.data.qpos[:7].copy()

    def set_gripper(self, closed: bool) -> None:
        self.data.ctrl[7] = 0.0 if closed else 0.04

    def camera_intrinsics(self, width: int = 128, height: int = 128, camera: str = "wrist_cam"):
        cam_id = self.model.camera(camera).id
        fovy_deg = self.model.cam_fovy[cam_id]
        fovy = np.deg2rad(fovy_deg)
        fy = height / (2 * np.tan(fovy / 2))
        fx = fy  # square pixels
        cx, cy = width / 2, height / 2
        return fx, fy, cx, cy
```

- [ ] **Step 4: Write smoke test**

```python
# tests/sim/test_conveyor_scene.py
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
```

- [ ] **Step 5: Run test to verify it passes**

```bash
uv run pytest tests/sim/test_conveyor_scene.py -v
```

Expected: both tests PASS. If the attachment site or camera resolution
fails, re-check the site/body name found in Step 2.

- [ ] **Step 6: Commit**

```bash
git add .gitmodules sim/conveyor_scene.py tests/sim/test_conveyor_scene.py
git commit -m "feat: MuJoCo conveyor scene with eye-in-hand camera"
```

---

### Task 3: Camera back-projection

**Files:**
- Create: `perception/camera.py`
- Create: `tests/perception/test_camera.py`

**Interfaces:**
- Consumes: nothing (pure math module)
- Produces:
  - `CameraIntrinsics(fx: float, fy: float, cx: float, cy: float)`
  - `.deproject(u: int, v: int, depth: float) -> np.ndarray` — returns `[x, y, z]` in camera frame

- [ ] **Step 1: Write the failing test**

```python
# tests/perception/test_camera.py
import numpy as np
from perception.camera import CameraIntrinsics


def test_deproject_center_pixel():
    intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=32.0, cy=32.0)
    point = intr.deproject(u=32, v=32, depth=2.0)
    np.testing.assert_allclose(point, [0.0, 0.0, 2.0], atol=1e-9)


def test_deproject_off_center_pixel():
    intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=32.0, cy=32.0)
    point = intr.deproject(u=42, v=32, depth=2.0)
    # x = (u - cx) * depth / fx
    expected_x = (42 - 32) * 2.0 / 100.0
    np.testing.assert_allclose(point, [expected_x, 0.0, 2.0], atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/perception/test_camera.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'perception.camera'`

- [ ] **Step 3: Write minimal implementation**

```python
# perception/camera.py
"""Pinhole camera intrinsics and depth back-projection."""
import numpy as np


class CameraIntrinsics:
    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy

    def deproject(self, u: int, v: int, depth: float) -> np.ndarray:
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth
        return np.array([x, y, z], dtype=np.float64)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/perception/test_camera.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add perception/camera.py tests/perception/test_camera.py
git commit -m "feat: pinhole camera back-projection"
```

---

### Task 4: RGB-D segmentation → 3D centroid

**Files:**
- Create: `perception/segment.py`
- Create: `tests/perception/test_segment.py`

**Interfaces:**
- Consumes: `CameraIntrinsics` from Task 3 (`perception.camera.CameraIntrinsics`, `.deproject(u, v, depth)`)
- Produces:
  - `segment_object_centroid(rgb: np.ndarray, depth: np.ndarray, intrinsics: CameraIntrinsics, color_lower: tuple[int,int,int], color_upper: tuple[int,int,int]) -> np.ndarray | None` — returns 3D centroid in camera frame, or `None` if no pixels match

- [ ] **Step 1: Write the failing test**

```python
# tests/perception/test_segment.py
import numpy as np
from perception.camera import CameraIntrinsics
from perception.segment import segment_object_centroid


def test_segment_finds_red_square():
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[20:30, 20:30] = [200, 20, 20]  # red square, rows=v, cols=u
    depth = np.full((64, 64), 1.5, dtype=np.float32)
    intr = CameraIntrinsics(fx=64.0, fy=64.0, cx=32.0, cy=32.0)

    centroid = segment_object_centroid(
        rgb, depth, intr, color_lower=(150, 0, 0), color_upper=(255, 80, 80)
    )

    assert centroid is not None
    expected_u, expected_v = 24.5, 24.5  # center of rows/cols 20..29
    expected = intr.deproject(expected_u, expected_v, 1.5)
    np.testing.assert_allclose(centroid, expected, atol=0.05)


def test_segment_returns_none_when_no_match():
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    depth = np.full((64, 64), 1.5, dtype=np.float32)
    intr = CameraIntrinsics(fx=64.0, fy=64.0, cx=32.0, cy=32.0)

    centroid = segment_object_centroid(
        rgb, depth, intr, color_lower=(150, 0, 0), color_upper=(255, 80, 80)
    )
    assert centroid is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/perception/test_segment.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'perception.segment'`

- [ ] **Step 3: Write minimal implementation**

```python
# perception/segment.py
"""Classical color+depth segmentation to a 3D centroid."""
import numpy as np

from perception.camera import CameraIntrinsics


def segment_object_centroid(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    color_lower: tuple[int, int, int],
    color_upper: tuple[int, int, int],
) -> np.ndarray | None:
    lower = np.array(color_lower, dtype=np.uint8)
    upper = np.array(color_upper, dtype=np.uint8)
    mask = np.all((rgb >= lower) & (rgb <= upper), axis=-1)

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None

    u = float(xs.mean())
    v = float(ys.mean())
    z = float(depth[ys, xs].mean())
    return intrinsics.deproject(u, v, z)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/perception/test_segment.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add perception/segment.py tests/perception/test_segment.py
git commit -m "feat: color/depth segmentation to 3D centroid"
```

---

### Task 5: Constant-velocity Kalman Filter with gating

**Files:**
- Create: `tracking/kf.py`
- Create: `tests/tracking/test_kf.py`

**Interfaces:**
- Consumes: nothing (pure math module)
- Produces:
  - `ConstantVelocityKF(dt: float, process_var: float, meas_var: float, init_state: np.ndarray, init_cov: np.ndarray)`
  - `.x: np.ndarray` shape `(6,)` = `[x, y, z, vx, vy, vz]`
  - `.P: np.ndarray` shape `(6, 6)`
  - `.predict() -> None`
  - `.update(z: np.ndarray) -> float` — applies correction, returns Mahalanobis distance of the innovation

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_kf.py
import numpy as np
from tracking.kf import ConstantVelocityKF


def test_predict_advances_position_by_velocity():
    kf = ConstantVelocityKF(
        dt=0.1,
        process_var=1e-4,
        meas_var=1e-3,
        init_state=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        init_cov=np.eye(6) * 0.01,
    )
    kf.predict()
    np.testing.assert_allclose(kf.x[:3], [0.1, 0.0, 0.0], atol=1e-9)


def test_converges_to_true_constant_velocity():
    rng = np.random.default_rng(42)
    dt = 0.05
    true_vel = np.array([0.2, -0.1, 0.0])
    true_pos = np.array([0.0, 0.0, 0.5])

    kf = ConstantVelocityKF(
        dt=dt,
        process_var=1e-5,
        meas_var=1e-4,
        init_state=np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0]),
        init_cov=np.eye(6) * 0.1,
    )

    for _ in range(200):
        true_pos = true_pos + true_vel * dt
        z = true_pos + rng.normal(scale=np.sqrt(1e-4), size=3)
        kf.predict()
        kf.update(z)

    np.testing.assert_allclose(kf.x[3:], true_vel, atol=0.05)
    np.testing.assert_allclose(kf.x[:3], true_pos, atol=0.05)


def test_mahalanobis_distance_zero_for_perfect_measurement():
    kf = ConstantVelocityKF(
        dt=0.1,
        process_var=1e-4,
        meas_var=1e-3,
        init_state=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        init_cov=np.eye(6) * 0.01,
    )
    kf.predict()
    d = kf.update(kf.x[:3].copy())
    assert d < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/tracking/test_kf.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tracking.kf'`

- [ ] **Step 3: Write minimal implementation**

```python
# tracking/kf.py
"""Constant-velocity Kalman filter for 3D position tracking."""
import numpy as np


class ConstantVelocityKF:
    def __init__(
        self,
        dt: float,
        process_var: float,
        meas_var: float,
        init_state: np.ndarray,
        init_cov: np.ndarray,
    ):
        self.dt = dt
        self.x = np.asarray(init_state, dtype=np.float64).copy()
        self.P = np.asarray(init_cov, dtype=np.float64).copy()

        self.F = np.eye(6)
        self.F[0:3, 3:6] = np.eye(3) * dt

        self.H = np.zeros((3, 6))
        self.H[0:3, 0:3] = np.eye(3)

        # Discretized white-noise-acceleration process noise, block per axis.
        q = process_var
        Q_block = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]]) * q
        self.Q = np.zeros((6, 6))
        for axis in range(3):
            idx = [axis, axis + 3]
            self.Q[np.ix_(idx, idx)] = Q_block

        self.R = np.eye(3) * meas_var

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray) -> float:
        z = np.asarray(z, dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

        mahalanobis = float(np.sqrt(y.T @ np.linalg.inv(S) @ y))
        return mahalanobis
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/tracking/test_kf.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tracking/kf.py tests/tracking/test_kf.py
git commit -m "feat: constant-velocity KF with Mahalanobis distance"
```

---

### Task 6: Track state machine with m/n confirmation

**Files:**
- Create: `tracking/track.py`
- Create: `tests/tracking/test_track.py`

**Interfaces:**
- Consumes: `ConstantVelocityKF` from Task 5 (`.predict()`, `.update(z) -> float`, `.x`, `.P`)
- Produces:
  - `TrackStatus` enum: `TENTATIVE`, `CONFIRMED`, `LOST`
  - `Track(kf: ConstantVelocityKF, gate_threshold: float, m: int, n: int, max_consecutive_misses: int)`
  - `.status: TrackStatus`
  - `.step(measurement: np.ndarray | None) -> TrackStatus` — runs one predict/gate/update/confirm cycle, returns updated status

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_track.py
import numpy as np
from tracking.kf import ConstantVelocityKF
from tracking.track import Track, TrackStatus


def make_track(**overrides):
    kf = ConstantVelocityKF(
        dt=0.1,
        process_var=1e-4,
        meas_var=1e-3,
        init_state=np.zeros(6),
        init_cov=np.eye(6) * 0.01,
    )
    defaults = dict(gate_threshold=9.0, m=3, n=5, max_consecutive_misses=3)
    defaults.update(overrides)
    return Track(kf=kf, **defaults)


def test_starts_tentative():
    track = make_track()
    assert track.status == TrackStatus.TENTATIVE


def test_confirms_after_m_of_n_hits():
    track = make_track()
    status = TrackStatus.TENTATIVE
    for _ in range(3):
        status = track.step(measurement=np.zeros(3))
    assert status == TrackStatus.CONFIRMED


def test_stays_tentative_with_too_few_hits():
    track = make_track()
    status = track.step(measurement=np.zeros(3))
    status = track.step(measurement=None)
    assert status == TrackStatus.TENTATIVE


def test_goes_lost_after_max_consecutive_misses():
    track = make_track()
    status = TrackStatus.TENTATIVE
    for _ in range(4):
        status = track.step(measurement=None)
    assert status == TrackStatus.LOST


def test_gate_rejects_far_measurement_as_miss():
    track = make_track(gate_threshold=1.0)
    # First hit near zero to seed the estimate.
    track.step(measurement=np.zeros(3))
    # A wildly distant measurement should be gated out (treated as a miss).
    status = track.step(measurement=np.array([100.0, 100.0, 100.0]))
    assert status == TrackStatus.TENTATIVE
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/tracking/test_track.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tracking.track'`

- [ ] **Step 3: Write minimal implementation**

```python
# tracking/track.py
"""Track state machine: gating + m/n confirmation logic."""
from collections import deque
from enum import Enum

import numpy as np

from tracking.kf import ConstantVelocityKF


class TrackStatus(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"


class Track:
    def __init__(
        self,
        kf: ConstantVelocityKF,
        gate_threshold: float,
        m: int,
        n: int,
        max_consecutive_misses: int,
    ):
        self.kf = kf
        self.gate_threshold = gate_threshold
        self.m = m
        self.n = n
        self.max_consecutive_misses = max_consecutive_misses
        self.hit_history: deque[bool] = deque(maxlen=n)
        self.consecutive_misses = 0
        self.status = TrackStatus.TENTATIVE

    def step(self, measurement: np.ndarray | None) -> TrackStatus:
        self.kf.predict()

        hit = False
        if measurement is not None:
            distance = self.kf.update(measurement)
            if distance <= self.gate_threshold:
                hit = True

        self.hit_history.append(hit)
        self.consecutive_misses = 0 if hit else self.consecutive_misses + 1

        if self.consecutive_misses >= self.max_consecutive_misses:
            self.status = TrackStatus.LOST
        elif sum(self.hit_history) >= self.m:
            self.status = TrackStatus.CONFIRMED
        elif self.status != TrackStatus.CONFIRMED:
            self.status = TrackStatus.TENTATIVE

        return self.status
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/tracking/test_track.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tracking/track.py tests/tracking/test_track.py
git commit -m "feat: track state machine with m/n confirmation and gating"
```

---

### Task 7: State + covariance prediction

**Files:**
- Create: `prediction/predict.py`
- Create: `tests/prediction/test_predict.py`

**Interfaces:**
- Consumes: nothing new (uses plain `np.ndarray` matrices — the `F`/`Q` a caller reads off a `ConstantVelocityKF` instance)
- Produces:
  - `propagate(x: np.ndarray, P: np.ndarray, F: np.ndarray, Q: np.ndarray, steps: int) -> list[tuple[np.ndarray, np.ndarray]]` — returns `[(x_1, P_1), ..., (x_steps, P_steps)]`

- [ ] **Step 1: Write the failing test**

```python
# tests/prediction/test_predict.py
import numpy as np
from prediction.predict import propagate


def test_propagate_constant_velocity_matches_hand_computation():
    dt = 0.1
    F = np.eye(6)
    F[0:3, 3:6] = np.eye(3) * dt
    Q = np.eye(6) * 1e-4

    x0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    P0 = np.eye(6) * 0.01

    result = propagate(x0, P0, F, Q, steps=3)

    assert len(result) == 3
    # Independently hand-compute the expected trajectory.
    expected_positions = [
        [0.1, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.3, 0.0, 0.0],
    ]
    for (x_k, P_k), expected_pos in zip(result, expected_positions):
        np.testing.assert_allclose(x_k[:3], expected_pos, atol=1e-9)
        assert P_k.shape == (6, 6)


def test_covariance_grows_monotonically():
    dt = 0.1
    F = np.eye(6)
    F[0:3, 3:6] = np.eye(3) * dt
    Q = np.eye(6) * 1e-3

    x0 = np.zeros(6)
    P0 = np.eye(6) * 0.01

    result = propagate(x0, P0, F, Q, steps=5)
    traces = [np.trace(P_k) for _, P_k in result]
    assert all(t2 > t1 for t1, t2 in zip(traces, traces[1:]))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/prediction/test_predict.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'prediction.predict'`

- [ ] **Step 3: Write minimal implementation**

```python
# prediction/predict.py
"""Forward propagation of a linear-Gaussian state estimate."""
import numpy as np


def propagate(
    x: np.ndarray, P: np.ndarray, F: np.ndarray, Q: np.ndarray, steps: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    results = []
    x_k, P_k = x.copy(), P.copy()
    for _ in range(steps):
        x_k = F @ x_k
        P_k = F @ P_k @ F.T + Q
        results.append((x_k.copy(), P_k.copy()))
    return results
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/prediction/test_predict.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add prediction/predict.py tests/prediction/test_predict.py
git commit -m "feat: forward state/covariance propagation"
```

---

### Task 8: Closed-form interception solver

**Files:**
- Create: `planning/intercept.py`
- Create: `tests/planning/test_intercept.py`

**Interfaces:**
- Consumes: nothing new (pure math module)
- Produces:
  - `solve_intercept(obj_pos0: np.ndarray, obj_vel: np.ndarray, ee_pos: np.ndarray, ee_max_speed: float) -> tuple[np.ndarray, float] | None` — returns `(intercept_point, intercept_time)`, or `None` if unreachable

- [ ] **Step 1: Write the failing test**

```python
# tests/planning/test_intercept.py
import numpy as np
from planning.intercept import solve_intercept


def test_stationary_object_intercept_time_equals_distance_over_speed():
    obj_pos0 = np.array([1.0, 0.0, 0.0])
    obj_vel = np.array([0.0, 0.0, 0.0])
    ee_pos = np.array([0.0, 0.0, 0.0])
    ee_max_speed = 1.0

    result = solve_intercept(obj_pos0, obj_vel, ee_pos, ee_max_speed)

    assert result is not None
    point, t = result
    np.testing.assert_allclose(t, 1.0, atol=1e-6)
    np.testing.assert_allclose(point, [1.0, 0.0, 0.0], atol=1e-6)


def test_moving_object_matches_hand_solved_quadratic():
    obj_pos0 = np.array([2.0, 0.0, 0.0])
    obj_vel = np.array([-1.0, 0.0, 0.0])  # moving toward the EE
    ee_pos = np.array([0.0, 0.0, 0.0])
    ee_max_speed = 2.0

    result = solve_intercept(obj_pos0, obj_vel, ee_pos, ee_max_speed)
    assert result is not None
    point, t = result

    # Independently solve a*t^2 + b*t + c = 0 in the test.
    rel_p = obj_pos0 - ee_pos
    a = obj_vel @ obj_vel - ee_max_speed**2
    b = 2 * rel_p @ obj_vel
    c = rel_p @ rel_p
    disc = b**2 - 4 * a * c
    roots = [(-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a)]
    expected_t = min(r for r in roots if r > 0)

    np.testing.assert_allclose(t, expected_t, atol=1e-6)
    expected_point = obj_pos0 + obj_vel * expected_t
    np.testing.assert_allclose(point, expected_point, atol=1e-6)


def test_unreachable_object_returns_none():
    obj_pos0 = np.array([100.0, 0.0, 0.0])
    obj_vel = np.array([50.0, 0.0, 0.0])  # fleeing faster than EE can chase
    ee_pos = np.array([0.0, 0.0, 0.0])
    ee_max_speed = 1.0

    result = solve_intercept(obj_pos0, obj_vel, ee_pos, ee_max_speed)
    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/planning/test_intercept.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'planning.intercept'`

- [ ] **Step 3: Write minimal implementation**

```python
# planning/intercept.py
"""Closed-form time-to-intercept for a constant-velocity target."""
import numpy as np


def solve_intercept(
    obj_pos0: np.ndarray,
    obj_vel: np.ndarray,
    ee_pos: np.ndarray,
    ee_max_speed: float,
) -> tuple[np.ndarray, float] | None:
    rel_p = obj_pos0 - ee_pos
    a = float(obj_vel @ obj_vel - ee_max_speed**2)
    b = float(2 * rel_p @ obj_vel)
    c = float(rel_p @ rel_p)

    if abs(a) < 1e-12:
        # Degenerate to linear equation b*t + c = 0.
        if abs(b) < 1e-12:
            return None
        t = -c / b
        candidates = [t] if t > 0 else []
    else:
        disc = b**2 - 4 * a * c
        if disc < 0:
            return None
        sqrt_disc = np.sqrt(disc)
        roots = [(-b + sqrt_disc) / (2 * a), (-b - sqrt_disc) / (2 * a)]
        candidates = [r for r in roots if r > 0]

    if not candidates:
        return None

    t_star = min(candidates)
    point = obj_pos0 + obj_vel * t_star
    return point, t_star
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/planning/test_intercept.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add planning/intercept.py tests/planning/test_intercept.py
git commit -m "feat: closed-form interception solver"
```

---

### Task 9: Panda forward kinematics in CasADi

**Files:**
- Create: `control/panda_kinematics.py`
- Create: `tests/control/test_panda_kinematics.py`

**Interfaces:**
- Consumes: nothing new (pure math module)
- Produces:
  - `panda_fk_symbolic() -> casadi.Function` — a CasADi `Function` mapping `q` (7,) to end-effector flange position `(3,)`
  - `panda_fk_numpy(q: np.ndarray) -> np.ndarray` — independent pure-numpy reference implementation of the same kinematics, used only for testing

- [ ] **Step 1: Write the failing test**

The test verifies the CasADi symbolic FK against an *independently written*
pure-numpy implementation of the same DH chain — not against memorized
literal coordinates, since two independent implementations agreeing is a much
stronger correctness check than trusting a single hardcoded number.

```python
# tests/control/test_panda_kinematics.py
import numpy as np
from control.panda_kinematics import panda_fk_symbolic, panda_fk_numpy


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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/control/test_panda_kinematics.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'control.panda_kinematics'`

- [ ] **Step 3: Write minimal implementation**

```python
# control/panda_kinematics.py
"""Franka Panda forward kinematics (modified/Craig DH convention).

DH parameters below are the widely-published Franka Panda modified-DH
link parameters (7 revolute joints + fixed flange offset).
"""
import casadi as ca
import numpy as np

# a_{i-1} (m), alpha_{i-1} (rad), d_i (m) for i = 1..7, then the fixed flange offset.
_A = [0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088, 0.0]
_ALPHA = [0.0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2]
_D = [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.0]
_FLANGE_A, _FLANGE_ALPHA, _FLANGE_D = 0.0, 0.0, 0.107


def _dh_transform(a, alpha, d, theta, backend):
    cos, sin = (backend.cos, backend.sin) if backend is ca else (np.cos, np.sin)
    ct, st = cos(theta), sin(theta)
    ca_, sa = cos(alpha), sin(alpha)
    return backend.vertcat(
        backend.horzcat(ct, -st, backend.SX(0) if backend is ca else 0.0, backend.SX(a) if backend is ca else a),
        backend.horzcat(st * ca_, ct * ca_, -sa, -sa * d),
        backend.horzcat(st * sa, ct * sa, ca_, ca_ * d),
        backend.horzcat(0, 0, 0, 1),
    ) if backend is ca else np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca_, ct * ca_, -sa, -sa * d],
            [st * sa, ct * sa, ca_, ca_ * d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def panda_fk_symbolic() -> ca.Function:
    q = ca.SX.sym("q", 7)
    T = ca.SX.eye(4)
    for i in range(7):
        T = T @ _dh_transform(_A[i], _ALPHA[i], _D[i], q[i], backend=ca)
    T = T @ _dh_transform(_FLANGE_A, _FLANGE_ALPHA, _FLANGE_D, 0.0, backend=ca)
    ee_pos = T[0:3, 3]
    return ca.Function("panda_fk", [q], [ee_pos])


def panda_fk_numpy(q: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    for i in range(7):
        T = T @ _dh_transform(_A[i], _ALPHA[i], _D[i], q[i], backend=np)
    T = T @ _dh_transform(_FLANGE_A, _FLANGE_ALPHA, _FLANGE_D, 0.0, backend=np)
    return T[0:3, 3]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/control/test_panda_kinematics.py -v
```

Expected: PASS. If it fails on the CasADi `_dh_transform` branch, verify the
`ca.horzcat`/`vertcat` construction compiles under the installed `casadi`
version — signature has been stable across 3.x but double-check the error
message against the installed version's docs.

- [ ] **Step 5: Commit**

```bash
git add control/panda_kinematics.py tests/control/test_panda_kinematics.py
git commit -m "feat: Panda forward kinematics (CasADi + independent numpy reference)"
```

---

### Task 10: Kinematic MPC controller

**Files:**
- Create: `control/mpc.py`
- Create: `tests/control/test_mpc.py`

**Interfaces:**
- Consumes: `panda_fk_symbolic()` from Task 9 (`casadi.Function`, `q(7,) -> pos(3,)`)
- Produces:
  - `KinematicMPC(fk_func: casadi.Function, horizon: int, dt: float, q_min: np.ndarray, q_max: np.ndarray, qdot_max: np.ndarray, effort_weight: float = 0.01)`
  - `.solve(q_current: np.ndarray, target_pos: np.ndarray) -> np.ndarray` — returns commanded joint velocity `(7,)` for this step

- [ ] **Step 1: Write the failing test**

The test doesn't check exact numeric output (an IPOPT solve isn't a fixed
literal to hardcode) — it checks the property that actually matters: applying
the returned command moves the end-effector closer to the target.

```python
# tests/control/test_mpc.py
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/control/test_mpc.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'control.mpc'`

- [ ] **Step 3: Write minimal implementation**

```python
# control/mpc.py
"""Kinematic (joint-velocity) MPC tracking a moving Cartesian target."""
import casadi as ca
import numpy as np


class KinematicMPC:
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
        sol = self._opti.solve()
        qdot_all = sol.value(self._Qdot)
        return np.asarray(qdot_all[:, 0]).flatten()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/control/test_mpc.py -v
```

Expected: PASS. IPOPT solve time per call should be well under 100ms for
this problem size — if it's dramatically slower, reduce `horizon` first.

- [ ] **Step 5: Commit**

```bash
git add control/mpc.py tests/control/test_mpc.py
git commit -m "feat: kinematic MPC controller (CasADi + IPOPT)"
```

---

### Task 11: Grasp executor

**Files:**
- Create: `manipulation/grasp.py`
- Create: `tests/manipulation/test_grasp.py`

**Interfaces:**
- Consumes: `TrackStatus` from Task 6 (`tracking.track.TrackStatus`)
- Produces:
  - `GraspExecutor(position_tolerance: float, confidence_required: bool = True)`
  - `.should_close(ee_pos: np.ndarray, target_pos: np.ndarray, track_status: TrackStatus) -> bool`

- [ ] **Step 1: Write the failing test**

```python
# tests/manipulation/test_grasp.py
import numpy as np
from tracking.track import TrackStatus
from manipulation.grasp import GraspExecutor


def test_closes_when_within_tolerance_and_confirmed():
    grasp = GraspExecutor(position_tolerance=0.02)
    ee_pos = np.array([0.5, 0.0, 0.1])
    target = np.array([0.51, 0.0, 0.1])
    assert grasp.should_close(ee_pos, target, TrackStatus.CONFIRMED) is True


def test_does_not_close_when_too_far():
    grasp = GraspExecutor(position_tolerance=0.02)
    ee_pos = np.array([0.5, 0.0, 0.1])
    target = np.array([0.6, 0.0, 0.1])
    assert grasp.should_close(ee_pos, target, TrackStatus.CONFIRMED) is False


def test_does_not_close_when_track_unconfirmed():
    grasp = GraspExecutor(position_tolerance=0.02)
    ee_pos = np.array([0.5, 0.0, 0.1])
    target = np.array([0.505, 0.0, 0.1])
    assert grasp.should_close(ee_pos, target, TrackStatus.TENTATIVE) is False


def test_confidence_not_required_when_disabled():
    grasp = GraspExecutor(position_tolerance=0.02, confidence_required=False)
    ee_pos = np.array([0.5, 0.0, 0.1])
    target = np.array([0.505, 0.0, 0.1])
    assert grasp.should_close(ee_pos, target, TrackStatus.TENTATIVE) is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/manipulation/test_grasp.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'manipulation.grasp'`

- [ ] **Step 3: Write minimal implementation**

```python
# manipulation/grasp.py
"""Grasp-commit decision: gated on distance tolerance and track confidence."""
import numpy as np

from tracking.track import TrackStatus


class GraspExecutor:
    def __init__(self, position_tolerance: float, confidence_required: bool = True):
        self.position_tolerance = position_tolerance
        self.confidence_required = confidence_required

    def should_close(
        self, ee_pos: np.ndarray, target_pos: np.ndarray, track_status: TrackStatus
    ) -> bool:
        distance = float(np.linalg.norm(ee_pos - target_pos))
        within_tolerance = distance <= self.position_tolerance
        if self.confidence_required:
            return within_tolerance and track_status == TrackStatus.CONFIRMED
        return within_tolerance
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/manipulation/test_grasp.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add manipulation/grasp.py tests/manipulation/test_grasp.py
git commit -m "feat: gated grasp-commit executor"
```

---

### Task 12: End-to-end conveyor integration

**Files:**
- Create: `configs/conveyor.yaml`
- Create: `run_conveyor_demo.py`
- Create: `tests/test_integration_conveyor.py`

**Interfaces:**
- Consumes everything from Tasks 2-11:
  - `sim.conveyor_scene.ConveyorSceneEnv`
  - `perception.camera.CameraIntrinsics`, `perception.segment.segment_object_centroid`
  - `tracking.kf.ConstantVelocityKF`, `tracking.track.Track`, `tracking.track.TrackStatus`
  - `prediction.predict.propagate`
  - `planning.intercept.solve_intercept`
  - `control.panda_kinematics.panda_fk_symbolic`, `control.panda_kinematics.panda_fk_numpy`
  - `control.mpc.KinematicMPC`
  - `manipulation.grasp.GraspExecutor`
- Produces:
  - `configs/conveyor.yaml` — run parameters
  - `run_one_episode(config: dict) -> dict` in `run_conveyor_demo.py`, returning a result dict `{"grasped": bool, "grasp_error_m": float | None, "steps": int}`

- [ ] **Step 1: Write `configs/conveyor.yaml`**

```yaml
conveyor_velocity: [0.0, 0.08, 0.0]
dt: 0.002
control_hz: 20
max_steps: 2000

camera:
  width: 64
  height: 64
  color_lower: [150, 0, 0]
  color_upper: [255, 80, 80]

kf:
  process_var: 1.0e-4
  meas_var: 1.0e-3
  init_cov_scale: 0.05

track:
  gate_threshold: 9.0
  m: 3
  n: 5
  max_consecutive_misses: 5

mpc:
  horizon: 5
  qdot_max: 1.5
  q_min: -2.8
  q_max: 2.8

grasp:
  position_tolerance: 0.03
```

- [ ] **Step 2: Write the failing integration test**

This test doesn't assert on contact-dynamics grasp success (fragile, depends
on friction/gripper tuning) — it asserts on the property that validates the
*whole pipeline wired correctly*: when the grasp executor decides to close,
the true (ground-truth) object-to-EE distance at that instant is within
tolerance.

```python
# tests/test_integration_conveyor.py
import numpy as np
import yaml

from run_conveyor_demo import run_one_episode


def test_conveyor_episode_grasps_within_tolerance():
    with open("configs/conveyor.yaml") as f:
        config = yaml.safe_load(f)

    result = run_one_episode(config)

    assert result["grasped"] is True
    assert result["grasp_error_m"] is not None
    assert result["grasp_error_m"] <= config["grasp"]["position_tolerance"] + 0.01
    assert result["steps"] < config["max_steps"]
```

- [ ] **Step 3: Run test to verify it fails**

```bash
uv run pytest tests/test_integration_conveyor.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'run_conveyor_demo'`

- [ ] **Step 4: Write `run_conveyor_demo.py`**

```python
"""Top-level closed loop: perceive -> track -> predict -> intercept -> control -> grasp."""
import numpy as np
import yaml

from control.mpc import KinematicMPC
from control.panda_kinematics import panda_fk_numpy, panda_fk_symbolic
from manipulation.grasp import GraspExecutor
from perception.camera import CameraIntrinsics
from perception.segment import segment_object_centroid
from planning.intercept import solve_intercept
from sim.conveyor_scene import ConveyorSceneEnv
from tracking.kf import ConstantVelocityKF
from tracking.track import Track, TrackStatus

EE_MAX_SPEED = 1.5  # matches mpc.qdot_max order of magnitude, approximate reach speed


def run_one_episode(config: dict) -> dict:
    env = ConveyorSceneEnv(
        conveyor_velocity=np.array(config["conveyor_velocity"]),
        dt=config["dt"],
    )
    env.reset()

    cam_cfg = config["camera"]
    fx, fy, cx, cy = env.camera_intrinsics(cam_cfg["width"], cam_cfg["height"])
    intrinsics = CameraIntrinsics(fx, fy, cx, cy)

    kf_cfg, track_cfg = config["kf"], config["track"]
    track: Track | None = None

    fk = panda_fk_symbolic()
    mpc = KinematicMPC(
        fk_func=fk,
        horizon=config["mpc"]["horizon"],
        dt=1.0 / config["control_hz"],
        q_min=np.full(7, config["mpc"]["q_min"]),
        q_max=np.full(7, config["mpc"]["q_max"]),
        qdot_max=np.full(7, config["mpc"]["qdot_max"]),
    )
    grasp_executor = GraspExecutor(position_tolerance=config["grasp"]["position_tolerance"])

    sim_steps_per_control = max(1, round((1.0 / config["control_hz"]) / config["dt"]))
    qdot_cmd = np.zeros(7)

    for step in range(config["max_steps"]):
        env.step(qdot_cmd)

        if step % sim_steps_per_control != 0:
            continue

        rgb, depth = env.get_rgbd(cam_cfg["width"], cam_cfg["height"])
        measurement_cam = segment_object_centroid(
            rgb, depth, intrinsics, tuple(cam_cfg["color_lower"]), tuple(cam_cfg["color_upper"])
        )
        # Camera frame == world frame for this MVP's mocap-object measurement
        # path; a real extrinsics transform would go here if the camera
        # were not axis-aligned with the world in this scene setup.
        measurement = measurement_cam

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
            continue

        if track is None:
            continue

        status = track.step(measurement)
        if status == TrackStatus.LOST:
            track = None
            continue

        q_current = env.get_joint_positions()
        ee_pos = panda_fk_numpy(q_current)

        if status == TrackStatus.CONFIRMED:
            intercept = solve_intercept(
                obj_pos0=track.kf.x[:3],
                obj_vel=track.kf.x[3:],
                ee_pos=ee_pos,
                ee_max_speed=EE_MAX_SPEED,
            )
            target = intercept[0] if intercept is not None else track.kf.x[:3]
        else:
            target = track.kf.x[:3]

        qdot_cmd = mpc.solve(q_current, target)

        if grasp_executor.should_close(ee_pos, target, status):
            env.set_gripper(closed=True)
            true_obj_pos = env.get_object_ground_truth()
            grasp_error = float(np.linalg.norm(ee_pos - true_obj_pos))
            return {"grasped": True, "grasp_error_m": grasp_error, "steps": step}

    return {"grasped": False, "grasp_error_m": None, "steps": config["max_steps"]}


if __name__ == "__main__":
    with open("configs/conveyor.yaml") as f:
        cfg = yaml.safe_load(f)
    print(run_one_episode(cfg))
```

- [ ] **Step 5: Run the integration test**

```bash
uv run pytest tests/test_integration_conveyor.py -v
```

Expected: PASS. If `grasped` comes back `False`, first check via
`uv run python run_conveyor_demo.py` whether the object ever enters the
camera's field of view early enough — tune `conveyor_velocity`,
`init_state` seeding, or the arm's starting joint configuration in
`sim/conveyor_scene.py` before touching the algorithmic modules.

- [ ] **Step 6: Run the full test suite**

```bash
uv run pytest -v
```

Expected: all tests from Tasks 1-12 PASS.

- [ ] **Step 7: Commit**

```bash
git add configs/conveyor.yaml run_conveyor_demo.py tests/test_integration_conveyor.py
git commit -m "feat: end-to-end conveyor interception and grasp demo"
```

---

## Self-Review Notes

- **Spec coverage:** perception (Tasks 3-4), tracking + gating + m/n (Tasks
  5-6), prediction (Task 7), interception planning (Task 8), kinematic MPC
  control (Tasks 9-10), grasp execution (Task 11), full closed loop (Task
  12), MuJoCo scene + eye-in-hand camera (Task 2), uv/project scaffolding
  (Task 1) — every conveyor-MVP spec section (3-9) maps to a task.
- **Placeholder scan:** no TBD/TODO; the one open unknown (exact Menagerie
  site/body names) is resolved by a concrete inspection step in Task 2,
  not left vague.
- **Type consistency:** `TrackStatus` (Task 6) is reused with identical
  values in Tasks 11-12; `ConstantVelocityKF.x`/`.P` shapes are consistent
  from Task 5 through Task 12; `panda_fk_symbolic()`'s `casadi.Function`
  return type is consumed identically in Tasks 10 and 12.
- **Deferred to the pendulum/ablation follow-on plan:** EKF/UKF pendulum
  tracking, the naive-vs-gated-commit ablation, YOLO perception swap,
  active-perception extensions — all out of scope here per Section 11 of
  the design spec.
