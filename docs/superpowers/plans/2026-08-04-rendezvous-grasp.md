# Rendezvous Grasp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Franka actually pick up the cube — replace the pursuit
approach with a rendezvous state machine and fix three perception-geometry
defects, so `contact_verified` reads `True` for a real, lift-verified grasp.

**Architecture:** `perception/yolo_segment.py` gains border rejection, a
top-face centroid, and a world-Z height offset (returning a world-frame
point). `run_conveyor_demo.py::run_one_episode` replaces its
chase-and-commit loop with a TRACK → GOTO → WAIT → CLOSE state machine that
parks the arm ahead of the object and closes as it arrives.

**Tech Stack:** Python 3.11, `mujoco`, `ultralytics`, `casadi`, `uv`.

## Global Constraints

- Full design rationale, measured evidence, and exact expected numbers:
  `docs/superpowers/specs/2026-08-04-rendezvous-grasp-design.md`. Read it.
- Do NOT modify `tests/test_integration_conveyor.py`. Its
  `assert result["contact_verified"] is True` is the acceptance gate and
  must pass by fixing the system, never by weakening the test.
- Do NOT modify `perception/segment.py` (the classical baseline stays as-is
  and stays tested), `tracking/`, `prediction/`, `planning/`, `control/`,
  `manipulation/`, or `sim/`.
- All constants must be module-level named constants with a comment giving
  the measured justification — no bare magic numbers inline.
- `uv run ruff check .` and `uv run ruff format --check .` must pass (CI
  runs both).

---

### Task 1: Perception geometry fixes

**Files:**
- Modify: `perception/yolo_segment.py`
- Modify: `tests/perception/test_yolo_segment.py`

**Interfaces produced (Task 2 consumes):**

```python
_TOP_FACE_DEPTH_TOL_M = 0.010
def yolo_centroid(
    rgb, depth, model, intrinsics,
    color_lower, color_upper,
    cam_pos, cam_mat,          # NEW: required, world transform happens inside
    depth_bias: float = 0.0,
    reject_border: bool = True,
) -> np.ndarray | None        # NOW RETURNS A WORLD-FRAME POINT
```

- [ ] **Step 1: Update the existing tests for the new contract**

`tests/perception/test_yolo_segment.py` currently converts `yolo_centroid`'s
camera-frame return to world via `_camera_point_to_world`. The function now
returns world frame directly. Update both existing tests to pass
`cam_pos`/`cam_mat` and drop the external conversion — the ground-truth
comparison and the `None` assertion are otherwise unchanged.

Get `cam_pos`/`cam_mat` the same way the demo does:
```python
cam_id = env.model.camera("wrist_cam").id
cam_pos = env.data.cam_xpos[cam_id].copy()
cam_mat = env.data.cam_xmat[cam_id].reshape(3, 3).copy()
```

- [ ] **Step 2: Add a test pinning border rejection**

```python
def test_yolo_centroid_rejects_border_clipped_detection():
    """A detection touching the image edge has a provably unreliable
    centroid: the object's true centre projects off-image and the measured
    centroid is clamped inward (design spec 2.2 -- measured max residual
    11.2mm vs 2.5mm once rejected). Treat it as a miss."""
    from ultralytics import YOLO
    env = ConveyorSceneEnv(conveyor_velocity=np.array([0.0, 0.08, 0.0]))
    env.reset()
    # y = -0.20 puts the cube at the very edge of the wrist camera's view.
    _place_cube_in_view(env, x=0.5, y=-0.20, z=0.05)
    model = YOLO(str(MODEL_PATH))
    rgb, depth = env.get_rgbd(64, 64)
    fx, fy, cx, cy = env.camera_intrinsics(64, 64)
    intrinsics = CameraIntrinsics(fx, fy, cx, cy)
    cam_id = env.model.camera("wrist_cam").id
    cam_pos = env.data.cam_xpos[cam_id].copy()
    cam_mat = env.data.cam_xmat[cam_id].reshape(3, 3).copy()
    args = (rgb, depth, model, intrinsics, (150, 0, 0), (255, 80, 80),
            cam_pos, cam_mat)
    assert yolo_centroid(*args, depth_bias=OBJECT_HALF_HEIGHT_M,
                         reject_border=True) is None
```

- [ ] **Step 3: Run the tests, confirm they fail**

`uv run pytest tests/perception/test_yolo_segment.py -v` — expect failures
(signature mismatch / border test not implemented).

- [ ] **Step 4: Implement**

Rewrite `yolo_centroid`'s body. Keep the existing module docstring's
explanation of the zero-colour-match-is-a-miss rule and extend it for the
new behaviour. The logic, in order:

```python
results = model.predict(source=rgb, verbose=False)[0]
if len(results.boxes) == 0:
    return None
confs = results.boxes.conf.cpu().numpy()
best = int(np.argmax(confs))
x_min, y_min, x_max, y_max = results.boxes.xyxy[best].cpu().numpy()

height_px, width_px = depth.shape
if reject_border and (
    x_min <= 1 or y_min <= 1 or x_max >= width_px - 1 or y_max >= height_px - 1
):
    return None

y0, y1 = int(max(0, y_min)), int(min(height_px, y_max + 1))
x0, x1 = int(max(0, x_min)), int(min(width_px, x_max + 1))
region_rgb, region_depth = rgb[y0:y1, x0:x1], depth[y0:y1, x0:x1]
lower = np.array(color_lower, dtype=np.uint8)
upper = np.array(color_upper, dtype=np.uint8)
mask = np.all((region_rgb >= lower) & (region_rgb <= upper), axis=-1)
if mask.sum() == 0:
    return None

# The top face is nearest the camera; restrict to it so the centroid is
# the face's centre (whose x/y equal the cube's) rather than a silhouette
# centre pulled toward the optical axis by a visible side face.
top_face = mask & (region_depth <= float(region_depth[mask].min()) + _TOP_FACE_DEPTH_TOL_M)
if top_face.sum() == 0:
    return None
ys, xs = np.nonzero(top_face)
u = x0 + float(xs.mean())
v = y0 + float(ys.mean())
z = float(region_depth[top_face].mean())

point_cam = intrinsics.deproject(u, v, z)
point_world = _camera_point_to_world(point_cam, cam_pos, cam_mat)
# The measured point is on the TOP FACE; the centre is depth_bias below it
# in WORLD Z -- not along the camera ray (design spec 2.3).
return point_world - np.array([0.0, 0.0, depth_bias])
```

`_camera_point_to_world` currently lives in `run_conveyor_demo.py`.
Importing it there from `perception/` would be a circular/layering
inversion, so **move** it into `perception/camera.py` as a public
`camera_point_to_world(point_cam, cam_pos, cam_mat)`, and have
`run_conveyor_demo.py` import it from there (keeping its existing
`_camera_point_to_world` name as an alias assignment is acceptable to avoid
touching its other call sites). Update
`tests/perception/test_yolo_segment.py`'s import accordingly.

- [ ] **Step 5: Verify and commit**

```bash
uv run pytest tests/perception/ -v
uv run ruff check . && uv run ruff format --check .
git add perception/ tests/perception/ run_conveyor_demo.py
git commit -m "fix: reject border-clipped detections, use top-face centroid and world-Z offset"
```

---

### Task 2: Rendezvous approach in the closed loop

**Files:**
- Modify: `run_conveyor_demo.py`

**Interfaces consumed:** Task 1's `yolo_centroid` (world-frame return,
`cam_pos`/`cam_mat` args, `reject_border`) and
`perception.camera.camera_point_to_world`.

- [ ] **Step 1: Add the constants**

```python
# Rendezvous approach (design spec 2.1/3.2). The arm parks AHEAD of the
# object on its path and lets it arrive, rather than chasing it: measured
# with pursuit, the arm never converges (best approach 3.15cm, falling
# behind to a 5-8cm steady state), so the commit necessarily fired while
# the arm was still moving and it then coasted ~2cm into the object.
_PLATFORM_TOP_Z = 0.03                  # sim/conveyor_scene.py's platform top
_GRASP_Z = _PLATFORM_TOP_Z + OBJECT_HALF_HEIGHT_M   # cube centre height
_RENDEZVOUS_TIME_BUDGET_S = 3.0         # object-travel time placed ahead of it
_SETTLE_TOL_M = 0.008
_GOTO_TIMEOUT_S = 4.0
_GOTO_STALL_QDOT = 0.05
_GOTO_STALL_DIST_M = 0.03
_CLOSE_LEAD_S = 0.05                    # finger-closing dead time
```

- [ ] **Step 2: Replace the approach logic**

Inside `run_one_episode`'s control-tick block, replace the pursuit logic
(the `moving` gate, the `solve_intercept`/`blend`/`target` computation, and
the `grasp_executor.should_close(...)` trigger) with a phase machine.
Initialise before the loop: `phase = "TRACK"`, `rendezvous = None`,
`goto_ticks = 0`, `prev_offset = None`, `last_meas = None`,
`last_meas_t = None`, and `tick_dt = 1.0 / config["control_hz"]`.

Perception call becomes (note world frame, no external transform):
```python
measurement = yolo_centroid(
    rgb, depth, detector, intrinsics,
    tuple(cam_cfg["color_lower"]), tuple(cam_cfg["color_upper"]),
    cam_pos, cam_mat, depth_bias=OBJECT_HALF_HEIGHT_M,
)
```
Record `last_meas`/`last_meas_t = step * config["dt"]` whenever
`measurement is not None`.

After `track.step(...)` and the existing LOST handling:

```python
q_current = env.get_joint_positions()
ee_pos = panda_tcp_numpy(q_current)
obj_est = track.kf.x[:3].copy()
obj_vel = track.kf.x[3:].copy()
vel_horizontal = obj_vel.copy()
vel_horizontal[2] = 0.0        # the cube slides on a plane; a nonzero z
                               # velocity estimate is noise and must not be
                               # extrapolated (it shifted an earlier
                               # rendezvous point 1.9cm too high)
speed = float(np.linalg.norm(vel_horizontal))

if phase == "TRACK":
    if status == TrackStatus.CONFIRMED and speed > 1e-3:
        rendezvous = obj_est + (vel_horizontal / speed) * (speed * _RENDEZVOUS_TIME_BUDGET_S)
        rendezvous[2] = _GRASP_Z
        phase, goto_ticks = "GOTO", 0
    qdot_cmd = np.zeros(7)
    continue

if phase == "GOTO":
    qdot_cmd = mpc.solve(q_current, rendezvous)
    goto_ticks += 1
    dist = float(np.linalg.norm(ee_pos - rendezvous))
    stalled = float(np.linalg.norm(qdot_cmd)) < _GOTO_STALL_QDOT and dist < _GOTO_STALL_DIST_M
    if dist < _SETTLE_TOL_M or stalled or goto_ticks * tick_dt > _GOTO_TIMEOUT_S:
        phase = "WAIT"
        qdot_cmd = np.zeros(7)
    continue

# phase == "WAIT": arm stationary, gripper open, straddling the path.
qdot_cmd = np.zeros(7)
if last_meas is None:
    continue
_, tcp_rot = panda_tcp_pose_numpy(q_current)
closing_axis = tcp_rot[:, 1]
elapsed = step * config["dt"] - last_meas_t
predicted = last_meas + obj_vel * (elapsed + _CLOSE_LEAD_S)
offset = float(np.dot(predicted - ee_pos, closing_axis))
crossed = prev_offset is not None and (offset == 0.0 or (prev_offset < 0.0) != (offset < 0.0))
prev_offset = offset
if not crossed:
    continue
```

The existing grasp-commit body (set gripper, `stop_conveyor_object`,
`grasp_error`, settle, lift, `is_grasped`, result dict) then runs
**unchanged** after this point.

`panda_tcp_pose_numpy` must be added to the existing
`from control.panda_kinematics import (...)`. `solve_intercept`,
`GraspExecutor`, `EE_MAX_SPEED`, and `_CLOSE_RANGE_M` become unused —
remove those imports/constants and the now-dead `grasp_executor`
construction, but leave `planning/intercept.py` and `manipulation/grasp.py`
themselves untouched (other callers/tests use them).

- [ ] **Step 3: Run the acceptance test**

```bash
uv run python run_conveyor_demo.py
uv run pytest -v
```

Expected (design spec §4): `contact_verified: True`, `grasp_error_m` ≈
0.013, `object_peak_height_gain_m` ≈ 0.09, and
`test_conveyor_episode_grasps_within_tolerance` **passing** — the first
time since Round 4.

If it does not pass, report the real numbers and stop; do not modify the
test or loosen its assertions.

- [ ] **Step 4: Update the module docstring**

Add a numbered point 12 to `run_conveyor_demo.py`'s module docstring,
matching the existing style, recording: pursuit never converged (82% of the
closing-axis error was the arm, measured), the rendezvous phase machine
that replaced it, and the resulting before/after numbers from §4 of the
design spec.

- [ ] **Step 5: Verify and commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add run_conveyor_demo.py
git commit -m "feat: rendezvous approach -- the arm now actually picks up the cube"
```
