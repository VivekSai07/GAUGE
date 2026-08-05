# YOLO Perception Integration — Design Spec

## 1. Purpose & Scope

Round 4 (`docs/superpowers/specs/2026-07-29-dynamic-object-tracking-manipulation-design.md`,
Section 12) established that the grasp-survival gap is a targeting-precision
problem: the gripper needs ~3cm accuracy along its closing axis, and the
current RGB color-threshold segmentation's per-measurement noise sits right
at that floor. A follow-up experiment
(`docs/superpowers/specs/2026-08-02-yolo-detector-precision-validation-design.md`)
validated that a fine-tuned YOLO detector, combined with the existing
color-threshold logic for depth-gating, cuts that noise by 43.8% (mean 3D
error 1.08cm → 0.61cm) — with a documented tail caveat (worst-case error
went from 1.66cm to 2.75cm) that a follow-up investigation
([issue #32](https://github.com/VivekSai07/GAUGE/issues/32)) traced
entirely to frame-edge clipping, not rotation or lighting, and judged
unlikely to matter at real grasp-commit geometry (where the arm is actively
centering the object) — though that claim is not yet confirmed against real
episode telemetry.

This spec covers wiring that validated hybrid detector into the real
closed-loop pipeline, replacing color-threshold segmentation as the
**default** perception method (a deliberate choice — see Section 4), and
determining honestly whether it actually closes the Round 4 gap:
does `contact_verified` finally read `True` for a real, lift-verified grasp?

**This is a genuine experiment, not a guaranteed fix.** The YOLO precision
result was measured on isolated 3D localization accuracy, uniformly sampled
across the object's full operating envelope — not on the specific
distribution of frames a real closed-loop episode actually produces at the
moment of grasp-commit. Wiring it in and running the real episode is the
only way to find out. If `contact_verified` still reads `False` afterward,
that is a real, reportable result — not something to explain away.

## 2. Background

- Validated hybrid approach (`experiments/yolo_precision/evaluate.py`,
  commit `567297c`): `model.predict()` on the rendered RGB frame → highest-
  confidence bounding box → pixel center `(u, v)` → depth averaged over only
  the pixels *inside that box* that also match the existing color threshold
  → zero color-matched pixels = a genuine miss (`None`), not a fallback
  estimate. This exact zero-fallback-miss semantic is what took the result
  from NO-GO to GO (`docs/superpowers/specs/2026-08-02-yolo-detector-precision-validation-design.md`,
  Section 7) — it must be preserved exactly, not simplified back toward the
  earlier full-box-mean version that was measured worse.
- Trained checkpoint: `experiments/yolo_precision/data/cube_detector.pt`
  (gitignored), `yolo11n.pt` fine-tuned 50 epochs, mAP50=0.9949,
  mAP50-95=0.97713.
- Current call site: `run_conveyor_demo.py::run_one_episode`, one call to
  `perception.segment.segment_object_centroid(rgb, depth, intrinsics,
  tuple(cam_cfg["color_lower"]), tuple(cam_cfg["color_upper"]),
  depth_bias=OBJECT_HALF_HEIGHT_M)` per control tick, returning a camera-
  frame 3D point or `None`, immediately transformed to world frame via
  `_camera_point_to_world` and fed to the Kalman filter / track.
- `CameraIntrinsics.deproject(u, v, depth) -> np.ndarray` (`perception/camera.py`)
  is the shared pinhole back-projection both the classical and YOLO paths
  use — already reused as-is by the experiment, not reimplemented.

## 3. Architecture

New module `perception/yolo_segment.py`, matching `perception/segment.py`'s
existing shape (a small, focused file with one clear responsibility) rather
than modifying `segment_object_centroid` in place — `perception/segment.py`
stays the working, tested, classical baseline; nothing about its docstring
("Classical color+depth segmentation to a 3D centroid") becomes untrue.

```python
def yolo_centroid(
    rgb: np.ndarray,
    depth: np.ndarray,
    model,  # ultralytics.YOLO, loaded once by the caller
    intrinsics: CameraIntrinsics,
    color_lower: tuple[int, int, int],
    color_upper: tuple[int, int, int],
    depth_bias: float = 0.0,
) -> np.ndarray | None:
```

Internals: `model.predict(source=rgb, verbose=False)[0]` → if no boxes,
return `None` → highest-confidence box → `(u, v)` = box center → clip the
box to the depth array's bounds → color-mask the RGB pixels inside that
clipped region using `color_lower`/`color_upper` → if zero pixels match,
return `None` → else `z = masked_depth.mean() + depth_bias` →
`intrinsics.deproject(u, v, z)`. This is a direct, unmodified port of
`experiments/yolo_precision/evaluate.py`'s `_yolo_bbox_center_to_3d`, with
one interface difference: it takes `color_lower`/`color_upper` as
parameters (like `segment_object_centroid` already does) instead of the
experiment's hardcoded module constants, so the real pipeline's single
source of truth stays `configs/conveyor.yaml`'s existing `camera.
color_lower`/`camera.color_upper` keys — no new config keys, no drift
between the validated experiment's thresholds and the real pipeline's.

**Checkpoint:** move (not copy) `experiments/yolo_precision/data/
cube_detector.pt` → `perception/models/cube_detector.pt`, git-tracked.
`yolo_segment.py` loads it via a module-level path constant, same pattern
`sim/conveyor_scene.py` already uses for its own asset paths
(`_MENAGERIE_DIR = Path(__file__).parent / "assets/menagerie/..."`).

**`run_conveyor_demo.py`:** one call-site change. Load the model once
during episode setup (same place `mpc`/`grasp_executor`/`fk` are already
constructed once, not per-tick):
```python
from ultralytics import YOLO
from perception.yolo_segment import yolo_centroid
...
detector = YOLO(str(_MODEL_PATH))
```
Then replace the existing `segment_object_centroid(...)` call with
`yolo_centroid(rgb, depth, detector, intrinsics, tuple(cam_cfg["color_lower"]),
tuple(cam_cfg["color_upper"]), depth_bias=OBJECT_HALF_HEIGHT_M)`. Return
type is identical (`np.ndarray | None`); no other code in the control loop
changes — the KF, `Track`, prediction, planning, and grasp-commit logic
already handle a `None` measurement (a miss) via existing gating/max-
consecutive-misses logic, unmodified since Round 1.

**Dependencies:** `ultralytics` moves from `[dependency-groups] yolo-precision`
into `[project.dependencies]` — the main pipeline now requires it to run at
all, so it belongs there, not in an opt-in group. Remove the now-redundant
`yolo-precision` group (its only member was `ultralytics`, now promoted);
`experiments/yolo_precision/train.py`/`generate_dataset.py` continue to
work unchanged since `ultralytics` is still installed, just via the main
dependency set instead of a separate group.

**Inference device:** no `device=0` override — `model.predict()` auto-
selects CUDA if available, CPU otherwise. A nano model's single-frame
inference is fast enough on CPU that the pipeline stays runnable without a
GPU (training remains the only step that meaningfully benefits from one,
and that's already a separate, one-time script). The only consequence of
slow inference on a given machine is `--render`'s real-time pacing drifting
behind wall-clock — cosmetic, not a correctness issue; the headless path
has no timing constraint at all.

## 4. Why default, not opt-in (explicit tradeoff)

The project's stated philosophy has been deliberately lightweight ("Pure
Python, no ROS2/C++/Pinocchio/acados," no GPU requirement). Making YOLO the
default perception method breaks that: `uv sync` now pulls `torch`, and the
pipeline requires a committed 5MB model file to run at all. This is a
conscious choice, not an oversight — the alternative (config-gated, color-
threshold still default) would leave the measurably-better perception path
un-exercised by default and by the test suite, understating what the
project actually demonstrates. The tradeoff is accepted; it is not
revisited by later tasks without a deliberate reason to.

## 5. Testing

- `tests/perception/test_yolo_segment.py` (new): integration-style, using
  the real committed checkpoint (not mocked) — render a synthetic frame via
  `ConveyorSceneEnv` with a known cube pose, confirm `yolo_centroid` returns
  a 3D point within a reasonable tolerance of ground truth, and confirm it
  returns `None` on a frame with no cube in view (matching
  `segment_object_centroid`'s existing miss-handling test pattern in
  `tests/perception/test_segment.py`, which stays unchanged).
- `tests/perception/test_segment.py`: unchanged — the classical baseline
  stays tested as-is.
- `tests/test_integration_conveyor.py::test_conveyor_episode_grasps_within_tolerance`
  is the real acceptance test, **already in the repo, already asserting
  `contact_verified is True`** — it has been failing since Round 4 for
  exactly this reason. No change to its assertions. If the integration
  works, this test goes green with zero modification, which is itself the
  proof. If it doesn't, that's reported honestly (README/design-spec
  update, matching every prior round's practice) and becomes the next
  root-causing thread — the assertion is never weakened to force a pass.
- Full `uv run pytest -v` must be run and its result reported verbatim
  either way.

## 6. Documentation

Whatever the outcome, update `README.md`'s "Demonstrated result" section
and add a new dated round to the main design spec's Section 12 history
(matching the established "Round N" narrative pattern), stating plainly
whether `contact_verified` now reads `True` for a real reason, and if not,
what was actually observed instead.
