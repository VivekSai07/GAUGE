# YOLO Perception Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace RGB color-threshold segmentation with the validated YOLO
hybrid detector as the real pipeline's default perception method, and
determine honestly whether `contact_verified` finally reads `True` for a
real, lift-verified grasp.

**Architecture:** A new `perception/yolo_segment.py` module ports the
already-validated hybrid detection (`experiments/yolo_precision/evaluate.py`'s
`_yolo_bbox_center_to_3d`) into a reusable function with the same interface
shape as the existing `segment_object_centroid`. `run_conveyor_demo.py`
swaps its one call site; nothing downstream (tracking, prediction,
planning, control) changes, since both return the same `np.ndarray | None`.

**Tech Stack:** Python 3.11, `mujoco`, `ultralytics` (promoted from an
experiment-only dependency group to a main dependency), `uv`.

## Global Constraints

- No change to `perception/segment.py`, `tracking/`, `prediction/`,
  `planning/`, `control/`, `manipulation/`, or `sim/` — this plan is
  additive (`perception/yolo_segment.py`) plus one call-site swap in
  `run_conveyor_demo.py`.
- `perception.segment.segment_object_centroid`'s color-mask + zero-match =
  miss semantics (`experiments/yolo_precision/evaluate.py` commit
  `567297c`'s `_yolo_bbox_center_to_3d`) must be preserved exactly in
  `yolo_centroid` — no fallback estimate on a zero-pixel color match; that
  exact behavior is what took the validation experiment from NO-GO to GO.
- `ultralytics` becomes a main dependency (`[project.dependencies]`), not
  an opt-in group — accepted tradeoff, not to be re-litigated by this plan
  (design spec Section 4).
- Inference must not hardcode `device=0` — auto-select (CUDA if available,
  CPU otherwise) so the pipeline stays runnable without a GPU.
- The success criterion is the **existing, unmodified** assertion in
  `tests/test_integration_conveyor.py::test_conveyor_episode_grasps_within_tolerance`
  (`assert result["contact_verified"] is True`) — never weakened to force
  a pass. Report the real result either way.

---

## File Structure

```
pyproject.toml                          # ultralytics -> main deps, drop yolo-precision group
.gitignore                              # *.pt -> /*.pt (stop blocking the tracked checkpoint)
perception/
├── models/
│   └── cube_detector.pt                # moved from experiments/yolo_precision/data/, git-tracked
├── segment.py                          # unchanged
└── yolo_segment.py                     # new: yolo_centroid()
tests/perception/
└── test_yolo_segment.py                # new
run_conveyor_demo.py                    # one call-site swap
README.md                               # updated with the real outcome
docs/superpowers/specs/2026-07-29-dynamic-object-tracking-manipulation-design.md  # new dated round
```

---

### Task 1: Promote the dependency, relocate the checkpoint, fix gitignore scoping

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Modify: `experiments/yolo_precision/train.py` (docstring note only)
- Move: `experiments/yolo_precision/data/cube_detector.pt` → `perception/models/cube_detector.pt`

**Interfaces:** none (no application code yet) — produces the git-tracked
checkpoint file and the `ultralytics` package as an importable main
dependency for Task 2.

- [ ] **Step 1: Promote `ultralytics` to main dependencies**

Edit `pyproject.toml`. Remove the entire `yolo-precision` dependency group
(lines currently reading `yolo-precision = [...]`), and add `ultralytics`
to `[project.dependencies]`:

```toml
[project]
dependencies = [
    "mujoco>=3.2.0",
    "casadi>=3.6.5",
    "numpy>=1.26",
    "pyyaml>=6.0",
    "defusedxml>=0.7.1",
    "opencv-python>=5.0.0.93",
    "ultralytics>=8.3.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-cov>=7.1.0",
    "ruff>=0.16.1",
]
```

Also remove the `[tool.uv.sources]` (`torch`/`torchvision` → `pytorch-cu130`)
and `[[tool.uv.index]]` (`pytorch-cu130`) blocks entirely. They forced a
CUDA-specific torch build for every install, which is right for a one-machine
training script but wrong for a main dependency anyone might install
without a matching GPU — plain PyPI resolution (CPU-capable by default) is
what makes the pipeline installable without a GPU, matching this plan's
Global Constraints.

- [ ] **Step 2: Note the training-time consequence in `train.py`**

`experiments/yolo_precision/train.py` still needs a CUDA-enabled torch
build to actually use the GPU for training (its `model.train(..., device=0)`
call requires it). Since Step 1 removed the project-wide CUDA index
override, retraining after this change needs that build installed
manually. Add this note to the top of `experiments/yolo_precision/train.py`'s
module docstring (after the existing `Requires ...` line):

```
Note: as of the YOLO perception integration, `ultralytics`/`torch` are
main project dependencies (plain PyPI resolution, CPU-capable by
default) rather than a GPU-pinned experiment-only group -- retraining
with this script's `device=0` requires a CUDA-enabled torch build
installed manually first, e.g.:
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
(adjust the cu-version to match your own GPU driver).
```

- [ ] **Step 3: Fix `.gitignore` so the checkpoint can be tracked**

Edit `.gitignore`. Change the bare `*.pt` line to `/*.pt` (anchors the
pattern to the repo root only, matching its original intent — catching
stray Ultralytics pretrained-checkpoint downloads that land in the current
working directory — without blocking an intentionally-tracked model
checkpoint anywhere else in the tree):

```
.venv/
__pycache__/
*.pyc
logs/
.coverage
experiments/yolo_precision/data/
# Ultralytics downloads pretrained checkpoints and writes training-run
# artifacts to the current working directory by default, which can land
# outside experiments/yolo_precision/data/ depending on how a script is invoked.
/*.pt
runs/
```

- [ ] **Step 4: Move the checkpoint and install the updated dependencies**

```bash
mkdir -p perception/models
git mv experiments/yolo_precision/data/cube_detector.pt perception/models/cube_detector.pt
uv sync
```

If `git mv` fails because the file isn't tracked yet (it's currently
gitignored, so this is expected), use `mv` instead and let the later
`git add` in Step 5 pick it up:

```bash
mv experiments/yolo_precision/data/cube_detector.pt perception/models/cube_detector.pt
```

- [ ] **Step 5: Verify and commit**

```bash
uv run pytest -v
```

Expected: 52 passed, 1 pre-existing failure
(`test_conveyor_episode_grasps_within_tolerance` — unrelated to this task,
still the known Round 4 limitation; Task 3 is what may finally change it).

```bash
git status
```

Confirm `perception/models/cube_detector.pt` shows as a new file about to
be added (not still ignored — if it still shows as ignored, the `.gitignore`
edit in Step 3 didn't take effect; re-check it before proceeding).

```bash
git add pyproject.toml uv.lock .gitignore experiments/yolo_precision/train.py perception/models/cube_detector.pt
git commit -m "feat: promote ultralytics to a main dependency, track the trained checkpoint"
```

---

### Task 2: `perception/yolo_segment.py` — the hybrid detector, TDD

**Files:**
- Create: `perception/yolo_segment.py`
- Create: `tests/perception/test_yolo_segment.py`

**Interfaces:**
- Consumes: `perception.camera.CameraIntrinsics` (existing, unchanged),
  `perception/models/cube_detector.pt` (Task 1's output), `sim.conveyor_scene.ConveyorSceneEnv`/`OBJECT_HALF_HEIGHT_M` (existing, for the test only).
- Produces (used by Task 3): `perception.yolo_segment.MODEL_PATH: Path`
  and `perception.yolo_segment.yolo_centroid(rgb: np.ndarray, depth:
  np.ndarray, model, intrinsics: CameraIntrinsics, color_lower: tuple[int,
  int, int], color_upper: tuple[int, int, int], depth_bias: float = 0.0)
  -> np.ndarray | None`. `model` is an already-constructed
  `ultralytics.YOLO` instance — this function never constructs or loads
  one itself, so callers control load timing (once per episode, not once
  per frame).

- [ ] **Step 1: Write the failing tests**

Create `tests/perception/test_yolo_segment.py`:

```python
import mujoco
import numpy as np

from perception.camera import CameraIntrinsics
from perception.yolo_segment import MODEL_PATH, yolo_centroid
from sim.conveyor_scene import OBJECT_HALF_HEIGHT_M, ConveyorSceneEnv


def _place_cube_in_view(env: ConveyorSceneEnv, x: float, y: float, z: float) -> None:
    """Teleport the conveyor object to a known pose within the same
    x/y/z envelope experiments/yolo_precision/generate_dataset.py trained
    on, and re-derive the physics state. No settling needed -- this is a
    single-frame render, not a dynamics test."""
    obj_jid = env.model.body("conveyor_object").jntadr[0]
    obj_qpos_addr = env.model.jnt_qposadr[obj_jid]
    env.data.qpos[obj_qpos_addr : obj_qpos_addr + 3] = [x, y, z]
    env.data.qpos[obj_qpos_addr + 3 : obj_qpos_addr + 7] = [1.0, 0.0, 0.0, 0.0]
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)


def test_yolo_centroid_finds_cube_near_ground_truth():
    from ultralytics import YOLO

    env = ConveyorSceneEnv(conveyor_velocity=np.array([0.0, 0.08, 0.0]))
    env.reset()
    _place_cube_in_view(env, x=0.5, y=-0.1, z=0.05)

    model = YOLO(str(MODEL_PATH))
    rgb, depth = env.get_rgbd(64, 64)
    fx, fy, cx, cy = env.camera_intrinsics(64, 64)
    intrinsics = CameraIntrinsics(fx, fy, cx, cy)

    result = yolo_centroid(
        rgb,
        depth,
        model,
        intrinsics,
        color_lower=(150, 0, 0),
        color_upper=(255, 80, 80),
        depth_bias=OBJECT_HALF_HEIGHT_M,
    )

    assert result is not None
    truth = env.get_object_ground_truth()
    np.testing.assert_allclose(result, truth, atol=0.05)


def test_yolo_centroid_returns_none_when_no_cube_in_view():
    from ultralytics import YOLO

    model = YOLO(str(MODEL_PATH))
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    depth = np.full((64, 64), 1.5, dtype=np.float32)
    intrinsics = CameraIntrinsics(fx=64.0, fy=64.0, cx=32.0, cy=32.0)

    result = yolo_centroid(
        rgb, depth, model, intrinsics, color_lower=(150, 0, 0), color_upper=(255, 80, 80)
    )
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/perception/test_yolo_segment.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'perception.yolo_segment'`.

- [ ] **Step 3: Implement `perception/yolo_segment.py`**

```python
"""YOLO-based 2D detection + color-masked depth sampling to a 3D centroid.

A direct port of experiments/yolo_precision/evaluate.py's
_yolo_bbox_center_to_3d (commit 567297c, the version that took the
validation experiment from NO-GO to GO -- see design spec
docs/superpowers/specs/2026-08-02-yolo-detector-precision-validation-design.md,
Section 7). The detector supplies the 2D pixel location (bounding box
center); the same color threshold perception/segment.py already uses
gates which pixels inside that box count for depth -- a zero-match is a
genuine miss (None), not a fallback estimate. That exact zero-fallback
semantic is what fixed the experiment's original full-box-mean approach,
and must not be simplified away here.
"""
from pathlib import Path

import numpy as np

from perception.camera import CameraIntrinsics

MODEL_PATH = Path(__file__).parent / "models" / "cube_detector.pt"


def yolo_centroid(
    rgb: np.ndarray,
    depth: np.ndarray,
    model,
    intrinsics: CameraIntrinsics,
    color_lower: tuple[int, int, int],
    color_upper: tuple[int, int, int],
    depth_bias: float = 0.0,
) -> np.ndarray | None:
    results = model.predict(source=rgb, verbose=False)[0]
    if len(results.boxes) == 0:
        return None
    confs = results.boxes.conf.cpu().numpy()
    best = int(np.argmax(confs))
    x_min, y_min, x_max, y_max = results.boxes.xyxy[best].cpu().numpy()
    u = (x_min + x_max) / 2.0
    v = (y_min + y_max) / 2.0
    y0, y1 = int(max(0, y_min)), int(min(depth.shape[0], y_max + 1))
    x0, x1 = int(max(0, x_min)), int(min(depth.shape[1], x_max + 1))
    region_rgb = rgb[y0:y1, x0:x1]
    region_depth = depth[y0:y1, x0:x1]
    lower = np.array(color_lower, dtype=np.uint8)
    upper = np.array(color_upper, dtype=np.uint8)
    mask = np.all((region_rgb >= lower) & (region_rgb <= upper), axis=-1)
    if mask.sum() == 0:
        return None
    z = float(region_depth[mask].mean()) + depth_bias
    return intrinsics.deproject(u, v, z)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/perception/test_yolo_segment.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add perception/yolo_segment.py tests/perception/test_yolo_segment.py
git commit -m "feat: add YOLO hybrid detector as a perception module"
```

---

### Task 3: Wire it into the real pipeline and report the actual result

**Files:**
- Modify: `run_conveyor_demo.py`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-07-29-dynamic-object-tracking-manipulation-design.md`

**Interfaces:**
- Consumes: `perception.yolo_segment.MODEL_PATH`,
  `perception.yolo_segment.yolo_centroid` (Task 2's output).

- [ ] **Step 1: Swap the import**

In `run_conveyor_demo.py`, line 174 currently reads:

```python
from perception.segment import segment_object_centroid
```

Replace it with:

```python
from perception.yolo_segment import MODEL_PATH, yolo_centroid
from ultralytics import YOLO
```

- [ ] **Step 2: Load the detector once per episode**

In `run_one_episode`, immediately after the existing `cam_id =
env.model.camera("wrist_cam").id` line (currently line 240), add:

```python
    detector = YOLO(str(MODEL_PATH))
```

- [ ] **Step 3: Replace the call site**

The existing block (currently lines 300-308):

```python
        rgb, depth = env.get_rgbd(cam_cfg["width"], cam_cfg["height"])
        measurement_cam = segment_object_centroid(
            rgb,
            depth,
            intrinsics,
            tuple(cam_cfg["color_lower"]),
            tuple(cam_cfg["color_upper"]),
            depth_bias=OBJECT_HALF_HEIGHT_M,
        )
```

becomes:

```python
        rgb, depth = env.get_rgbd(cam_cfg["width"], cam_cfg["height"])
        measurement_cam = yolo_centroid(
            rgb,
            depth,
            detector,
            intrinsics,
            tuple(cam_cfg["color_lower"]),
            tuple(cam_cfg["color_upper"]),
            depth_bias=OBJECT_HALF_HEIGHT_M,
        )
```

The `if measurement_cam is None: ... else: ...` block immediately after
(currently lines 309-314) is unchanged — both functions return the same
`np.ndarray | None`.

- [ ] **Step 4: Add a docstring point documenting the swap**

`run_conveyor_demo.py`'s module docstring numbers its deviations/changes
up to point 10 (ending around the current line 157, `"""`). Add point 11
immediately before the closing `"""`:

```
11. Perception swapped from perception.segment.segment_object_centroid
    (pure RGB color-threshold) to perception.yolo_segment.yolo_centroid
    (a YOLO-detected bounding box center, still using the same color
    threshold to gate which pixels inside that box count for depth) --
    see docs/superpowers/specs/2026-08-04-yolo-perception-integration-design.md.
    Validated in isolation to cut mean 3D localization error 43.8%
    (docs/superpowers/specs/2026-08-02-yolo-detector-precision-validation-design.md,
    Section 7) before being wired in here.
```

- [ ] **Step 5: Run the full suite and record the exact result**

```bash
uv run pytest -v
```

This is the actual point of the whole plan. Two possible outcomes, and
your remaining steps depend on which one you get — read both before running:

**Outcome A: `test_conveyor_episode_grasps_within_tolerance` now PASSES**
(53 passed, 0 failed). This means `contact_verified` is `True` for real —
the grasp survived a lift for the first time. Go to Step 6a.

**Outcome B: it still FAILS** (52 passed, 1 failed, same test as before).
This means the improved perception was not sufficient on its own. Go to
Step 6b. This is a real, valid, reportable result — do not treat it as a
task failure, and do not modify the test or its assertions to force a
pass.

Either way, also run:
```bash
uv run python run_conveyor_demo.py
```
and record the full printed result dict (`grasped`, `grasp_error_m`,
`contact_verified`, `object_height_gain_m`, `object_peak_height_gain_m`) —
you'll need the exact numbers for Step 6a or 6b.

- [ ] **Step 6a (Outcome A — it passed): document the win**

Update `README.md`'s "Demonstrated result" section (currently states
`contact_verified` reads `False`) to report the real outcome: state
plainly that `contact_verified` now reads `True`, cite the exact
`grasp_error_m` and `object_peak_height_gain_m` values from your run, and
name the YOLO perception swap as what changed. Add a new dated round to
`docs/superpowers/specs/2026-07-29-dynamic-object-tracking-manipulation-design.md`'s
Section 12 (after the existing "Round 4, continued" section), titled
"Round 5: perception precision closed the gap", describing what changed
and the exact before/after numbers. Then skip to Step 7.

- [ ] **Step 6b (Outcome B — it still failed): document the honest result**

Update `README.md`'s "Demonstrated result" section: keep it accurate —
`contact_verified` still reads `False`, but note that perception has been
upgraded to the validated YOLO hybrid (43.8% better isolated localization
accuracy) and this alone was not sufficient to close the gap; the real
run's exact `grasp_error_m`/`object_peak_height_gain_m` numbers go here
too, for comparison against the pre-integration figures already in the
file. Add a new dated round to
`docs/superpowers/specs/2026-07-29-dynamic-object-tracking-manipulation-design.md`'s
Section 12, titled "Round 5: better perception alone did not close the
gap", stating the result plainly and noting this as a genuine finding —
the remaining gap is now known not to be explained by the
color-threshold-segmentation noise floor identified in Round 4, which
narrows what's left to investigate (e.g., whether the KF/prediction
layer's own smoothing reintroduces lag that erodes the perception gain
before it reaches the grasp-commit decision — a real, specific next
hypothesis, not a vague "needs more work"). Then continue to Step 7.

- [ ] **Step 7: Commit**

```bash
git add run_conveyor_demo.py README.md docs/superpowers/specs/2026-07-29-dynamic-object-tracking-manipulation-design.md
git commit -m "feat: wire YOLO perception into the real pipeline"
```

Use a commit message matching whichever outcome actually happened —
`"feat: wire YOLO perception into the real pipeline -- grasp now survives a lift"`
for Outcome A, or
`"feat: wire YOLO perception into the real pipeline -- gap persists, narrows next hypothesis"`
for Outcome B. Either is a legitimate commit message for this project;
do not force Outcome A's wording onto Outcome B's result.
