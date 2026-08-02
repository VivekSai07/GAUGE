# YOLO Detector Precision Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine, with a real measured comparison, whether a YOLO detector
localizes the conveyor cube more precisely than the current RGB-threshold
segmentation — a cheap go/no-go gate before considering any change to the
real pipeline.

**Architecture:** Three standalone scripts under `experiments/yolo_precision/`
(matching the project's existing `experiments/` convention): generate an
exactly-labeled synthetic dataset from MuJoCo's own ground truth, fine-tune a
small pretrained YOLO model on it, then run both the new detector and the
existing `perception/segment.py` centroid through the same downstream 3D
back-projection math on a fresh held-out batch and compare against exact
ground truth.

**Tech Stack:** Python 3.11, `mujoco`, `opencv-python` (already a project
dependency), `ultralytics` (new, isolated dependency group — pulls `torch`
with CUDA support), `uv`.

## Global Constraints

- No change to `perception/`, `tracking/`, `prediction/`, `planning/`,
  `control/`, `manipulation/`, `sim/`, or `run_conveyor_demo.py` — this plan
  is entirely additive under `experiments/yolo_precision/`.
- `ultralytics` is added as its own `[dependency-groups]` entry in
  `pyproject.toml`, not to `[project.dependencies]` or the default `dev`
  group — running the main pipeline or test suite must not require it.
- Generated data (images, labels, trained weights, training run artifacts)
  lives under `experiments/yolo_precision/data/` and is gitignored — it is
  regenerable, not source.
- Reuse existing project code where it exists: `sim.conveyor_scene.
  ConveyorSceneEnv`, `sim.conveyor_scene.OBJECT_HALF_HEIGHT_M`,
  `perception.segment.segment_object_centroid`, `perception.camera.
  CameraIntrinsics`, `run_conveyor_demo._camera_point_to_world`. Do not
  reimplement any of these.
- This is experiment/evaluation tooling, not production pipeline code — no
  `pytest` unit tests are required (per the design spec's Section 6);
  verification is visual (preview images with drawn boxes) and printed
  metrics (training mAP, evaluation error numbers), as specified per task
  below.

---

## File Structure

```
pyproject.toml                          # + yolo-precision dependency group
.gitignore                              # + experiments/yolo_precision/data/
experiments/yolo_precision/
├── generate_dataset.py                 # synthetic frames + exact labels + preview
├── train.py                             # fine-tune yolo11n.pt
├── evaluate.py                           # go/no-go comparison
└── data/                                 # gitignored: dataset, weights, runs
```

---

### Task 1: Synthetic dataset generation with exact auto-labels

**Files:**
- Modify: `pyproject.toml` (add `yolo-precision` dependency group)
- Modify: `.gitignore` (add `experiments/yolo_precision/data/`)
- Create: `experiments/yolo_precision/generate_dataset.py`

**Interfaces:**
- Produces (used by Task 3): `sample_frame(env, rng, x_range, y_range, z,
  width, height) -> dict | None` with keys `"rgb"` (`np.ndarray`, HxWx3
  uint8), `"bbox_px"` (`tuple[float, float, float, float]`, pixel
  `(x_min, y_min, x_max, y_max)`), `"world_pos"` (`np.ndarray`, shape
  `(3,)`, the cube's true world-frame center) — importable from
  `experiments.yolo_precision.generate_dataset` when run from the repo
  root, or via a bare `from generate_dataset import sample_frame` when
  `evaluate.py` is run directly (both scripts live in the same directory,
  so a bare sibling import works without an `__init__.py`).
- Produces on disk: `experiments/yolo_precision/data/dataset/{train,val}/
  images/*.png`, `.../labels/*.txt` (YOLO format), `.../data.yaml`.

- [ ] **Step 1: Add the `yolo-precision` dependency group**

Edit `pyproject.toml`, adding a new group after the existing `dev` group
(do not modify `dev` or `[project.dependencies]`):

```toml
[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-cov>=7.1.0",
    "ruff>=0.16.1",
]
yolo-precision = [
    "ultralytics>=8.3.0",
]
```

- [ ] **Step 2: Install the group and verify CUDA is visible**

```bash
uv sync --group yolo-precision
uv run --group yolo-precision python -c "import torch; print('cuda available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

Expected output includes `cuda available: True` and the GPU name (an
NVIDIA RTX PRO 2000 Blackwell laptop GPU is expected on this machine). If
`False`, STOP and report — do not proceed with CPU-only training silently,
since the plan's time expectations assume GPU.

- [ ] **Step 3: Gitignore the data directory**

Append to `.gitignore`:

```
experiments/yolo_precision/data/
```

- [ ] **Step 4: Write `generate_dataset.py`**

Create `experiments/yolo_precision/generate_dataset.py`:

```python
"""Synthetic dataset generation for the YOLO detector precision experiment.

Renders wrist-camera frames from ConveyorSceneEnv with the cube at
randomized positions/orientations spanning its real operating envelope,
and computes EXACT 2D bounding-box labels by projecting the cube's known
8 corners through the camera's true intrinsics/extrinsics -- no manual
annotation, no annotation noise, per the design spec
(docs/superpowers/specs/2026-08-02-yolo-detector-precision-validation-design.md).

Run: uv run --group yolo-precision python experiments/yolo_precision/generate_dataset.py
"""
import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np

from sim.conveyor_scene import ConveyorSceneEnv, OBJECT_HALF_HEIGHT_M

_DATA_DIR = Path(__file__).parent / "data"
_DATASET_DIR = _DATA_DIR / "dataset"

# Realistic operating envelope: the object's real travel range along the
# conveyor's Y axis (starts at y=-0.3, travels at 0.08 m/s, typically
# grasped by ~step 2025 at dt=0.002 -- i.e. y up to roughly -0.3+0.32=0.02,
# padded slightly), X held near the conveyor centerline with a little
# jitter (the object's real X velocity is 0 in configs/conveyor.yaml), Z at
# its resting height on the platform (sim/conveyor_scene.py's platform
# geometry puts a resting box's center at z=0.05).
_X_RANGE = (0.49, 0.51)
_Y_RANGE = (-0.35, 0.05)
_Z = 0.05


def _cube_corners_world(center: np.ndarray, yaw: float) -> np.ndarray:
    """8 corners of the cube in world frame, given its center and a
    rotation about world Z (a fresh-placed/reset object can land at any
    yaw; this is not the tilt-buildup bug tracked separately in the design
    spec's Round 4 -- pure yaw variety is realistic and desired here)."""
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    local_corners = signs * OBJECT_HALF_HEIGHT_M
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return center + local_corners @ rot.T


def _world_to_pixel(point_world, cam_pos, cam_mat, fx, fy, cx, cy):
    """Inverse of run_conveyor_demo.py's _camera_point_to_world: world point
    -> pinhole-camera-frame point (x=right, y=down, z=forward-depth) ->
    pixel (u, v). Returns None if the point is behind the camera plane."""
    local = cam_mat.T @ (point_world - cam_pos)
    point_cam = np.array([local[0], -local[1], -local[2]])
    if point_cam[2] <= 1e-6:
        return None
    u = fx * (point_cam[0] / point_cam[2]) + cx
    v = fy * (point_cam[1] / point_cam[2]) + cy
    return u, v


def sample_frame(env: ConveyorSceneEnv, rng: np.random.Generator, width: int = 64, height: int = 64) -> dict | None:
    """Place the cube at a random pose, render, and compute its exact pixel
    bounding box. Returns None if the box degenerates (fully clipped or
    behind the camera) -- callers should retry with a new sample."""
    obj_jid = env.model.body("conveyor_object").jntadr[0]
    obj_qpos_addr = env.model.jnt_qposadr[obj_jid]

    x = rng.uniform(*_X_RANGE)
    y = rng.uniform(*_Y_RANGE)
    yaw = rng.uniform(0.0, 2 * np.pi)
    center = np.array([x, y, _Z])

    env.data.qpos[obj_qpos_addr : obj_qpos_addr + 3] = center
    half_yaw = yaw / 2.0
    env.data.qpos[obj_qpos_addr + 3 : obj_qpos_addr + 7] = [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)]
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    rgb, _ = env.get_rgbd(width, height)
    cam_id = env.model.camera("wrist_cam").id
    cam_pos = env.data.cam_xpos[cam_id].copy()
    cam_mat = env.data.cam_xmat[cam_id].reshape(3, 3).copy()
    fx, fy, cx, cy = env.camera_intrinsics(width, height)

    corners = _cube_corners_world(center, yaw)
    pixels = [_world_to_pixel(c, cam_pos, cam_mat, fx, fy, cx, cy) for c in corners]
    pixels = [p for p in pixels if p is not None]
    if len(pixels) < 8:
        return None

    us = np.clip([p[0] for p in pixels], 0, width - 1)
    vs = np.clip([p[1] for p in pixels], 0, height - 1)
    x_min, x_max = float(us.min()), float(us.max())
    y_min, y_max = float(vs.min()), float(vs.max())
    if x_max <= x_min or y_max <= y_min:
        return None

    return {"rgb": rgb, "bbox_px": (x_min, y_min, x_max, y_max), "world_pos": center.copy()}


def _write_yolo_label(path: Path, bbox_px, width: int, height: int) -> None:
    x_min, y_min, x_max, y_max = bbox_px
    x_center = (x_min + x_max) / 2.0 / width
    y_center = (y_min + y_max) / 2.0 / height
    box_w = (x_max - x_min) / width
    box_h = (y_max - y_min) / height
    path.write_text(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")


def _generate_split(env, rng, split: str, count: int, width: int, height: int) -> None:
    images_dir = _DATASET_DIR / split / "images"
    labels_dir = _DATASET_DIR / split / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    attempts = 0
    while written < count:
        attempts += 1
        if attempts > count * 10:
            raise RuntimeError(f"Too many degenerate samples generating '{split}' split -- check camera/range setup.")
        sample = sample_frame(env, rng, width, height)
        if sample is None:
            continue
        name = f"{split}_{written:05d}"
        bgr = cv2.cvtColor(sample["rgb"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(images_dir / f"{name}.png"), bgr)
        _write_yolo_label(labels_dir / f"{name}.txt", sample["bbox_px"], width, height)
        written += 1


def _write_data_yaml(width: int, height: int) -> None:
    yaml_text = (
        f"path: {_DATASET_DIR.resolve()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "names:\n"
        "  0: cube\n"
    )
    (_DATASET_DIR / "data.yaml").write_text(yaml_text)


def _write_preview(env, rng, count: int, width: int, height: int) -> None:
    """Draw the exact label boxes on a handful of fresh samples so the
    labeling itself can be visually spot-checked (design spec Section 6)."""
    preview_dir = _DATA_DIR / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    attempts = 0
    while written < count:
        attempts += 1
        if attempts > count * 10:
            break
        sample = sample_frame(env, rng, width, height)
        if sample is None:
            continue
        bgr = cv2.cvtColor(sample["rgb"], cv2.COLOR_RGB2BGR)
        scaled = cv2.resize(bgr, (width * 6, height * 6), interpolation=cv2.INTER_NEAREST)
        x_min, y_min, x_max, y_max = sample["bbox_px"]
        cv2.rectangle(
            scaled,
            (int(x_min * 6), int(y_min * 6)),
            (int(x_max * 6), int(y_max * 6)),
            (0, 255, 0),
            2,
        )
        cv2.imwrite(str(preview_dir / f"preview_{written:03d}.png"), scaled)
        written += 1
    print(f"Wrote {written} preview images to {preview_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-count", type=int, default=800)
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview-count", type=int, default=10)
    args = parser.parse_args()

    env = ConveyorSceneEnv(conveyor_velocity=np.array([0.0, 0.08, 0.0]))
    env.reset()
    rng = np.random.default_rng(args.seed)

    print(f"Generating {args.train_count} train + {args.val_count} val frames at {args.width}x{args.height}...")
    _generate_split(env, rng, "train", args.train_count, args.width, args.height)
    _generate_split(env, rng, "val", args.val_count, args.width, args.height)
    _write_data_yaml(args.width, args.height)
    print(f"Dataset written to {_DATASET_DIR}")

    if args.preview_count > 0:
        _write_preview(env, rng, args.preview_count, args.width, args.height)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run it and visually verify the labels**

```bash
uv run --group yolo-precision python experiments/yolo_precision/generate_dataset.py
```

Expected: prints "Dataset written to ..." and "Wrote 10 preview images to
...". Then open several files in
`experiments/yolo_precision/data/preview/preview_*.png` (e.g. via the
Read tool, which can view images) and confirm the green box tightly
outlines the red cube in each one, at varied positions and rotations. If
any box is clearly offset from the cube or missing it, STOP — the
projection math has a bug; do not proceed to training on bad labels.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore experiments/yolo_precision/generate_dataset.py
git commit -m "feat: synthetic YOLO dataset generation with exact auto-labels"
```

(`experiments/yolo_precision/data/` is gitignored and will not be staged.)

---

### Task 2: Fine-tune YOLO on the synthetic dataset

**Files:**
- Create: `experiments/yolo_precision/train.py`

**Interfaces:**
- Consumes: `experiments/yolo_precision/data/dataset/data.yaml` (Task 1's
  output).
- Produces (used by Task 3): `experiments/yolo_precision/data/
  cube_detector.pt` — a trained Ultralytics YOLO checkpoint loadable via
  `ultralytics.YOLO("experiments/yolo_precision/data/cube_detector.pt")`.

- [ ] **Step 1: Write `train.py`**

Create `experiments/yolo_precision/train.py`:

```python
"""Fine-tune a small pretrained YOLO model on the synthetic cube dataset.

Run: uv run --group yolo-precision python experiments/yolo_precision/train.py
Requires experiments/yolo_precision/generate_dataset.py to have been run
first (produces data/dataset/data.yaml).
"""
import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

_DATA_DIR = Path(__file__).parent / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    data_yaml = _DATA_DIR / "dataset" / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} not found -- run generate_dataset.py first."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available -- this experiment expects the RTX PRO 2000 "
            "GPU. Aborting rather than silently training on CPU."
        )

    model = YOLO("yolo11n.pt")
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        project=str(_DATA_DIR / "runs"),
        name="cube_detector",
        exist_ok=True,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    dest = _DATA_DIR / "cube_detector.pt"
    shutil.copy(best_weights, dest)

    metrics = model.val(data=str(data_yaml))
    print(f"\nFinal validation mAP50: {metrics.box.map50:.4f}")
    print(f"Final validation mAP50-95: {metrics.box.map:.4f}")
    print(f"Weights saved to {dest}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run training**

```bash
uv run --group yolo-precision python experiments/yolo_precision/train.py
```

Expected: Ultralytics' training progress bars run for 50 epochs, then
prints `Final validation mAP50: <value>` and `Weights saved to
experiments/yolo_precision/data/cube_detector.pt`. mAP50 should be high
(>0.9) — this is a single-class, low-visual-variance object; if it's
notably lower, note it but proceed to Task 3 (the evaluation's own 3D
error numbers are the real signal, not mAP in isolation).

- [ ] **Step 3: Commit**

```bash
git add experiments/yolo_precision/train.py
git commit -m "feat: YOLO fine-tuning script for the cube detector"
```

---

### Task 3: Side-by-side precision evaluation and go/no-go verdict

**Files:**
- Create: `experiments/yolo_precision/evaluate.py`

**Interfaces:**
- Consumes: `sample_frame` from `generate_dataset.py` (Task 1),
  `experiments/yolo_precision/data/cube_detector.pt` (Task 2's output),
  `perception.segment.segment_object_centroid`, `perception.camera.
  CameraIntrinsics`, `run_conveyor_demo._camera_point_to_world`,
  `sim.conveyor_scene.OBJECT_HALF_HEIGHT_M`.
- Produces: printed go/no-go report; `experiments/yolo_precision/data/
  eval_samples/*.png` (side-by-side visual comparison frames).

- [ ] **Step 1: Write `evaluate.py`**

Create `experiments/yolo_precision/evaluate.py`:

```python
"""Go/no-go comparison: does the trained YOLO detector localize the cube
more precisely than the current RGB-threshold segmentation?

Run: uv run --group yolo-precision python experiments/yolo_precision/evaluate.py
Requires train.py to have produced data/cube_detector.pt.

Success criteria (design spec Section 4): candidate mean 3D error must be
both >=40% lower than baseline AND under 1cm absolute to justify a Phase 2
pipeline-integration spec.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from generate_dataset import sample_frame
from perception.camera import CameraIntrinsics
from perception.segment import segment_object_centroid
from run_conveyor_demo import _camera_point_to_world
from sim.conveyor_scene import ConveyorSceneEnv, OBJECT_HALF_HEIGHT_M

_DATA_DIR = Path(__file__).parent / "data"
_COLOR_LOWER = (150, 0, 0)
_COLOR_UPPER = (255, 80, 80)


def _yolo_bbox_center_to_3d(model, rgb, depth, intrinsics, cam_pos, cam_mat):
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
    region = depth[y0:y1, x0:x1]
    z = float(region.mean()) + OBJECT_HALF_HEIGHT_M
    point_cam = intrinsics.deproject(u, v, z)
    return _camera_point_to_world(point_cam, cam_pos, cam_mat), (x_min, y_min, x_max, y_max)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--seed", type=int, default=999)  # distinct from generate_dataset's default train/val seed
    parser.add_argument("--save-samples", type=int, default=10)
    args = parser.parse_args()

    weights = _DATA_DIR / "cube_detector.pt"
    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found -- run train.py first.")
    model = YOLO(str(weights))

    env = ConveyorSceneEnv(conveyor_velocity=np.array([0.0, 0.08, 0.0]))
    env.reset()
    rng = np.random.default_rng(args.seed)
    cam_id = env.model.camera("wrist_cam").id
    fx, fy, cx, cy = env.camera_intrinsics(args.width, args.height)
    intrinsics = CameraIntrinsics(fx, fy, cx, cy)

    baseline_errors, candidate_errors = [], []
    baseline_misses, candidate_misses = 0, 0
    samples_dir = _DATA_DIR / "eval_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    evaluated = 0
    attempts = 0
    while evaluated < args.count:
        attempts += 1
        if attempts > args.count * 10:
            raise RuntimeError("Too many degenerate samples during evaluation -- check camera/range setup.")
        sample = sample_frame(env, rng, args.width, args.height)
        if sample is None:
            continue
        evaluated += 1

        rgb = sample["rgb"]
        _, depth = env.get_rgbd(args.width, args.height)
        cam_pos = env.data.cam_xpos[cam_id].copy()
        cam_mat = env.data.cam_xmat[cam_id].reshape(3, 3).copy()
        truth = sample["world_pos"]

        baseline_point_cam = segment_object_centroid(
            rgb, depth, intrinsics, _COLOR_LOWER, _COLOR_UPPER, depth_bias=OBJECT_HALF_HEIGHT_M
        )
        if baseline_point_cam is None:
            baseline_misses += 1
            baseline_world = None
        else:
            baseline_world = _camera_point_to_world(baseline_point_cam, cam_pos, cam_mat)
            baseline_errors.append(float(np.linalg.norm(baseline_world - truth)))

        yolo_result = _yolo_bbox_center_to_3d(model, rgb, depth, intrinsics, cam_pos, cam_mat)
        if yolo_result is None:
            candidate_misses += 1
            candidate_world, candidate_box = None, None
        else:
            candidate_world, candidate_box = yolo_result
            candidate_errors.append(float(np.linalg.norm(candidate_world - truth)))

        if saved < args.save_samples:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            scaled = cv2.resize(bgr, (args.width * 6, args.height * 6), interpolation=cv2.INTER_NEAREST)
            if candidate_box is not None:
                x_min, y_min, x_max, y_max = candidate_box
                cv2.rectangle(scaled, (int(x_min * 6), int(y_min * 6)), (int(x_max * 6), int(y_max * 6)), (0, 255, 0), 2)
            cv2.putText(
                scaled,
                f"base_err={baseline_errors[-1]:.4f}" if baseline_world is not None else "base_err=MISS",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
            )
            cv2.putText(
                scaled,
                f"yolo_err={candidate_errors[-1]:.4f}" if candidate_world is not None else "yolo_err=MISS",
                (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
            )
            cv2.imwrite(str(samples_dir / f"eval_{saved:03d}.png"), scaled)
            saved += 1

    baseline_mean = float(np.mean(baseline_errors)) if baseline_errors else float("nan")
    baseline_max = float(np.max(baseline_errors)) if baseline_errors else float("nan")
    candidate_mean = float(np.mean(candidate_errors)) if candidate_errors else float("nan")
    candidate_max = float(np.max(candidate_errors)) if candidate_errors else float("nan")

    print(f"\n=== Evaluated {evaluated} frames ===")
    print(f"Baseline (RGB threshold): mean_err={baseline_mean:.4f}m max_err={baseline_max:.4f}m misses={baseline_misses}")
    print(f"Candidate (YOLO):         mean_err={candidate_mean:.4f}m max_err={candidate_max:.4f}m misses={candidate_misses}")

    if baseline_mean > 0 and not np.isnan(candidate_mean):
        improvement = 1.0 - (candidate_mean / baseline_mean)
        print(f"\nImprovement: {improvement * 100:.1f}%")
        go = improvement >= 0.40 and candidate_mean < 0.01
        print(f"GO/NO-GO (>=40% improvement AND <1cm absolute): {'GO' if go else 'NO-GO'}")
    else:
        print("\nCould not compute a verdict (insufficient successful detections).")

    print(f"\nSample comparison frames written to {samples_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the evaluation**

```bash
uv run --group yolo-precision python experiments/yolo_precision/evaluate.py
```

Expected: prints the evaluated frame count, both pipelines' mean/max error
and miss count, the improvement percentage, and a `GO`/`NO-GO` verdict.

- [ ] **Step 3: Visually verify a handful of sample frames**

Open several files in `experiments/yolo_precision/data/eval_samples/
eval_*.png` (via the Read tool) and confirm the drawn YOLO box actually
sits on the cube and the printed per-frame errors look consistent with
what's visible — don't trust the aggregate numbers without this check
(design spec Section 6).

- [ ] **Step 4: Commit**

```bash
git add experiments/yolo_precision/evaluate.py
git commit -m "feat: YOLO vs. RGB-threshold precision comparison and go/no-go report"
```

- [ ] **Step 5: Report the verdict**

Report the printed GO/NO-GO result, the improvement percentage, and both
miss counts back to the user, along with 2-3 of the saved sample
comparison images. This is the plan's actual deliverable — the decision
of whether to write a Phase 2 integration spec depends entirely on this
number, not on anything assumed during planning.
