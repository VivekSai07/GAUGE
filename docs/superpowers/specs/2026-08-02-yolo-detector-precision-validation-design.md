# YOLO Detector Precision Validation — Design Spec

## 1. Purpose & Scope

Round 4's root-cause investigation
(`docs/superpowers/specs/2026-07-29-dynamic-object-tracking-manipulation-design.md`,
Section 12) established, via five independent lines of evidence, that the
grasp-survival gap is a **targeting-precision** problem, not a
control-strategy one: the gripper needs ~3cm accuracy along its closing
axis, and the current pipeline's per-measurement 3D localization noise
(RGB color-threshold segmentation in `perception/segment.py`) sits right at
that floor.

This spec covers **one narrow, cheap experiment**: does a trained YOLO
object detector localize the cube more precisely, in 2D pixel space, than
the current color-threshold centroid? If yes by a meaningful margin,
integrating it into the real pipeline (a separate, larger, not-yet-planned
Phase 2) is worth doing. If no, we've spent an afternoon, not a redesign,
and the conclusion that segmentation accuracy is the bottleneck gets
sharpened rather than acted on blindly.

**Explicitly out of scope for this spec:** any change to
`perception/segment.py`, `tracking/`, or any other file in the main
pipeline. No `ultralytics`/`torch` dependency is added to the project's
main `pyproject.toml` dependencies — this experiment is self-contained
under `experiments/yolo_precision/` with its own dependency group, exactly
like `experiments/watch_conveyor_tracking.py` added `opencv-python` for
its own purposes without the main pipeline needing it.

## 2. Background

- Object: a 4cm cube (`OBJECT_HALF_HEIGHT_M = 0.02`, `sim/conveyor_scene.py`).
- Current segmentation: `perception/segment.py::segment_object_centroid` —
  RGB threshold mask, centroid of matching pixels, depth-based
  back-projection to a 3D camera-frame point via `perception/camera.py`.
  Known fragility already documented in
  `experiments/watch_conveyor_tracking.py`'s docstring: near-normal lighting
  incidence desaturates the object's rendered color, requiring a widened
  threshold window — a real, observed failure mode of pure RGB thresholding.
- Wrist camera: `wrist_cam`, mounted on the Panda's `hand` body,
  eye-in-hand (moves with the arm), `fovy=58`, historically rendered at
  64x64 in the closed-loop pipeline (`configs/conveyor.yaml`'s
  `camera.width/height`).
- Compute: NVIDIA RTX PRO 2000 Blackwell (laptop), 8GB VRAM, CUDA
  available.

## 3. Architecture

Three scripts under `experiments/yolo_precision/`, run in sequence, each
producing a persistent artifact the next one consumes (unlike
`experiments/watch_conveyor_tracking.py`'s single-file style, these three
phases have genuinely expensive, reusable intermediate outputs — a
dataset, a trained model — that shouldn't be regenerated on every run):

```
experiments/yolo_precision/
├── generate_dataset.py   # MuJoCo renders + exact auto-generated labels
├── train.py               # fine-tune yolo11n.pt on the generated set
├── evaluate.py             # side-by-side precision comparison, go/no-go
└── data/                   # generated dataset + trained weights (gitignored)
```

### 3.1 `generate_dataset.py`

Uses `ConveyorSceneEnv` (`sim/conveyor_scene.py`) to render synthetic
wrist-camera frames with the cube at randomized poses spanning the
*realistic* operating envelope — the x/y/z range and camera-relative
geometry the wrist camera actually sees during a real approach (roughly:
arm at or near `_RESET_QPOS`-family configurations, object within the
conveyor's travel path and within a few tens of cm of the camera, not
arbitrary poses that never occur in the real episode). For each frame:

1. Randomize the cube's world position (and optionally the arm's joint
   configuration, within its real operating range) via direct `qpos`
   manipulation (same pattern used throughout this session's isolation
   experiments) — no physics settling needed, this is a rendering
   exercise, not a dynamics one.
2. Render RGB at the same resolution the real pipeline uses.
3. Compute the **exact** ground-truth 2D bounding box by projecting the
   cube's 8 known 3D corners (from its geom size + world pose) through the
   camera's true intrinsics (`env.camera_intrinsics`) and extrinsics
   (`env.data.cam_xpos`/`cam_xmat`), then taking the pixel-space
   axis-aligned bounding box of the projected corners, clipped to the
   image. This is exact by construction — no manual annotation, no
   annotation noise, the core advantage of synthetic data.
4. Write image + YOLO-format label (`class x_center y_center width height`,
   normalized) to `data/dataset/{train,val}/`.

Output also includes a `data.yaml` (single class `cube`) for `ultralytics`
to consume directly.

A fixed random seed makes the dataset reproducible.

### 3.2 `train.py`

Loads `yolo11n.pt` (Ultralytics' smallest pretrained checkpoint) and
fine-tunes on the generated dataset for a modest number of epochs (start
at 50 — a single-class, low-visual-variance object converges fast; the
script should print final validation mAP so this is verifiable, not
assumed). Saves the best weights to `data/cube_detector.pt`.

### 3.3 `evaluate.py`

The actual go/no-go test. On a **fresh** batch of frames (same generation
method, different seed — not reused from training) with known ground-truth
3D cube positions:

For each frame, run both pipelines through the exact same downstream math:
- **Baseline:** `segment_object_centroid` (imported directly from
  `perception/segment.py`, unmodified) → pixel centroid → 3D point via
  `_camera_point_to_world`-equivalent back-projection, using the same
  `depth_bias=OBJECT_HALF_HEIGHT_M` correction the real pipeline uses.
- **Candidate:** trained YOLO model's highest-confidence detection → bbox
  center pixel → the same 3D back-projection path.

Both 3D estimates are compared against MuJoCo's exact ground truth
(`env.get_object_ground_truth()`). Report: mean and max 3D error for each
pipeline, the percentage improvement, and how many frames each pipeline
failed to detect the object at all (a real, comparable failure mode for
both — RGB thresholding already has documented dropout under some lighting
angles).

Also saves a handful of sample frames (e.g. 8-10) with both the baseline
centroid and the YOLO bounding box drawn on them side by side, so the
result can be visually spot-checked, not just trusted as a number —
consistent with this project's established discipline
(`experiments/watch_conveyor_tracking.py`, the whole Round 3/4 history) of
watching things work rather than trusting an aggregate metric alone.

## 4. Success Criteria (Go/No-Go)

Proceed to a Phase 2 integration spec **only if**:
- Candidate (YOLO) mean 3D error is **≥40% lower** than baseline, **and**
- Candidate mean 3D error is **under 1cm** absolute.

Rationale: the gripper's failure cliff (Section 12, Round 4) is ~3cm along
one axis; the current *aggregated, KF-filtered* commit-instant error is
~2-4cm. Halving the raw per-measurement noise floor (documented ~1-2cm) is
what would plausibly bring the filtered estimate reliably under the cliff.
If the bar isn't cleared, the result is still valuable: it rules out
"just swap the detector" as a fix and sharpens what Round 4 already
suspected.

If the experiment's own dataset generation reveals real problems (e.g.,
the synthetic distribution doesn't resemble real episode geometry, or
mAP is too low to trust the comparison), that's a finding to report
honestly, not a result to force past the threshold.

## 5. Dependencies & Environment

New dependency group (dev/experiment-only, not added to the main
`[project.dependencies]`): `ultralytics` (pulls in `torch` with CUDA
support). Installed via `uv` in a way that doesn't affect the main
pipeline's install footprint — e.g. an optional dependency group in
`pyproject.toml` (`[dependency-groups] yolo-precision = [...]`) or a
separate `requirements` file under `experiments/yolo_precision/`, decided
during planning based on what's cleanest with `uv`.

`data/` (generated images, labels, trained weights) is gitignored — these
are regenerable, potentially large artifacts, not source.

## 6. Testing

This is an experiment/evaluation script, not production pipeline code —
no `pytest` unit tests are required. Verification is:
- `generate_dataset.py`: spot-check a handful of generated label files
  against their images (a quick visual draw-and-inspect step, could be a
  `--preview` flag or a one-off script).
- `train.py`: printed validation mAP is a real, if imperfect, sanity
  signal that training worked at all.
- `evaluate.py`: the side-by-side sample images ARE the verification —
  if the drawn boxes/centroids don't visually make sense, the numbers
  shouldn't be trusted regardless of what they say.
