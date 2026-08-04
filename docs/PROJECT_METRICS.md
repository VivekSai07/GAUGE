# GAUGE — Project Metrics

A sourced record of every accuracy number and engineering metric this
project has produced, in chronological order. Each figure links back to
the GitHub issue, milestone, or design-spec section it came from — this
file summarizes; those are the primary record.

Visual version: [GAUGE — Project Metrics (dashboard)](https://claude.ai/code/artifact/9182e99d-2984-4d2e-b4e4-ec4156ed6ff2).

## Two different accuracy numbers, kept separate

This project reports two distinct metrics that are easy to conflate and
have been kept deliberately separate throughout:

- **Grasp-targeting error** — the full closed-loop result: perception,
  tracking, prediction, interception planning, and MPC control together,
  measured as TCP-to-object distance at the instant the gripper commits.
- **Perception localization error** — perception measured in isolation
  against MuJoCo's exact simulator ground truth, independent of tracking
  or control.

A number from one is never presented as evidence for the other below.

## Grasp-targeting error, by round

| Round | Milestone | Change | Error at commit |
|---|---|---|---|
| 1 | [MVP Pipeline](https://github.com/VivekSai07/GAUGE/milestone/1) | First end-to-end closed loop (perception → tracking → prediction → planning → control → grasp) | ~7.1cm |
| 2 | [Accuracy Improvements](https://github.com/VivekSai07/GAUGE/milestone/2) | Conveyor object switched from a scripted `mocap` ghost to a real physics body ([#6](https://github.com/VivekSai07/GAUGE/issues/6)); smoothed MPC approach target; TCP-consistent (not flange-consistent) grasp targeting | 7.1cm → 4.4cm |
| 3 | [Round 3: Physical Grasp Debugging](https://github.com/VivekSai07/GAUGE/milestone/4) | Orientation-aware MPC cost term (penalizes offset along the gripper's non-correctable local X axis) ([#10](https://github.com/VivekSai07/GAUGE/issues/10)); real contact check via MuJoCo's own contact array (both fingers simultaneously, not inferred from distance) | 4.4cm → 3.9cm, `contact_verified: True`* |
| 5 | YOLO Perception Integration | Color-threshold `segment_object_centroid` swapped for the validated YOLO-detected-box + color-gated-depth `yolo_centroid` at the real pipeline's single perception call site — see [design spec, Round 5](superpowers/specs/2026-07-29-dynamic-object-tracking-manipulation-design.md#round-5-better-perception-alone-did-not-close-the-gap) | 3.9cm → 3.77cm, `contact_verified: False` (unchanged) |

\* Round 3's `contact_verified: True` was checked immediately after the
gripper closed — proving momentary contact, not a real hold. Round 4
(below) found this insufficient.

## Round 4: the lift test, and what it revealed

[#22](https://github.com/VivekSai07/GAUGE/issues/22) added a real ~10cm
TCP lift (ramped over 40 MPC control ticks) after grasp-commit, and moved
`contact_verified` to after that lift instead of immediately after
closing. Result: `contact_verified` flipped to `False` — the object
Round 3 reported as grasped had never actually been held.

Root-causing this (not re-tuning) via `systematic-debugging`
([#25](https://github.com/VivekSai07/GAUGE/issues/25),
[#26](https://github.com/VivekSai07/GAUGE/issues/26)) found, via a
controlled isolation experiment (teleporting the object to the arm's
exact TCP position, bypassing perception/tracking):

- The grip mechanism is flawless at **zero** targeting error — symmetric
  closure, holds through the full lift, every time.
- A sharp, reproducible cliff along the gripper's closing axis: **up to
  ~3.0cm of misalignment succeeds; 3.67cm and beyond fails completely.**
  X/Z offsets up to 2cm barely matter.

**Eight-plus control-layer hypotheses tested against real runs, all ruled
out:**

| # | Hypothesis | Result |
|---|---|---|
| 1 | Covariance-gated commit (`grasp.cov_threshold`) | Never fires — KF covariance plateaus at ~3.7×10⁻⁴, doesn't converge further |
| 2 | Tighter `position_tolerance` (0.035 → 0.01) | `grasp_error_m` nearly halves (0.039 → 0.021m), but grip survival doesn't improve |
| 3 | KF `meas_var` swept 20× | No measurable effect (deterministic sim — no noise to average out) |
| 4 | Higher object friction (past 5.0) | Episode stops committing to grasp entirely — reproduces a known platform-tilt instability bug |
| 5 | Partial gripper closure sized to object width | Makes it worse — fingers stop before reaching a mis-centered object |
| 6 | Explicit closing-axis MPC gate | Drives arm within 1.7cm of *target*, but target itself carries the error |
| 7 | Debounce (wait for N stable ticks) | Helps (`grasp_error_m` → 0.018m at 30 ticks) but still fails the lift |
| 8 | Redesigned commit sequence (fresh re-measurement + fast momentum-kill) | Measurably improves (0.039 → 0.027m) but doesn't clear the ~3cm cliff |

Three more leads, sourced from external research
([#26](https://github.com/VivekSai07/GAUGE/issues/26)), also ruled out:
contact softening/stiffening (24 `solimp`/`solref`/`condim` configurations
swept, no effect), pre-impact velocity matching (no effect isolated or in
the full pipeline), and force/plateau-triggered grasp stop (breaks the
previously-working zero-offset case — for a 0.05kg object, any early stop
leaves insufficient squeeze force).

**Conclusion:** this is a targeting-precision limit, not a control-logic
bug — documented in full in
[the design spec's Round 4 sections](superpowers/specs/2026-07-29-dynamic-object-tracking-manipulation-design.md#12-demonstrated-accuracy--known-limitation).

## Perception localization error: YOLO vs. color-threshold segmentation

A cheap go/no-go experiment ([#28](https://github.com/VivekSai07/GAUGE/issues/28),
[#29](https://github.com/VivekSai07/GAUGE/issues/29)): does a trained YOLO
detector localize the cube more precisely than the existing RGB
color-threshold segmentation? Full writeup:
[design spec](superpowers/specs/2026-08-02-yolo-detector-precision-validation-design.md#7-results).

**Training:** `yolo11n.pt` fine-tuned for 50 epochs on 800 synthetic
frames with **exactly** auto-generated labels (the cube's known 3D
corners projected through the camera's true intrinsics/extrinsics — zero
manual annotation, zero annotation noise). Result: **mAP50 = 0.9949,
mAP50-95 = 0.9771**.

**Depth-sampling took four iterations to get right**, each on real
re-measured feedback, not tuning to a threshold:

| Attempt | Approach | Result |
|---|---|---|
| 1 | Mean depth over the full YOLO bounding box | NO-GO, −186.4% (worse than baseline) |
| 2 | Median-distance inlier filtering within the box | NO-GO, −231.0% (worse still — background pixels can be the *majority* of a rotated box) |
| 3 | Color-mask depth within the box | NO-GO, −24.9% (close, but zero-match frames fell back to the original bug) |
| 4 | Zero color-match = miss (matching the baseline's own semantics), not a fallback estimate | **GO, +43.8%** |

**Final measured result** (150 held-out frames, never used in training):

| | Baseline (RGB threshold) | Candidate (YOLO) |
|---|---|---|
| Mean error | 1.08cm | **0.61cm** |
| Max error | 1.66cm | 2.75cm |
| Misses | 9/150 | 9/150 (identical frame set) |

**Two qualifications that travel with the GO result:**

1. The candidate is a **hybrid** — YOLO for the 2D pixel location, the
   baseline's own color threshold for depth — not a pure learned
   pipeline. It inherits the baseline's lighting-dropout failure mode.
2. It's a **mean-error win with a worse tail** — relevant because the
   Round 4 problem is specifically a tail/cliff failure, not an
   average-accuracy one.

**Follow-up ([#32](https://github.com/VivekSai07/GAUGE/issues/32)):**
root-caused the tail. Reproducing the same 150-frame run with per-frame
instrumentation found the outliers are not rotation- or lighting-driven
(`corr(error, rotation_severity) = 0.174`) — they're **frame-edge
clipping** (`corr(error, near_frame_edge) = 0.687`,
`corr(error, color_matched_pixel_count) = -0.889`). Near-edge frames
(33/141) average 1.20cm error — worse than the baseline's overall mean;
non-edge frames (108/141) average 0.43cm. The worst single frame (2.75cm)
has the cube clipped to a 2-3px sliver with exactly 1 color-matched pixel
inside the box. Since a real grasp approach actively centers the object
in frame, this specific failure mode should be rare in the geometry that
actually matters at commit time — worth confirming with real episode
telemetry before scoping a pipeline-integration spec around this result.

## Engineering metrics

- **Tests:** 53 total (52 passing, 1 failing by design — the honest,
  documented, still-open Round 4 grasp-lift limitation, not a hidden
  regression).
- **GitHub issues:** 22 closed with a verified fix, 3 open on purpose
  ([#13](https://github.com/VivekSai07/GAUGE/issues/13) — grasp doesn't
  yet survive a lift; [#27](https://github.com/VivekSai07/GAUGE/issues/27)
  — a ~32° cube-tilt physics artifact in the main scene, already fixed
  once in an isolated experiment, not yet ported; [#32](https://github.com/VivekSai07/GAUGE/issues/32)
  — this session's tail-error finding).
- **Milestones:** 5, each grouping a coherent round of work — MVP,
  Accuracy Improvements, Developer Experience & CI/CD, Round 3 (Physical
  Grasp Debugging), YOLO Detector Precision Validation.
- **CI:** pytest across a Python 3.11/3.12 matrix, `ruff` lint + format
  checks, CodeQL and Dependency Review security scanning, submodule
  sparse-checkout caching.

## What's proven vs. what's still open

**Proven:** the tracking system — constant-velocity Kalman filter,
Mahalanobis gating, m/n track confirmation, closed-form interception,
kinematic MPC — runs reliably in real time. Grasp verification is honest:
a mechanical contact check is no longer treated as proof of a hold.

**Open:** the grasp does not yet survive a real lift in the full closed
loop, root-caused to a ~3cm targeting-precision cliff rather than a
control-strategy bug. The validated YOLO detector has now been wired into
the real pipeline (Round 5): `grasp_error_m` moved 0.0394 → 0.0377 and
`object_peak_height_gain_m` moved 0.0300 → 0.0327 — real but small gains.
`contact_verified` is still `False`. Better perception alone does not
close the gap; the concrete next hypothesis is the KF/prediction-smoothing
layer — re-tune or bypass the KF blend for the final-approach
re-measurement now that the raw measurement itself is meaningfully
better.
