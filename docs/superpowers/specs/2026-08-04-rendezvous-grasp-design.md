# Rendezvous Grasp + Perception Geometry Fixes — Design Spec

## 1. Purpose

Rounds 1–5 never produced a grasp that survived a lift (`contact_verified`
has read `False` since Round 4). Round 5 showed that a 43.8% more accurate
detector moved the closed-loop result by only ~4%, which meant the
remaining error was not in perception accuracy.

A full error-budget decomposition (this round) located it. At the grasp
commit instant, along the gripper's closing axis:

| stage | contribution |
|---|---|
| perception → KF | −0.0057 m |
| KF → target (blend) | −0.0003 m |
| **target → TCP (the arm)** | **−0.0270 m (82%)** |
| total | −0.0330 m |

The arm, not perception, was responsible for 82% of the error. This spec
covers the three root causes found and the changes that fix them.

## 2. Root causes

### 2.1 The approach is pursuit, and pursuit cannot converge here

`run_conveyor_demo.py` chases the object's live estimate and commits when
within `grasp.position_tolerance`. Measured with the commit logic removed
so the controller runs unimpeded, the arm's closest approach is a
**transient fly-by** (best 3.15 cm), after which it *falls behind* to a
5–8 cm steady state. It never converges, so:

- the commit must catch a fleeting instant;
- it fires while the arm is still moving (`|qdot| = 0.43 rad/s`), freezing
  a controller that has not reached its own target;
- the arm then coasts a further 1.95 cm during the finger-close window,
  dragging the object 1.73 cm with it.

Net result measured directly: TCP ends ~2.4 cm from the object centre,
beyond the cube's 2 cm half-width, and `finger_gap` closes to 0.0075 m —
the fingers shut *past* the cube, pinching an edge. `contact_verified`
during settle only ever meant "both fingers touch something".

### 2.2 Perception's apparent scale bias is frame-edge clipping

Measured against analytically-computed ground truth with the arm held
still, depth is accurate to 0.1 mm and centroids track truth closely —
*except* when the detection touches the image border, where the object's
true centre projects off-image and the measured centroid is clamped
inward. Fitting measured-vs-true position over a sweep:

| border-clipped detections | lateral slope | max residual |
|---|---|---|
| kept (current) | 0.9496 | 11.2 mm |
| rejected | **0.9930** | **2.5 mm** |

There is no intrinsic scale bias. This is the same defect as
[issue #32](https://github.com/VivekSai07/GAUGE/issues/32), now shown to be
the only real perception error and to be correctable.

### 2.3 `depth_bias` conflates two different corrections

`segment_object_centroid`/`yolo_centroid` add `depth_bias` to the measured
depth *before* deprojecting. The depth sensor sees the cube's **top face**;
its centre is half a cube-height below that in **world Z**, not along the
camera ray. The two coincide only when looking straight down, so off-axis
the ray-direction correction also displaces the point laterally. One
scalar cannot fix both:

| depth_bias | lateral slope | z error |
|---|---|---|
| 0.020 (shipped) | 0.9527 ❌ | −0.0006 ✅ |
| 0.034 | 1.0006 ✅ | −0.0147 ❌ |

## 3. Changes

### 3.1 `perception/yolo_segment.py`

Add a `reject_border: bool = True` parameter and change the centroid and
height correction:

1. **Reject border-clipped detections** — if the box touches the image
   edge, return `None`. The tracker already treats `None` as a miss and
   dead-reckons, which is strictly better than integrating a known-bad
   measurement.
2. **Top-face centroid** — within the colour mask, keep only pixels whose
   depth is within `_TOP_FACE_DEPTH_TOL_M` of the minimum (the top face is
   nearest the camera) and use *their* centroid and mean depth.
3. **World-Z height offset** — deproject the top-face point, then subtract
   `depth_bias` along **world Z** rather than adding it along the ray.
   This requires the camera pose, so the function returns a **world-frame**
   point and takes `cam_pos`/`cam_mat`.

This changes `yolo_centroid`'s contract from camera-frame to world-frame.
`perception/segment.py` (the classical baseline) is untouched.

### 3.2 `run_conveyor_demo.py` — rendezvous instead of pursuit

Replace the pursuit approach with a four-phase state machine:

- **TRACK** — hold still until the track is `CONFIRMED` with a usable
  velocity estimate.
- **GOTO** — compute a rendezvous point on the object's predicted path,
  `margin = speed * _RENDEZVOUS_TIME_BUDGET_S` ahead (a **time** budget, so
  it self-scales with conveyor speed), at a grasp height **derived** from
  known scene geometry (`_PLATFORM_TOP_Z + OBJECT_HALF_HEIGHT_M`), and
  drive there. Leave on convergence, stall, or timeout, so this phase can
  never hang.
- **WAIT** — arm fully stopped, gripper open, straddling the path.
  Dead-reckon the object from the last verified measurement.
- **CLOSE** — fire when the object's predicted position crosses the TCP
  along the closing axis, offset by `_CLOSE_LEAD_S` of finger-closing dead
  time. Then the existing lift + `is_grasped()` verification runs unchanged.

Crucially the steering target (the lead/rendezvous point) and the close
trigger (distance to the *object*) are **decoupled** — conflating them was
what made an earlier lead-compensation attempt worse, since the arm then
deliberately commits at the lead point, ahead of the cube.

## 4. Verification and expected result

The acceptance test already exists and is unmodified:
`tests/test_integration_conveyor.py::test_conveyor_episode_grasps_within_tolerance`
asserts `contact_verified is True` and has failed since Round 4.

Measured with the prototype (`configs/conveyor.yaml`'s own 0.08 m/s):

| | before | after |
|---|---|---|
| `grasp_error_m` | 0.0377 | **0.0135** |
| `finger_gap` | 0.0075 (closed past cube) | **0.042** (captured) |
| `object_peak_height_gain_m` | 0.033 | **0.090** |
| `contact_verified` | False | **True** |

## 5. Known limitation (to be recorded, not hidden)

The wrist camera is **blind during the final wait** (0% detection rate once
border-clipped detections are rejected), so the close trigger runs on pure
dead-reckoning from the last verified measurement. Slower objects mean a
longer blind window and more integrated drift, so the approach currently
succeeds at **4 of 6** tested conveyor speeds (0.06/0.08/0.10/0.12 m/s
pass; 0.04/0.05 fail). The project's configured speed passes. This is a
sensing-coverage limitation, not a tuning gap, and must be documented as an
open issue rather than papered over.
