# Look-At MPC Cost Term (#36) — Design Spec

## 1. Purpose

[Issue #36](https://github.com/VivekSai07/GAUGE/issues/36) tracks the wrist
camera's 0% detection rate during the WAIT phase, previously attributed to
eye-in-hand geometry (the camera can't see the object once it's close to
the gripper). Measuring the real episode found a different, more
actionable cause for most of the blind window: it isn't geometric
inevitability, it's a side effect of an MPC cost function that was never
asked to keep the camera pointed at the object.

## 2. Root cause, measured

Tracing detection liveness tick-by-tick against phase and distance-to-
rendezvous:

| step | phase | detection | dist to rendezvous | object y |
|---|---|---|---|---|
| 2300 | GOTO | LIVE | 0.267 | -0.044 |
| 2325 | GOTO | LIVE | 0.248 | -0.041 |
| **2350** | GOTO | **blind** | 0.197 | -0.038 |
| ... | GOTO | blind (every tick) | ... | ... |
| 2875 | WAIT entry | blind | 0.007 (settled) | 0.021 |

Detection dies the instant GOTO starts moving the arm (step 2350), while
the object is still nearby (`y = -0.038`, nowhere close to actually leaving
the camera's frame — the earlier, ruled-out FOV/occlusion hypothesis would
predict blindness only much later, near physical contact range). The cause
is `control/mpc.py::KinematicMPC`'s cost function: it minimizes TCP
position error to the rendezvous target, with no term at all constraining
the wrist's *orientation* to keep the camera's boresight on the object.
Nothing stops the solver from choosing a wrist orientation, among the many
that reach the target position equally well (a real nullspace — see the
existing `posture_weight` docstring in `mpc.py` for a related nullspace
finding), that happens to rotate the camera away.

A separate FOV-widening experiment (`fovy` 58° → 100°) was run first and
ruled out a genuine frame-exit/occlusion cause: widening FOV only changed
which frame edge the object exited through, confirming the camera is
*translating past* the object due to arm motion, not simply too narrow to
see it.

## 3. Fix: a look-at cost term, active during GOTO

Add a new, optional, additive cost term to `KinematicMPC`, following the
exact pattern `lateral_axis_weight` (Round 3) already established —
default off, zero behavior change for any existing caller:

- **`control/panda_kinematics.py`**: add `camera_pose_symbolic()` /
  `camera_pose_numpy()`, computing the wrist camera's world position and
  orientation from the same DH chain `panda_tcp_pose_symbolic`/`_numpy`
  already use, plus the fixed mount offset `sim/conveyor_scene.py` applies
  to the real MuJoCo camera (`pos="0 0 0.05"`, `euler="{pi} 0 0"` relative
  to the hand frame). Cross-verified against
  `sim/conveyor_scene.py`'s own already-established boresight check
  (`dot(cam_forward, direction_to_fingers) ≈ 1.0` at `qpos=0`).
- **`control/mpc.py`**: new constructor parameters `camera_fk_func`
  (`None` by default) and `look_at_weight` (`0.0` by default). When both
  are set, each horizon step's cost gains a term penalizing the tracked
  object's angular deviation from the camera's boresight (its local −Z
  axis, MuJoCo's camera convention) — e.g. `1 - cos(angle between
  boresight and direction-to-target)`, computed from a new `look_at_target`
  parameter.
- **`run_conveyor_demo.py`**: during **GOTO only**, pass the live KF
  estimate (`obj_est`, already computed every tick) as `look_at_target`.
  WAIT and CLOSE are unchanged — WAIT already holds the arm still once
  parked, so there's no motion for a look-at term to correct there, and
  blurring GOTO/WAIT into one continuous visual-servo phase is explicitly
  out of scope for this round (see Section 6).

## 4. Why GOTO only, not a redesign of WAIT

The user's original framing was closer to continuous visual servoing —
keep the object in view throughout the approach, not park-and-wait blind.
That's the right direction, but the measured root cause is narrower than
"redesign the whole approach": it's specifically that GOTO's motion
rotates the camera away needlessly. Fixing that first, then measuring how
much it actually shrinks the practical blind window, is more YAGNI-
consistent than committing to a bigger state-machine change up front. If
GOTO keeps the object in view close enough to grasp time that WAIT's
dead-reckoning window becomes small, no further redesign is needed. If a
meaningful blind gap remains even with `look_at_weight`, that's the
concrete, measured case for extending look-at behavior into WAIT too — a
follow-up, not a Section-3 change.

## 5. Verification

In order:

1. `camera_pose_numpy`/`camera_pose_symbolic` agree with each other, and
   the `qpos=0` boresight check matches `sim/conveyor_scene.py`'s existing
   verified value.
2. Re-run this spec's own GOTO-blindness trace (Section 2's table) with
   `look_at_weight > 0` and confirm detections continue meaningfully past
   step 2350, not just marginally.
3. Full closed-loop episode + the existing 6-speed sweep
   (0.04–0.12 m/s): must not regress `contact_verified` at any
   currently-passing speed; check whether `grasp_error_m` improves given a
   fresher, closer `last_meas` feeding WAIT's dead-reckoning.
4. `uv run pytest -v`: 57/57, unaffected by the default-off parameters.
5. A `look_at_weight` sweep, explicitly checking for interaction with the
   existing `lateral_axis_weight`/`posture_weight` terms — three
   orientation-related cost terms competing in the same nullspace is a
   real risk to check, not assumed safe by analogy to Round 3's isolated
   addition of `lateral_axis_weight` alone.

## 6. Known scope limits (stated up front)

- Does not touch WAIT/CLOSE logic — if `look_at_weight` doesn't shrink the
  blind window enough, extending live tracking into WAIT is the next,
  separate design, not silently folded into this one.
- Does not add a second (static eye-to-hand) camera — ruled out for this
  round in favor of the cheaper, more targeted fix; still the fallback if
  this doesn't sufficiently close the gap.
- Three-way interaction between `look_at_weight`, `lateral_axis_weight`,
  and `posture_weight` is flagged as an open risk (Section 5, item 5), not
  resolved by this spec — the implementation plan must include the sweep
  that checks it.
