# Cube Tilt (#27) + CLOSE Trigger Robustness — Implementation Plan

Spec: `docs/superpowers/specs/2026-08-05-cube-tilt-and-close-trigger-fix-design.md`

Both changes below are already prototyped and numerically verified (see
spec Section 4: 57/57 tests, 6/6 speeds passing `contact_verified`). This
plan documents them as discrete, reviewable tasks rather than one large
diff, matching this project's established SDD process.

## Global constraints

- Do not modify `tests/test_integration_conveyor.py` (the acceptance
  test).
- Do not change any friction values (`geom.set("friction", ...)`) — the
  base-drive fix must not require it, per the spec's rejection of the
  friction-decoupling alternative.
- `ConveyorSceneEnv.get_object_ground_truth()`'s external contract (world
  center of the object) must not change, even though its internal read
  source does.

## Task 1: Base-drive actuation in `sim/conveyor_scene.py`

- Move `conveyor_object` body's `pos` to the cube's base
  (`z = _PLATFORM_TOP_Z = 0.03` instead of `0.05`).
- Offset `conveyor_object_geom`'s `pos` to `f"0 0 {OBJECT_HALF_HEIGHT_M}"`
  within the body frame, so its world position/mass/inertia are
  unchanged.
- Update the `home` keyframe's appended object qpos z-value to match
  (`0.03` instead of `0.05`).
- Add `self._obj_geom_id` in `ConveyorSceneEnv.__init__` and change
  `get_object_ground_truth()` to return `self.data.geom_xpos[self._obj_geom_id].copy()`
  instead of `self.data.xpos[self._obj_body_id]`.
- Comments explaining *why* (moment-arm removal at the source, and why
  friction decoupling was rejected) — see spec Section 2.

**Verify:** a short standalone script (arm held still, object driven
3650 steps at friction=3.0) shows tilt ≤ ~0.5° (measured: 0.17°). Existing
test suite still 57/57.

## Task 2: Travel-axis CLOSE trigger in `run_conveyor_demo.py`

- In the WAIT-phase crossing check, replace the projection axis
  `tcp_rot[:, 1]` (gripper closing axis) with `vel_horizontal / speed`
  (the object's own travel direction), computed from the same `speed`
  variable already in scope from the rendezvous-point block above.
- Keep a defensive fallback to the old closing-axis behavior for the
  degenerate `speed <= 1e-3` case (shouldn't occur given WAIT is only
  entered from a CONFIRMED track with `speed > 1e-3`, but avoid an
  unguarded division).
- Comment explaining why the old axis was wrong (dominated by a
  near-constant world-X component at this arm posture) and why it was
  masked until Task 1 landed — see spec Section 3.

**Verify:** full `run_one_episode` at `configs/conveyor.yaml`'s speed
gives `contact_verified: True`, `grasp_error_m` ≈ 0.01. The 6-speed sweep
(0.04/0.05/0.06/0.08/0.10/0.12 m/s) all pass `contact_verified`.

## Final verification (whole-branch)

- `uv run pytest -v`: 57/57.
- Full episode at configured speed: `contact_verified: True`.
- 6-speed sweep: all pass.
- No changes outside `sim/conveyor_scene.py`, `run_conveyor_demo.py`, and
  this task's docs.

## Post-merge (not part of this plan's code tasks, tracked separately)

- Close #27, re-scope #36, update README/PROJECT_METRICS.md with Round 7
  numbers — per spec Section 6.
