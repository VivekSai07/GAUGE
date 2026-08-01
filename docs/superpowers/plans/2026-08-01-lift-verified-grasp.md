# Lift-Verified Grasp Check

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Follow superpowers:test-driven-development within the task: write the failing test first, then the minimal implementation that makes it pass.

**Goal:** Strengthen this project's grasp-success check. Today,
`run_one_episode()`'s `contact_verified` field (both fingers touching the
object, via `ConveyorSceneEnv.is_grasped()`) is checked immediately after
the gripper closes and a short settle — while the object is still sitting
essentially where it was grasped. That proves *contact*, not that the
object is actually held: a gripper that merely closed around an object
resting on the conveyor platform, without a firm enough grip to support its
own weight once airborne, would still read `contact_verified: True` today.

The fix, informed by `github.com/VivekSai07/robot-manipulation-playground`'s
`src/tasks/pick_and_place_m13_reactive.py` (a "Verify Lift" state: after
closing the gripper, the arm lifts the object ~0.15m and only then checks
`is_grasped()` — resetting the whole sequence if the lift-check fails): add
a real lift phase after grasp-commit, and move the `contact_verified` check
to *after* that lift, not before it.

**Architecture:** No new modules. The lift phase is a closed-loop control
segment identical in shape to the existing post-grasp settle loop already in
`run_conveyor_demo.py::run_one_episode` — it reuses the same `mpc`, `env`,
and render-pacing pattern already in scope there. Nothing outside that
function changes.

## Global Constraints

- Simulation-only, pure Python — same as the rest of this project (see
  `docs/superpowers/specs/2026-07-29-dynamic-object-tracking-manipulation-design.md`).
- TDD: the integration test change below must be written and observed to
  fail (`KeyError`/`AssertionError` against the current code) before any
  implementation code changes.
- Do not touch `perception/`, `tracking/`, `prediction/`, `planning/`,
  `control/`, or `manipulation/` — this is scoped entirely to the post-grasp
  tail of `run_conveyor_demo.py::run_one_episode` (and `sim/conveyor_scene.py`
  only if a helper genuinely can't live in `run_conveyor_demo.py`; prefer not
  touching it).
- `grasped` keeps its existing meaning (the distance-tolerance commit
  trigger fired) — do not conflate it with `contact_verified`. This mirrors
  the project's existing design philosophy: see
  `run_conveyor_demo.py`'s module docstring, point 9(c), for why these two
  fields have always been kept separate.
- Full existing test suite (`uv run pytest -v`) must stay green.

---

### Task 1: Lift phase + post-lift `contact_verified`, plus a height-gain proof field

**Files:**
- Modify: `run_conveyor_demo.py`
- Modify: `tests/test_integration_conveyor.py`

**Interfaces:**
- `run_one_episode()`'s returned dict gains one new key:
  `object_height_gain_m: float` — the conveyor object's world-frame Z
  position immediately after the post-lift settle, minus its Z position
  at the grasp-commit instant (before any lift motion starts). Positive and
  large only if the object was genuinely carried upward with the gripper,
  not merely resting.
- `contact_verified`'s *meaning* changes: it must now reflect
  `env.is_grasped()` evaluated **after** the lift phase and its settle, not
  immediately after the gripper closes. Its type/position in the dict is
  unchanged.

**Exact values to use** (do not re-derive or re-tune these — they are sized
generously against this project's existing constants, e.g.
`_POST_GRASP_SETTLE_STEPS = 200` and `mpc_cfg["qdot_max"]`, and re-tuning is
explicitly out of scope for this task):
- `_LIFT_HEIGHT_M = 0.10` — lift the TCP straight up (world +Z) by 10cm from
  its position at the grasp-commit instant.
- `_LIFT_CONTROL_TICKS = 40` — number of MPC control ticks to run the lift
  over, at the episode's existing `control_hz` (40 ticks @ 20Hz = 2
  simulated seconds — generous given `qdot_max=1.5`).
- `_POST_LIFT_SETTLE_STEPS = 200` — sim steps held with zero commanded
  joint velocity (gripper still closed) after the lift, before the final
  `is_grasped()` check. Same magnitude as the existing
  `_POST_GRASP_SETTLE_STEPS`.
- Height-gain test threshold: `> 0.05` (half the commanded lift — generous
  margin for controller tracking error, chosen so the test proves a real
  lift happened without being brittle to exact settle dynamics).

**Step 1 (RED): Extend the integration test first**

In `tests/test_integration_conveyor.py`, add to
`test_conveyor_episode_grasps_within_tolerance` (after the existing
`contact_verified` assertion):

```python
    # Added to close a real gap: the old contact_verified check ran
    # immediately after the gripper closed, while the object was still
    # sitting where it was grasped -- that proves contact, not a hold. A
    # gripper closed around an object resting on the platform, without
    # enough grip to support it once airborne, would have passed the old
    # check. contact_verified is now checked after a real ~10cm lift (see
    # run_conveyor_demo.py); object_height_gain_m is direct proof the
    # object was actually carried upward with the gripper, not left behind.
    assert result["object_height_gain_m"] > 0.05
```

Run `uv run pytest tests/test_integration_conveyor.py -v` and confirm it
fails against the current code (`KeyError: 'object_height_gain_m'`) — this
is the RED step. Do not proceed to Step 2 without observing this failure.

**Step 2 (GREEN): Implement the lift phase in `run_conveyor_demo.py`**

In `run_one_episode`, inside the `if grasp_executor.should_close(...)`
block, the current sequence is: close gripper → `stop_conveyor_object()` →
compute `grasp_error` → run `_POST_GRASP_SETTLE_STEPS` steps → build
`result` with `contact_verified: env.is_grasped()`.

Insert the lift phase after the existing `_POST_GRASP_SETTLE_STEPS` loop and
before building `result`:

1. Record `object_pos_before_lift = env.get_object_ground_truth()` (a copy;
   `get_object_ground_truth()` already returns `.copy()`).
2. Compute `lift_target = ee_pos + np.array([0.0, 0.0, _LIFT_HEIGHT_M])`
   (`ee_pos` is already in scope from the grasp-commit block above — the TCP
   position at commit).
3. Run `_LIFT_CONTROL_TICKS` control ticks. Each tick: read
   `q_current = env.get_joint_positions()`, solve
   `qdot_cmd = mpc.solve(q_current, lift_target)`, then step the sim
   `sim_steps_per_control` times with that `qdot_cmd` (mirror the exact
   per-step pattern already used for `_POST_GRASP_SETTLE_STEPS` — including
   `viewer.sync()` + real-time pacing when `viewer is not None`). The
   gripper stays closed throughout (no `set_gripper` call needed — `ctrl[7]`
   already holds its closed value).
4. Run `_POST_LIFT_SETTLE_STEPS` sim steps with `qdot_cmd = np.zeros(7)`,
   same per-step render/pacing pattern as the existing settle loop.
5. Compute:
   - `contact_verified = env.is_grasped()` (this replaces the old
     immediate-post-grasp call — there should now be exactly one
     `env.is_grasped()` call in this function, after the lift+settle).
   - `object_height_gain_m = float(env.get_object_ground_truth()[2] - object_pos_before_lift[2])`.
6. Add `object_height_gain_m` to the `result` dict alongside the existing
   keys.

Update the module docstring: add a new numbered point (after the existing
point 9) describing this change — why the old check was insufficient (a
closed gripper merely resting on an unlifted object would pass it), what
the reference repo's "Verify Lift" state contributed to this fix, and that
`object_height_gain_m` is the direct, human-checkable proof of a genuine
hold. Follow the existing docstring's voice (see points 1-9 for style/depth)
— state what was wrong, what changed, and why, without re-narrating this
plan document.

**Step 3: Run and confirm GREEN**

```bash
uv run pytest tests/test_integration_conveyor.py -v
uv run pytest -v   # full suite must stay green
```

If `object_height_gain_m` comes back at or below the 0.05 threshold (i.e.
the object slips during the lift with this project's current grasp tuning),
that is a real finding, not a task failure to paper over — report it in
DONE_WITH_CONCERNS with the actual measured value and consider it a signal,
not something to force green by loosening the threshold below the
already-generous 0.05m in this brief.

**Step 4: Self-review**

Confirm:
- Exactly one `env.is_grasped()` call in `run_one_episode`, positioned
  after the lift + post-lift settle.
- `grasped`'s existing meaning/value is untouched by this change.
- The lift loop and settle loop both honor `render`/`viewer` the same way
  the pre-existing settle loop does (sync + pacing when `viewer is not
  None`, no special-casing when it's `None`).
- No new constants duplicate values already defined elsewhere in the file
  (e.g. don't redefine `sim_steps_per_control`).
