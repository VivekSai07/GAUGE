"""Top-level closed loop: perceive -> track -> predict -> intercept -> control -> grasp.

Deviations from the Task 12 brief's suggested code, found during genuine
integration debugging (see task-12-report.md for the full narrative):

1. Camera-frame -> world-frame transform. The brief assumed "camera frame ==
   world frame" for the perception measurement path. This is false -- the
   wrist camera is eye-in-hand (mounted on the moving "hand" body, see
   sim/conveyor_scene.py), so its frame translates and rotates with the arm.
   `perception.segment.segment_object_centroid` returns a point in the
   pinhole camera frame (x=right, y=down, z=forward-depth), which must be
   transformed into world coordinates before it can be compared with
   FK-derived end-effector positions or fed into a world-frame
   constant-velocity KF. That transform is derived and empirically verified
   (against `env.get_object_ground_truth()`) in `_camera_point_to_world`
   below.

   Post-Task-14 fix (see design spec Section 12): the ~2-2.5cm residual
   against ground truth was root-caused to a systematic, not random, +0.02m
   Z-axis bias -- the segmented centroid lands on the box's visible top
   face, `sim.conveyor_scene.OBJECT_HALF_HEIGHT_M` above its volumetric
   center, not on the center itself, since the object's geometry (like the
   rest of this MVP) is known by design. This is now corrected by passing
   `depth_bias=OBJECT_HALF_HEIGHT_M` to `yolo_centroid` (see point 11
   below for the perception swap), which cut the measured mean residual
   from ~0.0205m (dominated by the +0.0196m Z bias) to ~0.0095m (Z bias
   ~0, leaving only genuine, unbiased x/y noise).

2. Real per-joint limits, not a uniform +-2.8 rad. `configs/conveyor.yaml`'s
   original `mpc.q_min`/`q_max` scalars do not match this Menagerie Panda's
   actual per-joint ranges (e.g. joint4 is [-3.0718, -0.0698] -- entirely
   negative -- and joint6 is [-0.0175, 3.7525] -- entirely positive).
   `run_one_episode` now reads the real ranges from `env.model.jnt_range`.

3. Deliberate hold-then-move staging, not "move as soon as tracked". Moving
   the arm immediately (as the brief's suggested code does) starts a large,
   fast excursion from the arm's home pose down to table height before the
   Kalman filter has accumulated more than 2-3 noisy measurements -- nowhere
   near enough to estimate the conveyor's velocity accurately (observed:
   estimated v_y as low as 0.005 m/s against a true 0.08 m/s after only a
   few updates). `run_one_episode` holds the arm still (zero commanded
   velocity) while the track is TENTATIVE and while the tracked object is
   still far (in x/y) from the arm's reachable footprint -- the object stays
   visible to the static camera for ~7 seconds either way (empirically
   confirmed), so waiting costs nothing and lets the KF's velocity estimate
   converge close to the true 0.08 m/s before any motion starts.

4. `EE_MAX_SPEED`/`solve_intercept` used only for the long-range leg, not
   continuously. The closed-form intercept solve is numerically well-behaved
   when the remaining distance is large, but becomes ill-conditioned as
   `ee_pos` approaches `obj_pos0` (the quadratic's coefficients shrink
   toward the degenerate case) -- an outlier target more than a meter away
   was observed when this was called every tick regardless of remaining
   distance. Once within `_CLOSE_RANGE_M`, this loop switches to tracking the
   live KF position directly (no lookahead), which is stable and, given the
   conveyor's slow (0.08 m/s) speed relative to the control rate, loses
   negligible accuracy from dropping the lookahead term.

5. [Superseded by point 8 below -- kept for history.] An earlier version of
   this loop targeted the *flange* ("hand" frame) and added a downward Z
   clearance offset to avoid driving the flange (and therefore the fingers,
   ~10cm below it) into the floor. Now that targeting/gating/reporting all
   use the TCP (fingertip midpoint) directly, no such offset is needed --
   targeting the object's own center height is correct, since that's where
   the fingertips should be to straddle and close around it.

6. `control.mpc.KinematicMPC` gained two additive, backward-compatible
   constructor parameters used here (`posture_target`/`posture_weight` and
   `terminal_weight`) -- see that module's docstring for what each does and
   why; both default to "off" so every pre-existing caller/test is
   unaffected.

7. The conveyor object is a real, physically-simulated body (free joint +
   velocity actuators), not a `mocap` body -- see `sim/conveyor_scene.py`'s
   module docstring for why. `get_object_ground_truth()`'s underlying
   representation changed accordingly, but this module's own code is
   unaffected (it only ever called that method, never touched mocap
   internals directly).

8. Target discontinuity at the lookahead-to-live-tracking switch, and the
   flange-vs-TCP frame gap, were both root-caused from a user-visible bug
   report (the render-mode demo visibly missing the object) rather than
   from a tuning sweep -- see design spec Section 12 for the full story.
   Fix (a): the hard switch at `_CLOSE_RANGE_M` (this loop used to jump
   `target` discontinuously between the lookahead intercept point and the
   live estimate right at that boundary) is now a continuous blend -- see
   the `blend` computation below. Fix (b): `ee_pos`/`target`/`grasp_error_m`
   all now use `panda_tcp_numpy`/`panda_tcp_symbolic` (the fingertip-pad
   midpoint) instead of `panda_fk_numpy`/`panda_fk_symbolic` (the flange),
   consistently throughout targeting, gating, and reporting -- previously
   only the reported metric was flange-based while nothing was TCP-based at
   all, which Section 11 flagged as a candidate fix; this implements it.

9. Round 2 (task 8, above) improved position accuracy but the user
   reported the arm still never actually picked up the cube. Direct
   instrumentation (not another tuning sweep) found the real mechanism:
   the object sat well-centered in aggregate 3D distance but ~3cm offset
   along the gripper's local X axis -- the one axis the fingers' closing
   motion physically cannot correct for (they only move along local Y).
   Comparing against a working reference
   (github.com/VivekSai07/robot-manipulation-playground) confirmed the
   missing piece: that controller does full 6D pose (position +
   orientation) tracking; this one had only ever tracked position. Three
   fixes, each independently verified:
   (a) `control.panda_kinematics.panda_tcp_pose_symbolic` exposes the
       gripper's orientation (not just position), and
       `control.mpc.KinematicMPC`'s new `lateral_axis_weight` cost term
       penalizes the target's offset along the gripper's local X axis
       specifically -- this is what actually centers the object between
       the fingers, not just gets the TCP close to it.
   (b) The conveyor object's velocity actuators used to run for the
       *entire* episode with nothing to ever stop them, so even a
       mechanically successful grasp was fighting the object's own
       commanded motion forever. `env.stop_conveyor_object()` is now
       called the instant the gripper closes.
   (c) Even after (a) and (b), the object was slipping out of the closed
       gripper under gravity -- confirmed via a real contact check
       (`env.is_grasped()`, ported from the same reference repo's
       `grasp_controller.py::is_grasped` pattern: both fingers
       simultaneously in contact, MuJoCo's own contact array, not
       inferred from distance) held True only briefly before flipping to
       False. Raising the object's friction (`sim/conveyor_scene.py`)
       from 1.0 to 3.0 fixed this -- verified via a long-hold check
       (1000 settle steps) that contact now stays True for a sustained
       ~0.6s window, not just an instant. `contact_verified` in this
       function's return value, and `tests/test_integration_conveyor.py`,
       now check this directly instead of trusting distance alone.

10. `contact_verified` (point 9c) was still checked immediately after the
    gripper closed, while the object was still sitting exactly where it
    was grasped -- a closed gripper merely resting on an ungripped object
    (friction and geometry alone holding it in the fingers' footprint,
    no real grip force) could pass that check without ever being able to
    support the object's weight once airborne. The reference repo
    (github.com/VivekSai07/robot-manipulation-playground)'s state machine
    has a distinct "Verify Lift" state after its grasp state for exactly
    this reason -- contact at rest is not proof of a hold. `run_one_episode`
    now commands a real ~10cm TCP lift (`_LIFT_HEIGHT_M`, straight up from
    the grasp-commit TCP position, held for `_LIFT_CONTROL_TICKS` control
    ticks via the same `mpc.solve` used for tracking) and a post-lift
    settle (`_POST_LIFT_SETTLE_STEPS`, gripper still closed) before the
    single `env.is_grasped()` call that now determines `contact_verified`.
    `object_height_gain_m` and `object_peak_height_gain_m` are added to the
    return value as direct, human-checkable proof that the object was
    actually carried upward with the gripper, not left behind on the
    platform while the gripper merely closed around empty space above it.
    Both are measured relative to the object's world-frame Z position at
    the grasp-commit instant (captured immediately after `set_gripper` /
    `stop_conveyor_object`, before the post-grasp settle loop runs --
    capturing it after that loop would let the object's settle-induced sag
    silently eat into the baseline). `object_height_gain_m` is the FINAL
    gain: Z after the full lift+settle minus that baseline -- it can come
    back down (even below zero) if the object slips after being lifted.
    `object_peak_height_gain_m` is the highest gain observed at any point
    during the lift+settle window -- direct proof a genuine lift happened
    at all, even one that didn't hold; a final-only reading cannot tell
    "never lifted" apart from "lifted then slipped back down."
11. Perception swapped from perception.segment.segment_object_centroid
    (pure RGB color-threshold) to perception.yolo_segment.yolo_centroid
    (a YOLO-detected bounding box center, still using the same color
    threshold to gate which pixels inside that box count for depth) --
    see docs/superpowers/specs/2026-08-04-yolo-perception-integration-design.md.
    Validated in isolation to cut mean 3D localization error 43.8%
    (docs/superpowers/specs/2026-08-02-yolo-detector-precision-validation-design.md,
    Section 7) before being wired in here.
12. Pursuit replaced with a rendezvous phase machine (design spec
    docs/superpowers/specs/2026-08-04-rendezvous-grasp-design.md, Sections
    2.1/3.2). An error-budget decomposition at the grasp-commit instant,
    along the gripper's closing axis, found the arm itself -- not
    perception -- responsible for 82% of the total error (-0.0270m of
    -0.0330m). Root cause: chasing the object's live estimate and
    committing on proximity is pursuit, and pursuit measurably never
    converges here (best transient approach 3.15cm, then falling behind
    to a 5-8cm steady state), so the commit fired mid-motion
    (`|qdot| = 0.43 rad/s`) and the arm coasted a further ~2cm into the
    object during the finger-close window.
    Fix: `run_one_episode` now runs four phases -- TRACK (hold still until
    CONFIRMED with a usable velocity estimate), GOTO (drive to a
    rendezvous point placed `speed * _RENDEZVOUS_TIME_BUDGET_S` ahead of
    the object on its predicted path, at the known cube-center height
    `_GRASP_Z`; horizontal velocity only -- a nonzero z estimate is noise,
    not real vertical motion, and extrapolating it shifted an earlier
    rendezvous point 1.9cm too high), WAIT (arm fully stopped, gripper
    open, dead-reckoning the object from the last verified measurement),
    and the close trigger (fires when the dead-reckoned object position
    crosses the TCP along the gripper's closing axis, `_CLOSE_LEAD_S`
    ahead to cover finger-closing dead time). The steering target (the
    rendezvous point) and the close trigger (distance to the object
    itself) are deliberately decoupled -- conflating them made an earlier
    lead-compensation attempt worse, since the arm would then commit at
    the lead point, ahead of the cube, instead of where the cube actually
    is. `solve_intercept`/`GraspExecutor`/`EE_MAX_SPEED`/`_CLOSE_RANGE_M`
    are no longer used by this loop (their modules are untouched --
    other callers/tests still use them).
    Measured with the prototype (design spec Section 4, `configs/
    conveyor.yaml`'s 0.08 m/s): `grasp_error_m` 0.0377 -> 0.0135,
    `finger_gap` 0.0075 (closed past the cube) -> 0.042 (genuine
    capture), `object_peak_height_gain_m` 0.033 -> 0.090,
    `contact_verified` False -> True -- the first real grasp since
    Round 4. Known limitation (design spec Section 5, not hidden): the
    wrist camera sees nothing during the final WAIT (0% detection once
    border-clipped detections are rejected), so the close trigger runs on
    pure dead-reckoning; this currently passes at 4 of 6 tested conveyor
    speeds (0.06/0.08/0.10/0.12 m/s pass, 0.04/0.05 fail) -- a
    sensing-coverage gap, not a tuning one. The project's configured speed
    (0.08 m/s) passes.
"""

import time

import mujoco.viewer
import numpy as np
import yaml
from ultralytics import YOLO

from control.mpc import KinematicMPC
from control.panda_kinematics import (
    panda_tcp_numpy,
    panda_tcp_pose_numpy,
    panda_tcp_pose_symbolic,
    panda_tcp_symbolic,
)
from perception.camera import CameraIntrinsics, camera_point_to_world
from perception.yolo_segment import MODEL_PATH, yolo_centroid
from sim.conveyor_scene import OBJECT_HALF_HEIGHT_M, ConveyorSceneEnv
from tracking.kf import ConstantVelocityKF
from tracking.track import Track, TrackStatus

# Rendezvous approach (design spec 2.1/3.2). The arm parks AHEAD of the
# object on its path and lets it arrive, rather than chasing it: measured
# with pursuit, the arm never converges (best approach 3.15cm, falling
# behind to a 5-8cm steady state), so the commit necessarily fired while
# the arm was still moving and it then coasted ~2cm into the object.
_PLATFORM_TOP_Z = 0.03  # sim/conveyor_scene.py's platform top
_GRASP_Z = _PLATFORM_TOP_Z + OBJECT_HALF_HEIGHT_M  # cube centre height
_RENDEZVOUS_TIME_BUDGET_S = 3.0  # object-travel time placed ahead of it
# The WAIT phase's blind dead-reckoning window is bounded by this budget,
# and it must stay comfortably under configs/conveyor.yaml's
# track.max_consecutive_misses (in control ticks: budget * control_hz vs.
# max_consecutive_misses) -- if the track goes LOST mid-WAIT, `track` is
# set to None and `continue`s (see the LOST-handling branch below), the
# close trigger is never re-evaluated, and the episode silently runs to
# max_steps with grasped: False and no indication why. Measured headroom
# at the current 3.0s budget / 20 Hz control_hz (=60 ticks) against
# max_consecutive_misses: 100 is real but not huge -- raising this budget
# much past ~5.0s (100 ticks) would eat it entirely.
_SETTLE_TOL_M = 0.008
_GOTO_TIMEOUT_S = 4.0
_GOTO_STALL_QDOT = 0.05
_GOTO_STALL_DIST_M = 0.03
_CLOSE_LEAD_S = 0.05  # finger-closing dead time

# Extra sim steps held after the gripper-close command, purely so the
# commanded closing motion actually plays out in the physics (long enough to
# fully close: fingers take ~200 steps to reach the closed ctrlrange) instead
# of the episode ending on the same tick the command is issued. This does not
# change what's measured/reported -- `grasp_error_m` is still the true
# distance at the commit instant, before these extra steps run. Whether the
# fingers actually make and hold contact during these steps is not verified
# anywhere in this pipeline; see Section 12 for why the reported accuracy is
# a commit-instant distance, not a verified successful grasp.
_POST_GRASP_SETTLE_STEPS = 200
# Lift phase constants (see module docstring, point 10). The lift target is
# straight up (+Z) from the TCP's grasp-commit position; ticks/settle are
# sized generously against mpc_cfg["qdot_max"] and _POST_GRASP_SETTLE_STEPS
# respectively -- see task-1-brief.md for the sizing rationale.
_LIFT_HEIGHT_M = 0.10
_LIFT_CONTROL_TICKS = 40
_POST_LIFT_SETTLE_STEPS = 200


# Moved to perception/camera.py as the public `camera_point_to_world` (see
# design spec docs/superpowers/specs/2026-08-04-rendezvous-grasp-design.md
# Section 3.1) -- yolo_centroid now needs it internally to return a
# world-frame point directly. Aliased here so this module's other call
# sites don't need touching.
_camera_point_to_world = camera_point_to_world


def run_one_episode(config: dict, render: bool = False) -> dict:
    """Run one closed-loop episode.

    `render=True` opens an interactive MuJoCo viewer window, paces the sim
    to real time, and -- once the episode ends (grasped or not) -- holds the
    window open so the final state stays visible until you close it
    yourself. Purely opt-in: default behavior and every existing caller/test
    is unaffected.
    """
    env = ConveyorSceneEnv(
        conveyor_velocity=np.array(config["conveyor_velocity"]),
        dt=config["dt"],
    )
    env.reset()
    viewer = mujoco.viewer.launch_passive(env.model, env.data) if render else None

    cam_cfg = config["camera"]
    fx, fy, cx, cy = env.camera_intrinsics(cam_cfg["width"], cam_cfg["height"])
    intrinsics = CameraIntrinsics(fx, fy, cx, cy)
    cam_id = env.model.camera("wrist_cam").id
    detector = YOLO(str(MODEL_PATH))

    kf_cfg, track_cfg = config["kf"], config["track"]
    track: Track | None = None

    # Real per-joint limits (see module docstring, point 2), not the naive
    # uniform bound.
    q_min = env.model.jnt_range[:7, 0].copy()
    q_max = env.model.jnt_range[:7, 1].copy()
    q_home = env.get_joint_positions().copy()

    fk = panda_tcp_symbolic()
    pose_fk = panda_tcp_pose_symbolic()
    mpc_cfg = config["mpc"]
    mpc = KinematicMPC(
        fk_func=fk,
        horizon=mpc_cfg["horizon"],
        dt=1.0 / config["control_hz"],
        q_min=q_min,
        q_max=q_max,
        qdot_max=np.full(7, mpc_cfg["qdot_max"]),
        posture_target=q_home,
        posture_weight=mpc_cfg.get("posture_weight", 0.0),
        terminal_weight=mpc_cfg.get("terminal_weight", 0.0),
        pose_fk_func=pose_fk,
        lateral_axis_weight=mpc_cfg.get("lateral_axis_weight", 0.0),
    )

    sim_steps_per_control = max(1, round((1.0 / config["control_hz"]) / config["dt"]))
    qdot_cmd = np.zeros(7)
    result = None
    phase = "TRACK"
    rendezvous = None
    goto_ticks = 0
    prev_offset = None
    last_meas = None
    last_meas_t = None
    tick_dt = 1.0 / config["control_hz"]

    for step in range(config["max_steps"]):
        if viewer is not None and not viewer.is_running():
            result = {"grasped": False, "grasp_error_m": None, "steps": step}
            break

        step_start = time.perf_counter()
        env.step(qdot_cmd)

        if viewer is not None:
            viewer.sync()
            remaining = env.dt - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)

        if step % sim_steps_per_control != 0:
            continue

        rgb, depth = env.get_rgbd(cam_cfg["width"], cam_cfg["height"])
        cam_pos = env.data.cam_xpos[cam_id]
        cam_mat = env.data.cam_xmat[cam_id].reshape(3, 3)
        measurement = yolo_centroid(
            rgb,
            depth,
            detector,
            intrinsics,
            tuple(cam_cfg["color_lower"]),
            tuple(cam_cfg["color_upper"]),
            cam_pos,
            cam_mat,
            depth_bias=OBJECT_HALF_HEIGHT_M,
        )
        if measurement is not None:
            last_meas = measurement
            last_meas_t = step * config["dt"]

        if track is None and measurement is not None:
            kf = ConstantVelocityKF(
                dt=1.0 / config["control_hz"],
                process_var=kf_cfg["process_var"],
                meas_var=kf_cfg["meas_var"],
                init_state=np.array([*measurement, 0.0, 0.0, 0.0]),
                init_cov=np.eye(6) * kf_cfg["init_cov_scale"],
            )
            track = Track(
                kf=kf,
                gate_threshold=track_cfg["gate_threshold"],
                m=track_cfg["m"],
                n=track_cfg["n"],
                max_consecutive_misses=track_cfg["max_consecutive_misses"],
            )
            # A rebuilt track starts with zero velocity and a discontinuous
            # last_meas; a stale prev_offset from before the rebuild could
            # sign-differ against the new one and fire the grasp on that
            # discontinuity instead of on the object actually crossing the
            # TCP. Reset it so the first WAIT tick after (re-)acquisition
            # can only record, never fire -- same as on first WAIT entry.
            prev_offset = None
            qdot_cmd = np.zeros(7)
            continue

        if track is None:
            qdot_cmd = np.zeros(7)
            continue

        status = track.step(measurement)
        if status == TrackStatus.LOST:
            track = None
            prev_offset = None
            qdot_cmd = np.zeros(7)
            continue

        q_current = env.get_joint_positions()
        ee_pos = panda_tcp_numpy(q_current)
        obj_est = track.kf.x[:3].copy()
        obj_vel = track.kf.x[3:].copy()
        # The cube slides on a plane; a nonzero z velocity estimate is noise
        # and must not be extrapolated (it shifted an earlier rendezvous
        # point 1.9cm too high).
        vel_horizontal = obj_vel.copy()
        vel_horizontal[2] = 0.0
        speed = float(np.linalg.norm(vel_horizontal))

        if phase == "TRACK":
            if status == TrackStatus.CONFIRMED and speed > 1e-3:
                rendezvous = obj_est + (vel_horizontal / speed) * (
                    speed * _RENDEZVOUS_TIME_BUDGET_S
                )
                rendezvous[2] = _GRASP_Z
                phase, goto_ticks = "GOTO", 0
            qdot_cmd = np.zeros(7)
            continue

        if phase == "GOTO":
            qdot_cmd = mpc.solve(q_current, rendezvous)
            goto_ticks += 1
            dist = float(np.linalg.norm(ee_pos - rendezvous))
            stalled = (
                float(np.linalg.norm(qdot_cmd)) < _GOTO_STALL_QDOT
                and dist < _GOTO_STALL_DIST_M
            )
            if (
                dist < _SETTLE_TOL_M
                or stalled
                or goto_ticks * tick_dt > _GOTO_TIMEOUT_S
            ):
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
        # Deliberately uses the full obj_vel (including its z component)
        # here, NOT vel_horizontal -- even though the rendezvous-point
        # computation above zeroes z on the grounds that a nonzero z
        # estimate is noise. That rule is right for a lookahead extrapolated
        # over _RENDEZVOUS_TIME_BUDGET_S (seconds), where noise dominates;
        # it's wrong for this much shorter dead-reckoning window, where
        # keeping z measurably wins: grasp_error_m 0.0135 with the full
        # obj_vel vs. 0.0179 (32% worse) using vel_horizontal instead, both
        # still contact_verified: True. Do not "fix" this to match the rule
        # above without re-measuring.
        predicted = last_meas + obj_vel * (elapsed + _CLOSE_LEAD_S)
        offset = float(np.dot(predicted - ee_pos, closing_axis))
        crossed = prev_offset is not None and (
            offset == 0.0 or (prev_offset < 0.0) != (offset < 0.0)
        )
        prev_offset = offset
        if not crossed:
            continue

        env.set_gripper(closed=True)
        # Found via a user-reported visual grasp failure: without this,
        # the object's conveyor velocity actuator keeps commanding
        # motion forever, fighting the grip indefinitely (confirmed by
        # direct instrumentation). A real conveyor exerts no more
        # belt-driven force once an object is lifted off it.
        env.stop_conveyor_object()
        # Baseline for the lift-verification height metrics below, taken
        # at the grasp-commit instant -- i.e. right now, before the
        # settle loop just below runs. Capturing this AFTER that settle
        # loop (as an earlier version of this code did) silently
        # contaminates the baseline: the settle loop lets the object sag
        # under the closing gripper before the lift even starts, so part
        # of the real lift ends up hidden inside the "baseline" instead
        # of counted as gain. See module docstring, point 10.
        object_pos_before_lift = env.get_object_ground_truth().copy()
        true_obj_pos = env.get_object_ground_truth()
        grasp_error = float(np.linalg.norm(ee_pos - true_obj_pos))
        # Hold the closing command for a few more sim steps so the
        # gripper's commanded closing motion actually plays out in the
        # physics (see _POST_GRASP_SETTLE_STEPS) -- purely cosmetic for
        # a recorded demo; grasp_error was already computed above, at
        # the commit instant, and is unaffected by these extra steps.
        peak_obj_z = object_pos_before_lift[2]
        for _ in range(_POST_GRASP_SETTLE_STEPS):
            settle_start = time.perf_counter()
            env.step(np.zeros(7))
            peak_obj_z = max(peak_obj_z, float(env.get_object_ground_truth()[2]))
            if viewer is not None:
                viewer.sync()
                remaining = env.dt - (time.perf_counter() - settle_start)
                if remaining > 0:
                    time.sleep(remaining)

        # Lift phase (see module docstring, point 10): a closed gripper
        # merely resting on an unlifted object can pass a contact check
        # at the commit instant even though it has no real hold. Actually
        # commanding the TCP _LIFT_HEIGHT_M upward and re-checking contact
        # afterward -- mirroring the reference repo's "Verify Lift" state
        # -- is the only way to distinguish a genuine grip from one that
        # merely happened to be touching.
        #
        # The target is ramped linearly across the ticks (alpha grows
        # from 1/_LIFT_CONTROL_TICKS to 1.0) rather than fixed at the
        # full +10cm offset for every tick. A fixed target let the MPC's
        # terminal_weight drive near-max joint velocity and close ~90% of
        # the gap in the first ~0.37s, front-loading an unnecessary
        # inertial load on the grip during the fastest phase of the
        # motion (measured ~24% grip-force surcharge) instead of the
        # paced ~2s lift the tick count was sized for.
        for i in range(_LIFT_CONTROL_TICKS):
            alpha = (i + 1) / _LIFT_CONTROL_TICKS
            lift_target = ee_pos + alpha * np.array([0.0, 0.0, _LIFT_HEIGHT_M])
            q_current = env.get_joint_positions()
            qdot_cmd = mpc.solve(q_current, lift_target)
            for _ in range(sim_steps_per_control):
                lift_start = time.perf_counter()
                env.step(qdot_cmd)
                peak_obj_z = max(peak_obj_z, float(env.get_object_ground_truth()[2]))
                if viewer is not None:
                    viewer.sync()
                    remaining = env.dt - (time.perf_counter() - lift_start)
                    if remaining > 0:
                        time.sleep(remaining)
        qdot_cmd = np.zeros(7)
        for _ in range(_POST_LIFT_SETTLE_STEPS):
            lift_settle_start = time.perf_counter()
            env.step(qdot_cmd)
            peak_obj_z = max(peak_obj_z, float(env.get_object_ground_truth()[2]))
            if viewer is not None:
                viewer.sync()
                remaining = env.dt - (time.perf_counter() - lift_settle_start)
                if remaining > 0:
                    time.sleep(remaining)

        # Real, direct verification (env.is_grasped(): both fingers
        # simultaneously in contact with the object, MuJoCo contact
        # array, not inferred from distance) -- see design spec Section
        # 12 for why `grasped`/`grasp_error_m` alone were never enough
        # to confirm an actual pick. Now evaluated after the lift +
        # settle above (module docstring, point 10), not immediately
        # after the gripper closes.
        contact_verified = env.is_grasped()
        object_height_gain_m = float(
            env.get_object_ground_truth()[2] - object_pos_before_lift[2]
        )
        object_peak_height_gain_m = float(peak_obj_z - object_pos_before_lift[2])
        result = {
            "grasped": True,
            "grasp_error_m": grasp_error,
            "steps": step,
            "contact_verified": contact_verified,
            "object_height_gain_m": object_height_gain_m,
            "object_peak_height_gain_m": object_peak_height_gain_m,
        }
        break

    if result is None:
        result = {
            "grasped": False,
            "grasp_error_m": None,
            "steps": config["max_steps"],
            "contact_verified": False,
            "object_height_gain_m": None,
            "object_peak_height_gain_m": None,
        }

    if viewer is not None:
        # Hold the window open with the final state visible until the user
        # closes it themselves, instead of the process exiting immediately.
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)
        viewer.close()

    return result


if __name__ == "__main__":
    import sys

    with open("configs/conveyor.yaml") as f:
        cfg = yaml.safe_load(f)
    print(run_one_episode(cfg, render="--render" in sys.argv))
