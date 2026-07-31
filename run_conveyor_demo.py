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
   `depth_bias=OBJECT_HALF_HEIGHT_M` to `segment_object_centroid` below,
   which cut the measured mean residual from ~0.0205m (dominated by the
   +0.0196m Z bias) to ~0.0095m (Z bias ~0, leaving only genuine, unbiased
   x/y noise).

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
"""
import time

import mujoco.viewer
import numpy as np
import yaml

from control.mpc import KinematicMPC
from control.panda_kinematics import panda_tcp_numpy, panda_tcp_symbolic, panda_tcp_pose_symbolic
from manipulation.grasp import GraspExecutor
from perception.camera import CameraIntrinsics
from perception.segment import segment_object_centroid
from planning.intercept import solve_intercept
from sim.conveyor_scene import OBJECT_HALF_HEIGHT_M, ConveyorSceneEnv
from tracking.kf import ConstantVelocityKF
from tracking.track import Track, TrackStatus

# Used only for the long-range leg of the approach (see module docstring,
# point 4) -- a conservative Cartesian speed estimate for the reachability
# check in solve_intercept, well below qdot_max's raw joint-space bound to
# avoid over-projecting the rendezvous point past what the arm can actually
# achieve in practice.
EE_MAX_SPEED = 0.2
# Below this remaining distance, solve_intercept's quadratic becomes
# ill-conditioned (see module docstring, point 4); switch to direct live
# tracking instead.
_CLOSE_RANGE_M = 0.15
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


def _camera_point_to_world(
    point_cam: np.ndarray, cam_pos: np.ndarray, cam_mat: np.ndarray
) -> np.ndarray:
    """Transform a point from the pinhole camera frame (x=right, y=down,
    z=forward-depth) into world coordinates, given the camera's world
    position and its local-axes-in-world rotation matrix (MuJoCo's
    `cam_xmat`, reshaped to 3x3, columns = local x/y/z in world frame).
    """
    point_local = np.array([point_cam[0], -point_cam[1], -point_cam[2]])
    return cam_pos + cam_mat @ point_local


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

    kf_cfg, track_cfg = config["kf"], config["track"]
    track: Track | None = None

    # Real per-joint limits (see module docstring, point 2), not the naive
    # uniform bound.
    q_min = env.model.jnt_range[:7, 0].copy()
    q_max = env.model.jnt_range[:7, 1].copy()
    q_home = env.get_joint_positions().copy()
    home_ee_xy = panda_tcp_numpy(q_home)[:2].copy()

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
    grasp_executor = GraspExecutor(
        position_tolerance=config["grasp"]["position_tolerance"],
        # Optional covariance-gated commit (design spec Section 2/3.4's core
        # novelty axis) -- off by default (None) unless configs/conveyor.yaml
        # sets `grasp.cov_threshold`, so the shipped, already-tuned accuracy
        # figures in Section 12 are completely unaffected unless someone
        # deliberately opts in.
        cov_threshold=config["grasp"].get("cov_threshold"),
    )

    sim_steps_per_control = max(1, round((1.0 / config["control_hz"]) / config["dt"]))
    qdot_cmd = np.zeros(7)
    moving = False
    result = None

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
        measurement_cam = segment_object_centroid(
            rgb,
            depth,
            intrinsics,
            tuple(cam_cfg["color_lower"]),
            tuple(cam_cfg["color_upper"]),
            depth_bias=OBJECT_HALF_HEIGHT_M,
        )
        if measurement_cam is None:
            measurement = None
        else:
            cam_pos = env.data.cam_xpos[cam_id]
            cam_mat = env.data.cam_xmat[cam_id].reshape(3, 3)
            measurement = _camera_point_to_world(measurement_cam, cam_pos, cam_mat)

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
            qdot_cmd = np.zeros(7)
            continue

        if track is None:
            qdot_cmd = np.zeros(7)
            continue

        status = track.step(measurement)
        if status == TrackStatus.LOST:
            track = None
            moving = False
            qdot_cmd = np.zeros(7)
            continue

        q_current = env.get_joint_positions()
        ee_pos = panda_tcp_numpy(q_current)

        # Hold position until the track is CONFIRMED and the tracked
        # object has drifted (via the conveyor's own motion) close to
        # the arm's reachable footprint -- see module docstring, point 3.
        xy_dist_to_home = float(np.linalg.norm(track.kf.x[:2] - home_ee_xy))
        if not moving:
            if status == TrackStatus.CONFIRMED and xy_dist_to_home < _CLOSE_RANGE_M:
                moving = True
            else:
                qdot_cmd = np.zeros(7)
                continue

        # Blend the lookahead intercept point smoothly into the live
        # estimate as the arm closes in, instead of hard-switching at
        # _CLOSE_RANGE_M. The hard switch used to make `target` jump
        # discontinuously (the lookahead point leads the object by design;
        # the live estimate doesn't) right at the switch boundary -- the
        # arm, still catching up toward the pre-switch target, would
        # overshoot past the object exactly where the switch fired. `blend`
        # is 1.0 (pure lookahead) far away, 0.0 (pure live estimate) at
        # zero distance, and linear in between, so `target` now moves
        # continuously with no jump. See design spec Section 12.
        live_dist = float(np.linalg.norm(ee_pos - track.kf.x[:3]))
        live_estimate = track.kf.x[:3].copy()
        if status == TrackStatus.CONFIRMED:
            intercept = solve_intercept(
                obj_pos0=track.kf.x[:3],
                obj_vel=track.kf.x[3:],
                ee_pos=ee_pos,
                ee_max_speed=EE_MAX_SPEED,
            )
            lookahead_point = intercept[0] if intercept is not None else live_estimate
        else:
            lookahead_point = live_estimate
        blend = float(np.clip(live_dist / _CLOSE_RANGE_M, 0.0, 1.0))
        target = blend * lookahead_point + (1.0 - blend) * live_estimate
        # No Z clearance added here (see module docstring, point 5): now
        # that `ee_pos`/`target` are both TCP (fingertip) positions rather
        # than the flange, targeting the object's own center height is
        # correct -- the fingertips should be AT that height to straddle
        # and close around the object, not offset above it.

        qdot_cmd = mpc.solve(q_current, target)

        if grasp_executor.should_close(ee_pos, target, status, covariance=track.kf.P):
            env.set_gripper(closed=True)
            # Found via a user-reported visual grasp failure: without this,
            # the object's conveyor velocity actuator keeps commanding
            # motion forever, fighting the grip indefinitely (confirmed by
            # direct instrumentation). A real conveyor exerts no more
            # belt-driven force once an object is lifted off it.
            env.stop_conveyor_object()
            true_obj_pos = env.get_object_ground_truth()
            grasp_error = float(np.linalg.norm(ee_pos - true_obj_pos))
            # Hold the closing command for a few more sim steps so the
            # gripper's commanded closing motion actually plays out in the
            # physics (see _POST_GRASP_SETTLE_STEPS) -- purely cosmetic for
            # a recorded demo; grasp_error was already computed above, at
            # the commit instant, and is unaffected by these extra steps.
            for _ in range(_POST_GRASP_SETTLE_STEPS):
                settle_start = time.perf_counter()
                env.step(np.zeros(7))
                if viewer is not None:
                    viewer.sync()
                    remaining = env.dt - (time.perf_counter() - settle_start)
                    if remaining > 0:
                        time.sleep(remaining)
            # Real, direct verification (env.is_grasped(): both fingers
            # simultaneously in contact with the object, MuJoCo contact
            # array, not inferred from distance) -- see design spec Section
            # 12 for why `grasped`/`grasp_error_m` alone were never enough
            # to confirm an actual pick.
            result = {
                "grasped": True,
                "grasp_error_m": grasp_error,
                "steps": step,
                "contact_verified": env.is_grasped(),
            }
            break

    if result is None:
        result = {
            "grasped": False,
            "grasp_error_m": None,
            "steps": config["max_steps"],
            "contact_verified": False,
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
