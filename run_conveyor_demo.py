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

5. A small downward-clearance offset on the Z target. The "hand" frame that
   `panda_fk_numpy` returns sits well above the gripper fingers (~6cm at a
   downward-pointing orientation); driving it exactly to the perceived
   object height puts the open fingers *through* the floor, producing real
   contact forces (confirmed via `env.data.ncon`/`env.data.contact`) that
   visibly destabilized the joint trajectory. `_Z_CLEARANCE_M` raises the
   commanded height slightly to reduce (not fully eliminate) this.

6. `control.mpc.KinematicMPC` gained two additive, backward-compatible
   constructor parameters used here (`posture_target`/`posture_weight` and
   `terminal_weight`) -- see that module's docstring for what each does and
   why; both default to "off" so every pre-existing caller/test is
   unaffected.

Current, shipped state (post Task 15's final-review correction -- see
task-15-report.md and Section 12 of the design spec for the full history):
`configs/conveyor.yaml`'s `grasp.position_tolerance` is **0.075m**, chosen via
a full deterministic re-sweep as the value that lets `GraspExecutor` commit a
grasp at true (ground-truth) end-effector-to-object error of **~0.0709m** at
the commit instant. This is the demonstrated, accepted MVP accuracy --
`tests/test_integration_conveyor.py` **passes** deterministically against it.
This is a deliberate, documented trade-off, not the originally-specified
0.03m target: the grasp gate (`GraspExecutor.should_close`, which fires on
distance to the *commanded target* -- the live Kalman estimate plus a small
offset, not ground truth) commits earlier against a looser tolerance, at a
still-converging point; a tighter tolerance commits later, against a
more-converged MPC solution, giving a smaller true error, all the way down to
~0.055m where the gate stops firing within the step budget at all. See
task-12-report.md, task-13-report.md, task-14-report.md, task-15-report.md,
and design-spec Section 12 for the full debugging narrative, root causes, and
what a genuine accuracy improvement would require.
"""
import time

import mujoco.viewer
import numpy as np
import yaml

from control.mpc import KinematicMPC
from control.panda_kinematics import panda_fk_numpy, panda_fk_symbolic
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
# Raises the commanded Z target above the perceived object height to reduce
# (not eliminate) gripper/floor contact -- see module docstring, point 5.
_Z_CLEARANCE_M = 0.015
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
    home_ee_xy = panda_fk_numpy(q_home)[:2].copy()

    fk = panda_fk_symbolic()
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
        ee_pos = panda_fk_numpy(q_current)

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

        live_dist = float(np.linalg.norm(ee_pos - track.kf.x[:3]))
        if status == TrackStatus.CONFIRMED and live_dist > _CLOSE_RANGE_M:
            intercept = solve_intercept(
                obj_pos0=track.kf.x[:3],
                obj_vel=track.kf.x[3:],
                ee_pos=ee_pos,
                ee_max_speed=EE_MAX_SPEED,
            )
            target = (intercept[0] if intercept is not None else track.kf.x[:3]).copy()
        else:
            target = track.kf.x[:3].copy()
        target[2] += _Z_CLEARANCE_M

        qdot_cmd = mpc.solve(q_current, target)

        if grasp_executor.should_close(ee_pos, target, status, covariance=track.kf.P):
            env.set_gripper(closed=True)
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
            result = {"grasped": True, "grasp_error_m": grasp_error, "steps": step}
            break

    if result is None:
        result = {"grasped": False, "grasp_error_m": None, "steps": config["max_steps"]}

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
