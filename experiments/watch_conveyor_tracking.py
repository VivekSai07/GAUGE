"""Standalone conveyor-only tracking check: NO robot arm.

Isolates the perception pipeline from the rest of the stack to answer one
question visually: does the segmentation centroid actually land on the
cube, or something else? Conveyor mechanism (rollers + belt + a real,
physically-simulated, velocity-actuated free body) adapted from
github.com/felixokolo/MuJoCo_tutorials/1/conveyor.xml, scaled to match this
project's existing conventions (OBJECT_HALF_HEIGHT_M=0.02, belt height
matching sim/conveyor_scene.py's platform).

Run: uv run python experiments/watch_conveyor_tracking.py
Opens two windows: the interactive MuJoCo 3D viewer (free-look, scroll to
zoom, drag to orbit -- inspect the cube/belt from any angle) and the
top-down tracking overlay. Press 'q' in the tracking window, or close the
3D viewer window, to quit.
"""

import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np

# Cube wraps at cube_start[1] + _BELT_Y_HALF*2. Set so it wraps just AFTER
# rolling over the far roller (confirmed via instrumentation: flat until
# y~0.53, a real, modest 0-3.5deg tilt crossing the roller at y~0.53-0.58,
# then it correctly falls off the physical end of the conveyor past y~0.586
# -- nothing supports it beyond the roller. Wrapping at 0.56 shows the
# genuine roller interaction on every loop without the fall.
_BELT_Y_HALF = 0.48

_MODEL_XML = """
<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.002"/>
  <visual><headlight ambient="0.4 0.4 0.4"/></visual>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.01" rgba="0.5 0.5 0.5 1"/>
    <camera name="topdown" pos="0.5 0 1.1" euler="0 0 0" fovy="60"/>
    <!-- Real, colliding rollers (previous version disabled collision --
         a workaround, not a fix, for two real geometry errors below).
         Error 1: the reference repo's asset moves its object along X, so
         its roller axis (perpendicular to travel) is Y. Ours moves along
         Y, but the rotation was copied verbatim, leaving our roller axis
         parallel to travel instead of perpendicular -- euler="0 1.5708 0"
         (not "1.5708 0 0") correctly puts the axis along X here. Error 2:
         radius 0.02 put the roller's top 1cm *above* the belt surface (a
         wall to slam into, not a rounded end to roll over) -- radius 0.01
         with the same center height as the belt puts its top exactly
         flush with the belt surface (both at z=0.04). Positioned right at
         the belt's Y-ends, so the cube only reaches them at the very end
         of its travel, not mid-belt. -->
    <body name="roller1" pos="0.5 0.55 0.03" euler="0 1.5708 0">
      <joint name="roller1_joint" type="hinge" damping="0.01"/>
      <geom type="cylinder" size="0.01 0.16" rgba="0.2 0.2 0.2 1"/>
    </body>
    <body name="roller2" pos="0.5 -0.55 0.03" euler="0 1.5708 0">
      <joint name="roller2_joint" type="hinge" damping="0.01"/>
      <geom type="cylinder" size="0.01 0.16" rgba="0.2 0.2 0.2 1"/>
    </body>
    <body name="belt" pos="0.5 0 0.03">
      <geom type="box" size="0.15 0.55 0.01" rgba="0.15 0.15 0.15 1" friction="0.05 0.005 0.0001"/>
    </body>
    <body name="cube" pos="0.5 -0.4 0.06">
      <joint name="cube_joint" type="free" damping="0.1"/>
      <geom type="box" size="0.02 0.02 0.02" rgba="0.8 0.1 0.1 1" mass="0.05" friction="0.05 0.005 0.0001"/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="cube_vel" joint="cube_joint" gear="0 1 0 0 0 0" kv="50" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

# Widened from an initial [255,80,80] upper bound: directly under a
# top-down camera, the headlight hits the cube's flat top face at near-
# normal incidence, washing the rendered color toward [227,84,84] --
# desaturated just past the original G/B cutoff, causing real, confirmed
# detection dropouts at the exact center of the belt (directly below the
# camera). Pure RGB thresholding is inherently fragile to this kind of
# lighting-angle effect; the same fragility likely exists in the main
# project's perception/segment.py, whose wrist camera sees the same
# near-normal-incidence geometry during final approach. HSV-based
# thresholding (hue is largely lighting-invariant) would be the more
# robust long-term fix there; widening the window is the quick fix here.
_COLOR_LOWER = np.array([150, 0, 0])
_COLOR_UPPER = np.array([255, 110, 110])


def main() -> None:
    model = mujoco.MjModel.from_xml_string(_MODEL_XML)
    data = mujoco.MjData(model)

    cube_joint_id = model.body("cube").jntadr[0]
    cube_qpos_addr = model.jnt_qposadr[cube_joint_id]
    cube_start = data.qpos[cube_qpos_addr : cube_qpos_addr + 3].copy()

    # Rollers now physically collide with the cube (see the XML comment
    # above) -- but their own SPIN is still driven kinematically (direct
    # qpos increment), not by a velocity actuator. Rollers used to be
    # actuator-driven, and that specifically (not the collision) caused a
    # solver blowup: the thin cylinder's tiny rotational inertia meant the
    # actuator's kv gain demanded an acceleration the solver couldn't
    # resolve in one step ("Nan/Inf in QACC at DOF 0"), and that single
    # DOF's divergence poisoned the whole timestep's solve -- froze the
    # cube too, confirmed by direct isolation test (roller ctrl=0: cube
    # moved fine; roller ctrl active: cube froze). The visual spin doesn't
    # need real dynamics, so it's kinematic; the cube's collision with the
    # roller's actual geometry is real physics, computed by the solver
    # exactly like any other contact.
    roller1_qpos_addr = model.jnt_qposadr[model.body("roller1").jntadr[0]]
    roller2_qpos_addr = model.jnt_qposadr[model.body("roller2").jntadr[0]]
    roller_spin_rate = 4.0  # rad/s, visual only

    data.ctrl[model.actuator("cube_vel").id] = 0.08

    renderer = mujoco.Renderer(model, height=240, width=240)
    viewer = mujoco.viewer.launch_passive(model, data)

    while viewer.is_running():
        step_start = time.perf_counter()

        data.qpos[roller1_qpos_addr] += roller_spin_rate * model.opt.timestep
        data.qpos[roller2_qpos_addr] += roller_spin_rate * model.opt.timestep
        mujoco.mj_step(model, data)
        viewer.sync()

        # Wrap the cube back to the start once it clears the belt, so this
        # runs as a continuous loop instead of needing manual restarts.
        if data.qpos[cube_qpos_addr + 1] > cube_start[1] + _BELT_Y_HALF * 2:
            data.qpos[cube_qpos_addr : cube_qpos_addr + 3] = cube_start
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera="topdown")
        rgb = renderer.render().copy()

        mask = np.all((rgb >= _COLOR_LOWER) & (rgb <= _COLOR_UPPER), axis=-1)
        ys, xs = np.nonzero(mask)

        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if len(xs) > 0:
            u, v = round(xs.mean()), round(ys.mean())
            cv2.drawMarker(frame, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
            cv2.putText(
                frame,
                f"centroid=({u},{v})  n_px={len(xs)}",
                (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )
        else:
            cv2.putText(
                frame,
                "no cube detected",
                (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
            )

        frame = cv2.resize(frame, (480, 480), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("top-down tracking (green cross = detected centroid)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        remaining = model.opt.timestep - (time.perf_counter() - step_start)
        if remaining > 0:
            time.sleep(remaining)

    cv2.destroyAllWindows()
    viewer.close()


if __name__ == "__main__":
    main()
