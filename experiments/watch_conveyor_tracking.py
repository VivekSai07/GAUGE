"""Standalone conveyor-only tracking check: NO robot arm.

Isolates the perception pipeline from the rest of the stack to answer one
question visually: does the segmentation centroid actually land on the
cube, or something else? Conveyor mechanism (rollers + belt + a real,
physically-simulated, velocity-actuated free body) adapted from
github.com/felixokolo/MuJoCo_tutorials/1/conveyor.xml, scaled to match this
project's existing conventions (OBJECT_HALF_HEIGHT_M=0.02, belt height
matching sim/conveyor_scene.py's platform).

Run: uv run python experiments/watch_conveyor_tracking.py
Press 'q' in the window to quit.
"""
import cv2
import mujoco
import numpy as np

_BELT_Y_HALF = 0.45  # cube wraps before reaching the camera's FOV edge (~0.577 at this height/fovy)

_MODEL_XML = """
<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.002"/>
  <visual><headlight ambient="0.4 0.4 0.4"/></visual>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.01" rgba="0.5 0.5 0.5 1"/>
    <camera name="topdown" pos="0.5 0 1.1" euler="0 0 0" fovy="60"/>
    <!-- contype/conaffinity=0: rollers are visual only (spin for a "real
         conveyor" look). Their cylinders geometrically overlap the belt
         surface at each end (radius/length were sized for looks, not
         clearance) -- with collision left on, the cube runs into a roller
         near the far end and catastrophically tips (~46 deg, confirmed by
         direct instrumentation). The cube's actual motion is driven by its
         own velocity actuator below, not by belt/roller friction, so the
         rollers don't need to physically interact with anything. -->
    <body name="roller1" pos="0.5 0.52 0.03" euler="1.5708 0 0">
      <joint name="roller1_joint" type="hinge" damping="0.01"/>
      <geom type="cylinder" size="0.02 0.16" rgba="0.2 0.2 0.2 1" contype="0" conaffinity="0"/>
    </body>
    <body name="roller2" pos="0.5 -0.52 0.03" euler="1.5708 0 0">
      <joint name="roller2_joint" type="hinge" damping="0.01"/>
      <geom type="cylinder" size="0.02 0.16" rgba="0.2 0.2 0.2 1" contype="0" conaffinity="0"/>
    </body>
    <body name="belt" pos="0.5 0 0.03">
      <geom type="box" size="0.15 0.55 0.01" rgba="0.15 0.15 0.15 1" friction="0.001 0.005 0.001"/>
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

    # Rollers used to be driven by their own velocity actuator + an
    # equality constraint syncing them. Confirmed by direct isolation test
    # (set roller ctrl to 0, cube moved fine; left at its previous value,
    # cube froze): the thin cylinder's tiny rotational inertia meant the
    # actuator's kv gain demanded an acceleration the solver couldn't
    # resolve in one step (the "Nan/Inf in QACC at DOF 0" warning), and
    # that divergence poisoned the whole timestep's solve -- not just the
    # roller, the cube too. Rollers are purely decorative (the cube's
    # actual motion comes from cube_vel below), so they're now spun
    # kinematically -- direct qpos increment, no actuator, no dynamics,
    # no way to destabilize anything else.
    roller1_qpos_addr = model.jnt_qposadr[model.body("roller1").jntadr[0]]
    roller2_qpos_addr = model.jnt_qposadr[model.body("roller2").jntadr[0]]
    roller_spin_rate = 4.0  # rad/s, visual only

    data.ctrl[model.actuator("cube_vel").id] = 0.08

    renderer = mujoco.Renderer(model, height=240, width=240)

    while True:
        data.qpos[roller1_qpos_addr] += roller_spin_rate * model.opt.timestep
        data.qpos[roller2_qpos_addr] += roller_spin_rate * model.opt.timestep
        mujoco.mj_step(model, data)

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
            u, v = int(round(xs.mean())), int(round(ys.mean()))
            cv2.drawMarker(frame, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
            cv2.putText(
                frame, f"centroid=({u},{v})  n_px={len(xs)}", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
            )
        else:
            cv2.putText(
                frame, "no cube detected", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
            )

        frame = cv2.resize(frame, (480, 480), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("top-down tracking (green cross = detected centroid)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
