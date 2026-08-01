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

_BELT_Y_HALF = 0.5  # cube wraps around when it travels this far past start

_MODEL_XML = """
<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.002"/>
  <visual><headlight ambient="0.4 0.4 0.4"/></visual>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.01" rgba="0.5 0.5 0.5 1"/>
    <camera name="topdown" pos="0.5 0 1.1" euler="0 0 0" fovy="60"/>
    <body name="roller1" pos="0.5 0.52 0.03" euler="1.5708 0 0">
      <joint name="roller1_joint" type="hinge" damping="0.01"/>
      <geom type="cylinder" size="0.02 0.16" rgba="0.2 0.2 0.2 1"/>
    </body>
    <body name="roller2" pos="0.5 -0.52 0.03" euler="1.5708 0 0">
      <joint name="roller2_joint" type="hinge" damping="0.01"/>
      <geom type="cylinder" size="0.02 0.16" rgba="0.2 0.2 0.2 1"/>
    </body>
    <body name="belt" pos="0.5 0 0.03">
      <geom type="box" size="0.15 0.55 0.01" rgba="0.15 0.15 0.15 1" friction="0.001 0.005 0.001"/>
    </body>
    <body name="cube" pos="0.5 -0.4 0.05">
      <joint name="cube_joint" type="free" damping="0.1"/>
      <geom type="box" size="0.02 0.02 0.02" rgba="0.8 0.1 0.1 1" mass="0.05" friction="1.0 0.5 0.1"/>
    </body>
  </worldbody>
  <equality>
    <joint joint1="roller1_joint" joint2="roller2_joint"/>
  </equality>
  <actuator>
    <velocity name="roller_vel" joint="roller1_joint" kv="5" ctrlrange="-20 20"/>
    <velocity name="cube_vel" joint="cube_joint" gear="0 1 0 0 0 0" kv="50" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

_COLOR_LOWER = np.array([150, 0, 0])
_COLOR_UPPER = np.array([255, 80, 80])


def main() -> None:
    model = mujoco.MjModel.from_xml_string(_MODEL_XML)
    data = mujoco.MjData(model)

    cube_joint_id = model.body("cube").jntadr[0]
    cube_qpos_addr = model.jnt_qposadr[cube_joint_id]
    cube_start = data.qpos[cube_qpos_addr : cube_qpos_addr + 3].copy()

    data.ctrl[model.actuator("cube_vel").id] = 0.08
    data.ctrl[model.actuator("roller_vel").id] = 4.0  # visual only

    renderer = mujoco.Renderer(model, height=240, width=240)

    while True:
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
