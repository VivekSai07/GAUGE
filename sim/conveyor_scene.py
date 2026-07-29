"""MuJoCo conveyor scene: Panda arm + eye-in-hand camera + scripted mocap object.

Deviations from the Task 2 brief's literal code (see task-2-report.md for full
detail):

1. Menagerie's `franka_emika_panda/scene.xml` pulls in the robot body tree via
   a MuJoCo-only `<include file="panda.xml"/>` directive, which
   `xml.etree.ElementTree` (and therefore `defusedxml.ElementTree`, which
   shares its data model) does not understand -- it is not standard XML.
   Parsing `scene.xml` directly, as the brief's Step 3 code does, yields a
   tree with an empty include element and none of the robot's bodies/sites,
   so the attachment-site/body search would silently fail. We instead parse
   `panda.xml` (which contains the full robot definition) as the base tree
   and manually merge in the handful of extra elements `scene.xml` adds
   (headlight/haze visuals, skybox+groundplane assets, a directional light,
   and the floor plane).

2. There is no site named `attachment_site` anywhere in this Menagerie
   revision (`m.nsite == 0` for the compiled model) -- confirmed by running
   the brief's own Step 2 inspection script. Per the brief's own fallback
   instructions ("If the name differs from attachment_site/hand..."), the
   eye-in-hand camera is mounted directly on the last-link body, which
   Step 2 confirmed is named `hand`.

3. `defusedxml.ElementTree` deliberately does not export `SubElement`/
   `tostring` (it only hardens *parsing*, not tree construction), so the
   brief's `ET.SubElement(...)` calls would raise `AttributeError`. We parse
   with `defusedxml.ElementTree` (untrusted-input-safe) and build/serialize
   with the standard library `xml.etree.ElementTree` (their `Element`
   objects are interchangeable).

4. `mujoco.MjModel.from_xml_string(xml, assets)` expects `assets` to be a
   `Mapping[str, bytes]` of file contents, not `{"assets": "<dir path>"}` as
   in the brief (which would fail: values must be bytes, not a directory
   string). Rather than hand-packing every referenced mesh (~60 files) into
   an in-memory dict, we point the model's `<compiler meshdir=...>` at the
   absolute path of the real `assets/` directory on disk before serializing.
   MuJoCo resolves plain (non-VFS) asset paths as normal filesystem paths, so
   an absolute `meshdir` lets `from_xml_string` load meshes straight off disk
   with no assets dict needed.

5. The Panda's 7 arm actuators are position-servo ("general" actuators with
   `biastype="affine"`, i.e. a PD controller commanding a target joint angle
   via `ctrl`), not velocity actuators. Verified empirically: holding
   `ctrl[:7]` fixed at a commanded value makes the joint settle at that
   *angle* and stop, rather than moving continuously -- i.e. the brief's
   literal `self.data.ctrl[:7] = qdot_cmd` would not implement a velocity
   command at all (a nonzero qdot_cmd would just be a one-time position
   target, exactly like a real Franka joint-velocity controller layered
   on top of the arm's low-level joint-position/impedance control). `step()`
   therefore integrates an internal joint-position setpoint
   (`self._q_target += qdot_cmd * dt`, clipped to each actuator's
   `ctrlrange`) and drives the position actuators with that, which is the
   standard way to implement velocity control on top of a position-servoed
   arm and was verified to produce continuously increasing joint angles
   under a constant qdot_cmd.

6. The gripper actuator (`actuator8`) has `ctrlrange="0 255"` (a Menagerie
   convention remapping the physical 0-0.04 m finger opening to a 0-255
   command range, per the comment in panda.xml) with 0 = closed,
   255 = open (confirmed via the `home` keyframe: `ctrl="... 255"` pairs
   with `qpos="... 0.04 0.04"`, i.e. fully open). The brief's
   `ctrl[7] = 0.04 if not closed else 0.0` assumed the pre-remap 0-0.04
   range and would barely crack the gripper open; `set_gripper` uses
   `0.0` / `255.0` instead.

Post-review fixes (see task-2-report.md "Fix report" section for detail):

7. The wrist camera's `euler="0 0 0"` pointed it *away* from the
   fingers/workspace, back into the arm (`dot(cam_forward,
   direction_to_fingers) == -1.0` at every pose, verified empirically -- a
   fixed local-frame mismatch, not pose-dependent). panda.xml's
   `<compiler angle="radian">` means `euler` values are radians, not
   degrees, which also made an initial "180 0 0" (i.e. 180 *radians*) test
   rotation land nowhere near a clean 180 degree flip. The camera now uses
   `euler="{pi} 0 0"` (pi radians = 180 degrees), verified to give
   `dot(cam_forward, direction_to_fingers) ~= 1.0` both at qpos=0 and at an
   arbitrary bent pose.

8. `reset()` used to leave `qpos` at all-zeros, which is outside joint4's
   own range (`[-3.0718, -0.0698]`, does not include 0) -- with no
   commanded velocity, constraint-recovery forces alone visibly drifted the
   arm over the first ~20 steps. `reset()` now loads panda.xml's `home`
   keyframe (a valid, in-range resting pose) via
   `mj_resetDataKeyframe` when present, falling back to `mj_resetData`'s
   default zero pose otherwise.

Task 13 (shrink the arm's required excursion, see task-13-report.md):

9. `reset()` used to settle at panda.xml's `home` keyframe
   (`qpos[:7] = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]`), whose
   end-effector ("hand" body / `panda_fk_numpy` output) sits at roughly
   `(0.55, 0, 0.62)` -- about 0.55m above the conveyor object's operating
   height (~0.05m). Task 12's integration test floored at ~6-9cm true grasp
   error, hypothesized to be substantially driven by that large, fast
   vertical reconfiguration destabilizing the wrist/eye-in-hand-camera
   orientation (see task-12-report.md). `reset()` now overrides just the 7
   arm joints (after loading `home` for the gripper's open state) to
   `_RESET_QPOS`, a configuration whose end-effector sits at roughly
   `(0.59, 0, 0.38)` -- cutting the required descent to the object's
   height by ~42% versus `home` -- with its approach direction (tool
   z-axis) already pointing almost exactly straight down
   (`dot(z_axis, world -z) ~= 0.9998`), matching the orientation the arm
   needs for a top-down grasp.

   This height is a deliberately *not*-maximal excursion cut, and the
   reason is a genuine, counter-intuitive finding from this task's
   investigation, documented in full in task-13-report.md: lowering the
   arm further keeps shrinking the excursion, but it also shrinks the
   wrist camera's own ground-footprint FOV (mounted on the same body),
   which shortens how long the conveyor object stays visible while the
   loop holds still to confirm a track and build a velocity estimate. A
   systematic sweep of hand heights from ~0.11m to ~0.55m, each measured
   via the real segmentation/tracking/MPC pipeline (not just an analytic
   FOV estimate), found the *true* grasp accuracy (closest approach to the
   object's real, ground-truth position over a full episode) does not
   improve monotonically as height decreases -- it is non-monotonic, with
   very low heights (<0.2m) failing to confirm a track at all (object
   visible for as few as ~9 ticks, far short of `track.m`'s 25-hit
   requirement) and heights in the ~0.45-0.55m range performing *worse*
   than `home` itself despite being lower. `_RESET_QPOS` sits at the best
   height found in that sweep short of `home` itself: it reproduces
   `home`'s own best-ever true accuracy to within ~3mm while still cutting
   the required descent by roughly 42%. See task-13-report.md for the
   full sweep table, every candidate's numbers, and why this means the
   "shrink the excursion" hypothesis this task set out to test did not,
   in fact, close Task 12's residual accuracy gap.

   `_RESET_QPOS` was found via a grid search over `q2`/`q4`/`q6` (holding
   `q1 = q3 = q5 = 0`, `q7 = -0.7853` as in `home`, since that symmetric
   family already reproduces `home`'s straight-down tool orientation)
   against the *full* DH homogeneous transform (position + z-axis
   direction), independently cross-checked via `panda_fk_numpy`; confirmed
   to sit comfortably inside every joint's real `env.model.jnt_range`
   (>=1.0 rad of margin on every joint from either limit); confirmed
   collision-free at reset (`env.data.ncon == 0`); and confirmed
   dynamically stable via an actual 200-step zero-velocity hold (max joint
   drift ~0.0075 rad, vs. this test's 0.01 rad bound at the 20-step
   checkpoint it actually asserts).
"""
import math
from pathlib import Path

import defusedxml.ElementTree as DefusedET
import mujoco
import numpy as np
import xml.etree.ElementTree as ET

_MENAGERIE_DIR = Path(__file__).parent / "assets/menagerie/franka_emika_panda"
_PANDA_XML = _MENAGERIE_DIR / "panda.xml"
_SCENE_XML = _MENAGERIE_DIR / "scene.xml"
# Confirmed via Task 2 Step 2 inspection: this Menagerie revision has no
# `attachment_site` (nsite == 0). The last-link body before the fingers is
# named "hand" -- the brief's documented fallback -- so the eye-in-hand
# camera is mounted directly on that body instead of at a named site.
_ATTACHMENT_BODY = "hand"
# panda.xml declares <compiler angle="radian">, so this rotates the camera
# by pi radians (180 degrees) about its local X axis, flipping its boresight
# from "into the arm" to "toward the fingers/workspace" -- verified via
# dot(cam_forward, direction_to_fingers) ~= 1.0 at multiple poses (see
# module docstring, point 7).
_CAMERA_EULER = f"{math.pi} 0 0"
_HOME_KEYFRAME = "home"
# Task 13: arm-only resting configuration that cuts the required descent to
# the conveyor object's operating height by ~42% versus `home` (hand height
# ~0.38m vs. `home`'s ~0.62m), chosen from a systematic height sweep (~0.11m
# to ~0.55m, each candidate measured through the real segmentation/tracking/
# MPC pipeline) as the point that best preserves the closed loop's
# demonstrated true grasp accuracy -- lower heights shrink the wrist
# camera's own ground-footprint FOV along with the excursion, cutting how
# long the conveyor object stays visible before the track can even confirm
# (as few as ~9 ticks at the lowest heights tried, far short of `track.m`'s
# 25-hit requirement), so "as low as possible" is not the best choice here.
# See module docstring, point 9, and task-13-report.md for the full sweep
# table, every candidate's numbers, and verification (grid search, joint
# limit margins, collision check, zero-velocity-hold stability check,
# visible-ticks measurement).
_RESET_QPOS = np.array([0.0, 0.1618, 0.0, -2.0022, 0.0, 2.1831, -0.7853])


def _build_model_xml() -> str:
    tree = DefusedET.parse(_PANDA_XML)
    root = tree.getroot()

    # Point meshdir at an absolute path so from_xml_string can resolve mesh
    # files straight off disk without needing a VFS/assets dict.
    compiler = root.find("compiler")
    compiler.set("meshdir", str((_PANDA_XML.parent / "assets").resolve()))

    # scene.xml's <include file="panda.xml"/> isn't processed by
    # ElementTree, so we parsed panda.xml directly above. Manually merge in
    # the handful of extra elements scene.xml adds on top of the bare robot
    # (headlight/haze visuals, skybox+groundplane assets, light, floor).
    scene_root = DefusedET.parse(_SCENE_XML).getroot()
    scene_visual = scene_root.find("visual")
    if scene_visual is not None:
        root.append(scene_visual)
    scene_asset = scene_root.find("asset")
    if scene_asset is not None:
        asset = root.find("asset")
        for child in scene_asset:
            asset.append(child)
    worldbody = root.find("worldbody")
    scene_worldbody = scene_root.find("worldbody")
    if scene_worldbody is not None:
        for child in scene_worldbody:
            worldbody.append(child)

    # Find the attachment body so we can mount the eye-in-hand camera on it.
    attach_body = None
    for body in root.iter("body"):
        if body.get("name") == _ATTACHMENT_BODY:
            attach_body = body
            break
    if attach_body is None:
        raise RuntimeError(
            f"Could not find body '{_ATTACHMENT_BODY}' in {_PANDA_XML}. "
            "Re-run Task 2 Step 2 to find the correct body name."
        )

    # Attach an eye-in-hand RGB-D camera at the attachment body's frame.
    camera = ET.SubElement(attach_body, "camera")
    camera.set("name", "wrist_cam")
    camera.set("mode", "fixed")
    camera.set("pos", "0 0 0.05")
    camera.set("euler", _CAMERA_EULER)
    camera.set("fovy", "58")

    # Add a scripted (mocap) conveyor object -- driven by our own step(),
    # not MuJoCo contact/friction physics, since conveyor motion is exactly
    # constant-velocity by design.
    obj_body = ET.SubElement(worldbody, "body")
    obj_body.set("name", "conveyor_object")
    obj_body.set("mocap", "true")
    obj_body.set("pos", "0.5 -0.3 0.05")
    geom = ET.SubElement(obj_body, "geom")
    geom.set("name", "conveyor_object_geom")
    geom.set("type", "box")
    geom.set("size", "0.02 0.02 0.02")
    geom.set("rgba", "0.8 0.1 0.1 1")

    return ET.tostring(root, encoding="unicode")


class ConveyorSceneEnv:
    def __init__(self, conveyor_velocity: np.ndarray, dt: float = 0.002):
        xml_string = _build_model_xml()
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.model.opt.timestep = dt
        self.dt = dt
        self.conveyor_velocity = np.asarray(conveyor_velocity, dtype=np.float64)
        self.data = mujoco.MjData(self.model)
        self._obj_mocap_id = self.model.body("conveyor_object").mocapid[0]
        self._arm_ctrlrange = self.model.actuator_ctrlrange[:7].copy()
        # Renderer is (re)created lazily in get_rgbd() sized to whatever
        # width/height is actually requested (mujoco.Renderer has no resize
        # method, and the brief's fixed 128x128 renderer silently ignored
        # get_rgbd's width/height arguments -- verified by test_rgbd_shapes
        # failing with a 64x64 request against a hardcoded 128x128 renderer).
        self._renderer = None
        mujoco.mj_forward(self.model, self.data)
        # Internal joint-position setpoint driven by the position-servo arm
        # actuators; step() integrates qdot_cmd into this target (see module
        # docstring, point 5).
        self._q_target = self.data.qpos[:7].copy()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        # mj_resetData's default all-zero qpos is outside joint4's own range
        # ([-3.0718, -0.0698], does not include 0), which causes visible
        # constraint-recovery drift under a zero velocity command (see
        # module docstring, point 8). Prefer the model's "home" keyframe --
        # a valid, settled resting pose -- when available.
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, _HOME_KEYFRAME)
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        # Task 13: override the 7 arm joints to a configuration close to the
        # conveyor object's operating height/orientation instead of `home`'s
        # ~0.62m-high resting pose (see module docstring, point 9). The
        # gripper's open finger qpos/ctrl (set above by the `home` keyframe
        # when present) are preserved/reasserted explicitly so this still
        # works even if `_HOME_KEYFRAME` is ever missing.
        self.data.qpos[:7] = _RESET_QPOS
        self.data.qpos[7:9] = 0.04
        self.data.ctrl[7] = 255.0
        mujoco.mj_forward(self.model, self.data)
        self._q_target = self.data.qpos[:7].copy()
        self.data.ctrl[:7] = self._q_target

    def step(self, qdot_cmd: np.ndarray) -> None:
        self._q_target = np.clip(
            self._q_target + np.asarray(qdot_cmd, dtype=np.float64) * self.dt,
            self._arm_ctrlrange[:, 0],
            self._arm_ctrlrange[:, 1],
        )
        self.data.ctrl[:7] = self._q_target
        self.data.mocap_pos[self._obj_mocap_id] += self.conveyor_velocity * self.dt
        mujoco.mj_step(self.model, self.data)

    def get_rgbd(self, width: int = 128, height: int = 128, camera: str = "wrist_cam"):
        if (
            self._renderer is None
            or self._renderer.width != width
            or self._renderer.height != height
        ):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera=camera)
        rgb = self._renderer.render().copy()
        self._renderer.enable_depth_rendering()
        depth = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()
        return rgb, depth.astype(np.float32)

    def get_object_ground_truth(self) -> np.ndarray:
        return self.data.mocap_pos[self._obj_mocap_id].copy()

    def get_joint_positions(self) -> np.ndarray:
        return self.data.qpos[:7].copy()

    def set_gripper(self, closed: bool) -> None:
        # actuator8's ctrlrange is 0-255 (0 = closed, 255 = fully open), a
        # Menagerie convention remapping the physical 0-0.04 m opening --
        # confirmed via panda.xml's comment and its "home" keyframe
        # (ctrl=255 <-> qpos=0.04 0.04, i.e. open). See module docstring,
        # point 6.
        self.data.ctrl[7] = 0.0 if closed else 255.0

    def camera_intrinsics(self, width: int = 128, height: int = 128, camera: str = "wrist_cam"):
        cam_id = self.model.camera(camera).id
        fovy_deg = self.model.cam_fovy[cam_id]
        fovy = np.deg2rad(fovy_deg)
        fy = height / (2 * np.tan(fovy / 2))
        fx = fy  # square pixels
        cx, cy = width / 2, height / 2
        return fx, fy, cx, cy
