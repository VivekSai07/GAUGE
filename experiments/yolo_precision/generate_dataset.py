"""Synthetic dataset generation for the YOLO detector precision experiment.

Renders wrist-camera frames from ConveyorSceneEnv with the cube at
randomized positions/orientations spanning its real operating envelope,
and computes EXACT 2D bounding-box labels by projecting the cube's known
8 corners through the camera's true intrinsics/extrinsics -- no manual
annotation, no annotation noise, per the design spec
(docs/superpowers/specs/2026-08-02-yolo-detector-precision-validation-design.md).

Run (as a module, from the repo root -- a direct script-path invocation puts
this file's own directory, not the repo root, at sys.path[0], so `sim`
wouldn't be importable; verified via `ModuleNotFoundError: No module named
'sim'` under the literal `python experiments/yolo_precision/
generate_dataset.py` form):
    uv run --group yolo-precision python -m experiments.yolo_precision.generate_dataset
"""
import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np

from sim.conveyor_scene import OBJECT_HALF_HEIGHT_M, ConveyorSceneEnv

_DATA_DIR = Path(__file__).parent / "data"
_DATASET_DIR = _DATA_DIR / "dataset"

# Realistic operating envelope: the object's real travel range along the
# conveyor's Y axis (starts at y=-0.3, travels at 0.08 m/s, typically
# grasped by ~step 2025 at dt=0.002 -- i.e. y up to roughly -0.3+0.32=0.02,
# padded slightly), X held near the conveyor centerline with a little
# jitter (the object's real X velocity is 0 in configs/conveyor.yaml), Z at
# its resting height on the platform (sim/conveyor_scene.py's platform
# geometry puts a resting box's center at z=0.05).
_X_RANGE = (0.49, 0.51)
_Y_RANGE = (-0.35, 0.05)
_Z = 0.05


def _cube_corners_world(center: np.ndarray, yaw: float) -> np.ndarray:
    """8 corners of the cube in world frame, given its center and a
    rotation about world Z (a fresh-placed/reset object can land at any
    yaw; this is not the tilt-buildup bug tracked separately in the design
    spec's Round 4 -- pure yaw variety is realistic and desired here)."""
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    local_corners = signs * OBJECT_HALF_HEIGHT_M
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return center + local_corners @ rot.T


def _world_to_pixel(point_world, cam_pos, cam_mat, fx, fy, cx, cy):
    """Inverse of run_conveyor_demo.py's _camera_point_to_world: world point
    -> pinhole-camera-frame point (x=right, y=down, z=forward-depth) ->
    pixel (u, v). Returns None if the point is behind the camera plane."""
    local = cam_mat.T @ (point_world - cam_pos)
    point_cam = np.array([local[0], -local[1], -local[2]])
    if point_cam[2] <= 1e-6:
        return None
    u = fx * (point_cam[0] / point_cam[2]) + cx
    v = fy * (point_cam[1] / point_cam[2]) + cy
    return u, v


def sample_frame(env: ConveyorSceneEnv, rng: np.random.Generator, width: int = 64, height: int = 64) -> dict | None:
    """Place the cube at a random pose, render, and compute its exact pixel
    bounding box. Returns None if the box degenerates (fully clipped or
    behind the camera) -- callers should retry with a new sample."""
    obj_jid = env.model.body("conveyor_object").jntadr[0]
    obj_qpos_addr = env.model.jnt_qposadr[obj_jid]

    x = rng.uniform(*_X_RANGE)
    y = rng.uniform(*_Y_RANGE)
    yaw = rng.uniform(0.0, 2 * np.pi)
    center = np.array([x, y, _Z])

    env.data.qpos[obj_qpos_addr : obj_qpos_addr + 3] = center
    half_yaw = yaw / 2.0
    env.data.qpos[obj_qpos_addr + 3 : obj_qpos_addr + 7] = [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)]
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    rgb, _ = env.get_rgbd(width, height)
    cam_id = env.model.camera("wrist_cam").id
    cam_pos = env.data.cam_xpos[cam_id].copy()
    cam_mat = env.data.cam_xmat[cam_id].reshape(3, 3).copy()
    fx, fy, cx, cy = env.camera_intrinsics(width, height)

    corners = _cube_corners_world(center, yaw)
    pixels = [_world_to_pixel(c, cam_pos, cam_mat, fx, fy, cx, cy) for c in corners]
    pixels = [p for p in pixels if p is not None]
    if len(pixels) < 8:
        return None

    us = np.clip([p[0] for p in pixels], 0, width - 1)
    vs = np.clip([p[1] for p in pixels], 0, height - 1)
    x_min, x_max = float(us.min()), float(us.max())
    y_min, y_max = float(vs.min()), float(vs.max())
    if x_max <= x_min or y_max <= y_min:
        return None

    return {"rgb": rgb, "bbox_px": (x_min, y_min, x_max, y_max), "world_pos": center.copy()}


def _write_yolo_label(path: Path, bbox_px, width: int, height: int) -> None:
    x_min, y_min, x_max, y_max = bbox_px
    x_center = (x_min + x_max) / 2.0 / width
    y_center = (y_min + y_max) / 2.0 / height
    box_w = (x_max - x_min) / width
    box_h = (y_max - y_min) / height
    path.write_text(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")


def _generate_split(env, rng, split: str, count: int, width: int, height: int) -> None:
    images_dir = _DATASET_DIR / split / "images"
    labels_dir = _DATASET_DIR / split / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    attempts = 0
    while written < count:
        attempts += 1
        if attempts > count * 10:
            raise RuntimeError(f"Too many degenerate samples generating '{split}' split -- check camera/range setup.")
        sample = sample_frame(env, rng, width, height)
        if sample is None:
            continue
        name = f"{split}_{written:05d}"
        bgr = cv2.cvtColor(sample["rgb"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(images_dir / f"{name}.png"), bgr)
        _write_yolo_label(labels_dir / f"{name}.txt", sample["bbox_px"], width, height)
        written += 1


def _write_data_yaml(width: int, height: int) -> None:
    yaml_text = (
        f"path: {_DATASET_DIR.resolve()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "names:\n"
        "  0: cube\n"
    )
    (_DATASET_DIR / "data.yaml").write_text(yaml_text)


def _write_preview(env, rng, count: int, width: int, height: int) -> None:
    """Draw the exact label boxes on a handful of fresh samples so the
    labeling itself can be visually spot-checked (design spec Section 6)."""
    preview_dir = _DATA_DIR / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    attempts = 0
    while written < count:
        attempts += 1
        if attempts > count * 10:
            break
        sample = sample_frame(env, rng, width, height)
        if sample is None:
            continue
        bgr = cv2.cvtColor(sample["rgb"], cv2.COLOR_RGB2BGR)
        scaled = cv2.resize(bgr, (width * 6, height * 6), interpolation=cv2.INTER_NEAREST)
        x_min, y_min, x_max, y_max = sample["bbox_px"]
        cv2.rectangle(
            scaled,
            (int(x_min * 6), int(y_min * 6)),
            (int(x_max * 6), int(y_max * 6)),
            (0, 255, 0),
            2,
        )
        cv2.imwrite(str(preview_dir / f"preview_{written:03d}.png"), scaled)
        written += 1
    print(f"Wrote {written} preview images to {preview_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-count", type=int, default=800)
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview-count", type=int, default=10)
    args = parser.parse_args()

    env = ConveyorSceneEnv(conveyor_velocity=np.array([0.0, 0.08, 0.0]))
    env.reset()
    rng = np.random.default_rng(args.seed)

    print(f"Generating {args.train_count} train + {args.val_count} val frames at {args.width}x{args.height}...")
    _generate_split(env, rng, "train", args.train_count, args.width, args.height)
    _generate_split(env, rng, "val", args.val_count, args.width, args.height)
    _write_data_yaml(args.width, args.height)
    print(f"Dataset written to {_DATASET_DIR}")

    if args.preview_count > 0:
        _write_preview(env, rng, args.preview_count, args.width, args.height)


if __name__ == "__main__":
    main()
