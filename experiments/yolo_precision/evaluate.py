"""Go/no-go comparison: does the trained YOLO detector localize the cube
more precisely than the current RGB-threshold segmentation?

Run (as a module, from the repo root -- a direct script-path invocation puts
this file's own directory, not the repo root, at sys.path[0], so `perception`,
`run_conveyor_demo`, and `sim` wouldn't be importable):
    uv run --group yolo-precision python -m experiments.yolo_precision.evaluate
Requires train.py to have produced data/cube_detector.pt.

Success criteria (design spec Section 4): candidate mean 3D error must be
both >=40% lower than baseline AND under 1cm absolute to justify a Phase 2
pipeline-integration spec.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from experiments.yolo_precision.generate_dataset import sample_frame
from perception.camera import CameraIntrinsics
from perception.segment import segment_object_centroid
from run_conveyor_demo import _camera_point_to_world
from sim.conveyor_scene import ConveyorSceneEnv, OBJECT_HALF_HEIGHT_M

_DATA_DIR = Path(__file__).parent / "data"
_COLOR_LOWER = (150, 0, 0)
_COLOR_UPPER = (255, 80, 80)


def _yolo_bbox_center_to_3d(model, rgb, depth, intrinsics, cam_pos, cam_mat):
    results = model.predict(source=rgb, verbose=False)[0]
    if len(results.boxes) == 0:
        return None
    confs = results.boxes.conf.cpu().numpy()
    best = int(np.argmax(confs))
    x_min, y_min, x_max, y_max = results.boxes.xyxy[best].cpu().numpy()
    u = (x_min + x_max) / 2.0
    v = (y_min + y_max) / 2.0
    y0, y1 = int(max(0, y_min)), int(min(depth.shape[0], y_max + 1))
    x0, x1 = int(max(0, x_min)), int(min(depth.shape[1], x_max + 1))
    region_rgb = rgb[y0:y1, x0:x1]
    region_depth = depth[y0:y1, x0:x1]
    # Combine YOLO's box (tight, reliable localization -- 0 misses vs. the
    # baseline's 9) with the baseline's own proven depth-sampling approach
    # (only average depth over pixels that actually match the cube's
    # color), scoped to the YOLO box instead of the whole frame. Two prior
    # attempts (full-box mean, then median-distance inlier filtering) both
    # failed because a rotated square's own bounding box can have enough
    # of its area be background that purely-geometric filtering can't
    # separate cube pixels from background pixels -- color can.
    lower = np.array(_COLOR_LOWER, dtype=np.uint8)
    upper = np.array(_COLOR_UPPER, dtype=np.uint8)
    mask = np.all((region_rgb >= lower) & (region_rgb <= upper), axis=-1)
    if mask.sum() == 0:
        return None
    z = float(region_depth[mask].mean()) + OBJECT_HALF_HEIGHT_M
    point_cam = intrinsics.deproject(u, v, z)
    return _camera_point_to_world(point_cam, cam_pos, cam_mat), (x_min, y_min, x_max, y_max)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--seed", type=int, default=999)  # distinct from generate_dataset's default train/val seed
    parser.add_argument("--save-samples", type=int, default=10)
    args = parser.parse_args()

    weights = _DATA_DIR / "cube_detector.pt"
    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found -- run train.py first.")
    model = YOLO(str(weights))

    env = ConveyorSceneEnv(conveyor_velocity=np.array([0.0, 0.08, 0.0]))
    env.reset()
    rng = np.random.default_rng(args.seed)
    cam_id = env.model.camera("wrist_cam").id
    fx, fy, cx, cy = env.camera_intrinsics(args.width, args.height)
    intrinsics = CameraIntrinsics(fx, fy, cx, cy)

    baseline_errors, candidate_errors = [], []
    baseline_misses, candidate_misses = 0, 0
    samples_dir = _DATA_DIR / "eval_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    evaluated = 0
    attempts = 0
    while evaluated < args.count:
        attempts += 1
        if attempts > args.count * 10:
            raise RuntimeError("Too many degenerate samples during evaluation -- check camera/range setup.")
        sample = sample_frame(env, rng, args.width, args.height)
        if sample is None:
            continue
        evaluated += 1

        rgb = sample["rgb"]
        _, depth = env.get_rgbd(args.width, args.height)
        cam_pos = env.data.cam_xpos[cam_id].copy()
        cam_mat = env.data.cam_xmat[cam_id].reshape(3, 3).copy()
        truth = sample["world_pos"]

        baseline_point_cam = segment_object_centroid(
            rgb, depth, intrinsics, _COLOR_LOWER, _COLOR_UPPER, depth_bias=OBJECT_HALF_HEIGHT_M
        )
        if baseline_point_cam is None:
            baseline_misses += 1
            baseline_world = None
        else:
            baseline_world = _camera_point_to_world(baseline_point_cam, cam_pos, cam_mat)
            baseline_errors.append(float(np.linalg.norm(baseline_world - truth)))

        yolo_result = _yolo_bbox_center_to_3d(model, rgb, depth, intrinsics, cam_pos, cam_mat)
        if yolo_result is None:
            candidate_misses += 1
            candidate_world, candidate_box = None, None
        else:
            candidate_world, candidate_box = yolo_result
            candidate_errors.append(float(np.linalg.norm(candidate_world - truth)))

        if saved < args.save_samples:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            scaled = cv2.resize(bgr, (args.width * 6, args.height * 6), interpolation=cv2.INTER_NEAREST)
            if candidate_box is not None:
                x_min, y_min, x_max, y_max = candidate_box
                cv2.rectangle(scaled, (int(x_min * 6), int(y_min * 6)), (int(x_max * 6), int(y_max * 6)), (0, 255, 0), 2)
            cv2.putText(
                scaled,
                f"base_err={baseline_errors[-1]:.4f}" if baseline_world is not None else "base_err=MISS",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
            )
            cv2.putText(
                scaled,
                f"yolo_err={candidate_errors[-1]:.4f}" if candidate_world is not None else "yolo_err=MISS",
                (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
            )
            cv2.imwrite(str(samples_dir / f"eval_{saved:03d}.png"), scaled)
            saved += 1

    baseline_mean = float(np.mean(baseline_errors)) if baseline_errors else float("nan")
    baseline_max = float(np.max(baseline_errors)) if baseline_errors else float("nan")
    candidate_mean = float(np.mean(candidate_errors)) if candidate_errors else float("nan")
    candidate_max = float(np.max(candidate_errors)) if candidate_errors else float("nan")

    print(f"\n=== Evaluated {evaluated} frames ===")
    print(f"Baseline (RGB threshold): mean_err={baseline_mean:.4f}m max_err={baseline_max:.4f}m misses={baseline_misses}")
    print(f"Candidate (YOLO):         mean_err={candidate_mean:.4f}m max_err={candidate_max:.4f}m misses={candidate_misses}")

    if baseline_mean > 0 and not np.isnan(candidate_mean):
        improvement = 1.0 - (candidate_mean / baseline_mean)
        print(f"\nImprovement: {improvement * 100:.1f}%")
        go = improvement >= 0.40 and candidate_mean < 0.01
        print(f"GO/NO-GO (>=40% improvement AND <1cm absolute): {'GO' if go else 'NO-GO'}")
    else:
        print("\nCould not compute a verdict (insufficient successful detections).")

    print(f"\nSample comparison frames written to {samples_dir}")


if __name__ == "__main__":
    main()
