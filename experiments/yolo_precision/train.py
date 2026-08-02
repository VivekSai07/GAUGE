"""Fine-tune a small pretrained YOLO model on the synthetic cube dataset.

Run: uv run --group yolo-precision python experiments/yolo_precision/train.py
Requires experiments/yolo_precision/generate_dataset.py to have been run
first (produces data/dataset/data.yaml).
"""
import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

_DATA_DIR = Path(__file__).parent / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    data_yaml = _DATA_DIR / "dataset" / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} not found -- run generate_dataset.py first."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available -- this experiment expects the RTX PRO 2000 "
            "GPU. Aborting rather than silently training on CPU."
        )

    model = YOLO("yolo11n.pt")
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        project=str(_DATA_DIR / "runs"),
        name="cube_detector",
        exist_ok=True,
        workers=2,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    dest = _DATA_DIR / "cube_detector.pt"
    shutil.copy(best_weights, dest)

    metrics = model.val(data=str(data_yaml))
    print(f"\nFinal validation mAP50: {metrics.box.map50:.4f}")
    print(f"Final validation mAP50-95: {metrics.box.map:.4f}")
    print(f"Weights saved to {dest}")


if __name__ == "__main__":
    main()
