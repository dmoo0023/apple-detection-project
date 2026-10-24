"""Stage 8 exp-06: train a YOLO SEGMENTATION model (predicts a polygon mask
per apple, not just a box) and compare its CENTRE accuracy against the
detection (box) baseline -- the brief's specific ask for this experiment.

A segmentation prediction's circle (cx, cy, radius) is recovered from its
mask polygon: centre = polygon centroid, radius = the equivalent radius of
a circle with the same area as the polygon (radius = sqrt(area / pi)) --
a different, independent way to estimate the apple's extent than a box's
half-width/half-height average, and arguably closer to the true shape
since a mask traces the apple's outline, not just its bounding box.

Run with: .venv\\Scripts\\python.exe scripts\\train_yolo_seg.py [--run-id ID] [--epochs N]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_paths, load_camera  # noqa: E402
from src.evaluate import (  # noqa: E402
    Apple2D,
    match_all,
    per_match_errors,
    summarise,
    summary_to_markdown,
)
from src.geometry import Intrinsics  # noqa: E402
from src.tracking import RunRecord, record_run  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 42


def load_gt(image_ids: list[str]) -> list[Apple2D]:
    apples_df = pd.read_csv(load_paths()["derived_dir"] / "apples_2d.csv")
    apples_df = apples_df[apples_df["image_id"].isin(image_ids)]
    return [Apple2D(row.image_id, row.cx, row.cy, row.radius_px) for row in apples_df.itertuples()]


def load_depth_m(image_id: str, scale: float) -> np.ndarray:
    paths = load_paths()
    depth_name = image_id.replace("_RGB-", "_D-")
    raw = np.array(Image.open(paths["depth_dir"] / f"{depth_name}.png").convert("L")).astype(float)
    return raw * scale


def mask_polygon_to_circle(xy: np.ndarray) -> tuple[float, float, float]:
    """Centroid + area-equivalent radius of one predicted mask polygon, in pixels."""
    cx, cy = float(xy[:, 0].mean()), float(xy[:, 1].mean())
    # shoelace formula for polygon area
    x, y = xy[:, 0], xy[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    radius = float(np.sqrt(area / np.pi)) if area > 0 else 0.0
    return cx, cy, radius


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="exp-06-segmentation")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--model", default="yolov8n-seg.pt")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--iou", type=float, default=0.3)
    args = parser.parse_args()

    data_yaml = REPO_ROOT / "configs" / "yolo_data_seg.yaml"
    if not data_yaml.exists():
        print(f"Missing {data_yaml} -- run scripts/prepare_yolo_seg_dataset.py first.")
        return 1

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        seed=SEED,
        workers=0,
        device=0,
        project=str(REPO_ROOT / "runs" / "yolo"),
        name=args.run_id,
        exist_ok=True,
        verbose=False,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nBest weights: {best_weights}")

    val_yolo = YOLO(str(best_weights))
    val_dir = REPO_ROOT / "data" / "derived" / "yolo_dataset_seg" / "images" / "val"
    val_ids = sorted(p.stem for p in val_dir.glob("*.png"))

    gt_val = load_gt(val_ids)
    cam = load_camera()
    intr = Intrinsics.from_dict(cam)
    scale = cam["depth_scale_to_metres"]

    pred_val, depth_lookup = [], {}
    n_no_mask = 0
    for image_id in val_ids:
        pred = val_yolo.predict(str(val_dir / f"{image_id}.png"), verbose=False, device=0, iou=args.iou)[0]
        if pred.masks is not None:
            for mask_xy, box in zip(pred.masks.xy, pred.boxes):
                if len(mask_xy) < 3:
                    n_no_mask += 1
                    continue
                cx, cy, r = mask_polygon_to_circle(mask_xy)
                pred_val.append(Apple2D(image_id, cx, cy, r, score=float(box.conf[0])))
        depth_lookup[image_id] = load_depth_m(image_id, scale)
    if n_no_mask:
        print(f"Note: {n_no_mask} predictions had a degenerate (<3 point) mask and were skipped.")

    matches_val = match_all(gt_val, pred_val)
    match_df = per_match_errors(gt_val, pred_val, matches_val, depth_lookup, intr)

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    match_df.to_csv(results_dir / f"{args.run_id}_val_matches.csv", index=False)

    summary = summarise(gt_val, pred_val, matches_val, match_df)
    summary_md = summary_to_markdown(summary, f"YOLO-seg {args.model} ({args.run_id})", "val")
    (results_dir / f"{args.run_id}_val_summary.md").write_text(summary_md + "\n", encoding="utf-8")
    print("\n" + summary_md)

    det = summary["detection"]
    record_run(RunRecord(
        run_id=args.run_id,
        stage="stage-8",
        seed=SEED,
        config={"model": args.model, "epochs": args.epochs, "batch": args.batch, "imgsz": 640,
                "nms_iou": args.iou, "task": "segmentation",
                "n_train_images": len(list((REPO_ROOT / "data/derived/yolo_dataset_seg/images/train").glob("*.png")))},
        metrics_summary=f"val P={det['precision']:.3f} R={det['recall']:.3f} F1={det['f1']:.3f} mAP={det['mAP@0.5']:.3f} "
                         f"countMAE={summary['counting']['count_mae']:.2f} countBias={summary['counting']['count_bias']:+.2f}",
    ))

    print(f"\n## Exit check: training run complete, weights at {best_weights}, "
          f"val results saved to results/{args.run_id}_val_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
