"""Stage 8: fine-tune a small pretrained YOLO model on the currently
labelled train images, validate on val, evaluate through the shared
geometry/evaluation modules (same as Stage 6's classical baseline, so the
two are directly comparable), and log the run.

Windows note: workers=0, per the brief -- the Ultralytics data loader can
hang/crash on Windows with worker subprocesses.

Run with: .venv\\Scripts\\python.exe scripts\\train_yolo.py [--run-id ID] [--epochs N]
"""

import argparse
import json
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
    precision_recall_f1,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="model-v1-first-fit")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--iou", type=float, default=0.3,
                         help="NMS IoU threshold for val-time prediction (default 0.3, tightened from "
                              "Ultralytics' 0.7 default -- see docs/stage-8-notes.md: the looser default let "
                              "near-duplicate overlapping boxes on the same apple survive, inflating count MAE "
                              "from 6.00 to 2.60 at no F1 cost when tightened)")
    args = parser.parse_args()

    data_yaml = REPO_ROOT / "configs" / "yolo_data.yaml"
    if not data_yaml.exists():
        print(f"Missing {data_yaml} -- run scripts/prepare_yolo_dataset.py first.")
        return 1

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        seed=SEED,
        workers=0,  # Windows: avoid data-loader worker hangs/crashes
        device=0,
        project=str(REPO_ROOT / "runs" / "yolo"),
        name=args.run_id,
        exist_ok=True,
        verbose=False,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nBest weights: {best_weights}")

    # --- evaluate on val through the shared geometry/evaluation modules -----
    val_yolo = YOLO(str(best_weights))
    val_dir = REPO_ROOT / "data" / "derived" / "yolo_dataset" / "images" / "val"
    val_ids = sorted(p.stem for p in val_dir.glob("*.png"))

    gt_val = load_gt(val_ids)
    cam = load_camera()
    intr = Intrinsics.from_dict(cam)
    scale = cam["depth_scale_to_metres"]

    pred_val, depth_lookup = [], {}
    for image_id in val_ids:
        pred = val_yolo.predict(str(val_dir / f"{image_id}.png"), verbose=False, device=0, iou=args.iou)[0]
        for box in pred.boxes:
            x0, y0, x1, y1 = box.xyxy[0].tolist()
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            radius_px = ((x1 - x0) + (y1 - y0)) / 4  # average of half-width, half-height
            pred_val.append(Apple2D(image_id, cx, cy, radius_px, score=float(box.conf[0])))
        depth_lookup[image_id] = load_depth_m(image_id, scale)

    matches_val = match_all(gt_val, pred_val)
    match_df = per_match_errors(gt_val, pred_val, matches_val, depth_lookup, intr)

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    match_df.to_csv(results_dir / f"{args.run_id}_val_matches.csv", index=False)

    summary = summarise(gt_val, pred_val, matches_val, match_df)
    summary_md = summary_to_markdown(summary, f"YOLO {args.model} ({args.run_id})", "val")
    (results_dir / f"{args.run_id}_val_summary.md").write_text(summary_md + "\n", encoding="utf-8")
    print("\n" + summary_md)

    det = summary["detection"]
    record_run(RunRecord(
        run_id=args.run_id,
        stage="stage-8",
        seed=SEED,
        config={"model": args.model, "epochs": args.epochs, "batch": args.batch, "imgsz": 640,
                "nms_iou": args.iou,
                "n_train_images": len(list((REPO_ROOT / "data/derived/yolo_dataset/images/train").glob("*.png")))},
        metrics_summary=f"val P={det['precision']:.3f} R={det['recall']:.3f} F1={det['f1']:.3f} mAP={det['mAP@0.5']:.3f} "
                         f"countMAE={summary['counting']['count_mae']:.2f} countBias={summary['counting']['count_bias']:+.2f}",
    ))

    print(f"\n## Exit check: training run complete, weights at {best_weights}, "
          f"val results saved to results/{args.run_id}_val_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
