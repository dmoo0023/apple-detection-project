"""Stage 8: fine-tune a small pretrained YOLO model on the currently
labelled train images, validate on val, evaluate through the shared
geometry/evaluation modules (same as Stage 6's classical baseline, so the
two are directly comparable), and log the run.

Windows note: workers=0, per the brief -- the Ultralytics data loader can
hang/crash on Windows with worker subprocesses.

Run with: .venv\\Scripts\\python.exe scripts\\train_yolo.py [--run-id ID] [--epochs N]
"""

import argparse
import ast
import json
import sys
from pathlib import Path

import pandas as pd
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
from src.pairing import load_depth_m  # noqa: E402
from src.tracking import RunRecord, record_run  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEED = 42


def load_gt(image_ids: list[str]) -> list[Apple2D]:
    """Load the labelled circles for the given images from apples_2d.csv."""
    apples_df = pd.read_csv(load_paths()["derived_dir"] / "apples_2d.csv")
    apples_df = apples_df[apples_df["image_id"].isin(image_ids)]
    return [Apple2D(row.image_id, row.cx, row.cy, row.radius_px) for row in apples_df.itertuples()]


def logged_n_train_images(run_id: str) -> int | None:
    """The train-image count this run was trained on, from its latest training entry in
    runs/experiments.csv (an --eval-only re-evaluation doesn't change it)."""
    csv_path = REPO_ROOT / "runs" / "experiments.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    n = None
    for config_str in df.loc[df["run_id"] == run_id, "config"]:
        try:
            config = ast.literal_eval(config_str)
        except (ValueError, SyntaxError):
            continue
        n = config.get("n_train_images", n)
    return n


def main() -> int:
    """Train a YOLO model (or, with --eval-only, reuse its saved weights) and evaluate it on val. Returns 0, or 1 if inputs are missing."""
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
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="training seed (for seed-repeat experiments)")
    parser.add_argument("--lr0", type=float, default=0.01, help="initial learning rate (Ultralytics default 0.01)")
    parser.add_argument("--cos-lr", action="store_true", help="use cosine LR schedule instead of linear decay")
    parser.add_argument("--mosaic", type=float, default=1.0, help="mosaic augmentation probability (0 disables it)")
    parser.add_argument("--eval-only", action="store_true",
                         help="skip training; re-evaluate this run's existing runs/yolo/<run-id>/weights/best.pt on val")
    args = parser.parse_args()

    data_yaml = REPO_ROOT / "configs" / "yolo_data.yaml"
    if not data_yaml.exists():
        print(f"Missing {data_yaml} -- run scripts/prepare_yolo_dataset.py first.")
        return 1

    if args.eval_only:
        best_weights = REPO_ROOT / "runs" / "yolo" / args.run_id / "weights" / "best.pt"
        if not best_weights.exists():
            print(f"Missing {best_weights} -- nothing to re-evaluate for run {args.run_id!r}.")
            return 1
        return evaluate_on_val(args, best_weights)

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        seed=args.seed,
        lr0=args.lr0,
        cos_lr=args.cos_lr,
        mosaic=args.mosaic,
        workers=0,  # Windows: avoid data-loader worker hangs/crashes
        device=0,
        project=str(REPO_ROOT / "runs" / "yolo"),
        name=args.run_id,
        exist_ok=True,
        verbose=False,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nBest weights: {best_weights}")
    return evaluate_on_val(args, best_weights)


def evaluate_on_val(args: argparse.Namespace, best_weights: Path) -> int:
    """Evaluate a trained run on val through the shared geometry/evaluation modules,
    write its results/<run-id>_val_* files and log it in runs/experiments.csv."""
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
    if args.eval_only:
        config = {"model": args.model, "nms_iou": args.iou, "note": "re-evaluated existing weights (--eval-only)"}
        n_train = logged_n_train_images(args.run_id)
        if n_train is not None:  # keeps this run on the learning curve, now on the current val set
            config["n_train_images"] = n_train
    else:
        config = {"model": args.model, "epochs": args.epochs, "batch": args.batch, "imgsz": 640,
                  "nms_iou": args.iou, "lr0": args.lr0, "cos_lr": args.cos_lr, "mosaic": args.mosaic,
                  "n_train_images": len(list((REPO_ROOT / "data/derived/yolo_dataset/images/train").glob("*.png")))}
    record_run(RunRecord(
        run_id=args.run_id,
        stage="stage-8",
        seed=args.seed,
        config=config,
        metrics_summary=f"val P={det['precision']:.3f} R={det['recall']:.3f} F1={det['f1']:.3f} mAP={det['mAP@0.5']:.3f} "
                         f"countMAE={summary['counting']['count_mae']:.2f} countBias={summary['counting']['count_bias']:+.2f}",
    ))

    print(f"\n## Exit check: training run complete, weights at {best_weights}, "
          f"val results saved to results/{args.run_id}_val_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
