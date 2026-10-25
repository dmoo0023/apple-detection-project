"""Stage 8: one table comparing every experiment run against the baseline.

For each Stage 8 run this script:
  - reads the settings the run ACTUALLY trained with from its own weights
    file (`train_args` inside runs/yolo/<run>/weights/best.pt) -- not from
    the logs, which recorded the requested lr0 even though Ultralytics'
    `optimizer='auto'` replaced it (AdamW, lr0=0.002) for every run;
  - re-evaluates the saved weights on the current val split through the
    shared evaluation code (src/evaluate.py), so every row is scored on the
    same images by the same code;
  - writes one row per run to results/stage8_comparison.csv, and the
    mean/spread of the seed repeats to results/stage8_seed_summary.csv.

No training happens here. exp-06 (segmentation) is scored the same way as
on its branch: each predicted mask becomes a 2D circle (centroid +
area-equivalent radius, in pixels).

Run with: .venv\\Scripts\\python.exe scripts\\stage8_comparison.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from scripts.train_yolo import load_depth_m, load_gt  # noqa: E402
from src.config import load_camera  # noqa: E402
from src.evaluate import Apple2D, match_all, per_match_errors, summarise  # noqa: E402
from src.geometry import Intrinsics  # noqa: E402

BASELINE = "model-v1-40train"
N_TRAIN_IMAGES = 40  # every Stage 8 run trained on the same 40 labelled train images (runs/experiments.csv)
NMS_IOU = 0.3

# run_id, experiment, what changed vs the baseline, lr0 that was requested on the command line
RUNS = [
    (BASELINE, "baseline", "none (baseline: yolov8n pretrained, batch 8, 100 epochs, mosaic on, seed 42)", 0.01),
    ("exp-01-yolov8s", "01 model size", "model: yolov8n -> yolov8s", 0.01),
    ("exp-02-random-init", "02 pretrained vs random", "weights: COCO-pretrained -> random init", 0.01),
    ("exp-03-lr-cosine", "03 LR schedule", "LR schedule: linear -> cosine (requested lr0=0.001 was overridden, so lr0 is unchanged)", 0.001),
    ("exp-04-batch16-ep150", "04 batch and epochs", "batch: 8 -> 16; epochs: 100 -> 150", 0.01),
    ("exp-05-no-mosaic", "05 augmentation", "mosaic augmentation: on -> off", 0.01),
    ("exp-06-segmentation", "06 boxes vs masks", "box detector -> segmentation (circle from mask)", 0.01),
    ("seed-baseline-7", "07 seed repeat", "seed: 42 -> 7 (baseline config)", 0.01),
    ("seed-baseline-123", "07 seed repeat", "seed: 42 -> 123 (baseline config)", 0.01),
    ("seed-nomosaic-7", "07 seed repeat", "seed: 42 -> 7 (no-mosaic config, as exp-05)", 0.01),
    ("seed-nomosaic-123", "07 seed repeat", "seed: 42 -> 123 (no-mosaic config, as exp-05)", 0.01),
]

# seed-repeat groups: the seed-42 run of each config plus its two repeats
SEED_GROUPS = {
    "baseline (mosaic on)": [BASELINE, "seed-baseline-7", "seed-baseline-123"],
    "no mosaic": ["exp-05-no-mosaic", "seed-nomosaic-7", "seed-nomosaic-123"],
}


def mask_polygon_to_circle(xy: np.ndarray) -> tuple[float, float, float]:
    """Centroid + area-equivalent radius of one predicted mask polygon, in pixels
    (same conversion exp-06 used on its branch)."""
    cx, cy = float(xy[:, 0].mean()), float(xy[:, 1].mean())
    x, y = xy[:, 0], xy[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))  # shoelace formula
    radius = float(np.sqrt(area / np.pi)) if area > 0 else 0.0
    return cx, cy, radius


def predict_val(model: YOLO, val_dir: Path, val_ids: list[str], segment: bool) -> tuple[list[Apple2D], float]:
    """Run one model over the val images. Returns its predictions as 2D circles and the
    mean inference time per image, in seconds."""
    preds, elapsed = [], 0.0
    for image_id in val_ids:
        start = time.perf_counter()
        result = model.predict(str(val_dir / f"{image_id}.png"), verbose=False, device=0, iou=NMS_IOU)[0]
        elapsed += time.perf_counter() - start
        if segment:
            if result.masks is None:
                continue
            for mask_xy, box in zip(result.masks.xy, result.boxes):
                if len(mask_xy) < 3:
                    continue  # degenerate mask, no usable circle
                cx, cy, r = mask_polygon_to_circle(mask_xy)
                preds.append(Apple2D(image_id, cx, cy, r, score=float(box.conf[0])))
        else:
            for box in result.boxes:
                x0, y0, x1, y1 = box.xyxy[0].tolist()
                r = ((x1 - x0) + (y1 - y0)) / 4  # average of half-width, half-height
                preds.append(Apple2D(image_id, (x0 + x1) / 2, (y0 + y1) / 2, r, score=float(box.conf[0])))
    return preds, elapsed / len(val_ids)


def main() -> int:
    """Evaluate every Stage 8 run on val and write the comparison and seed-summary CSVs.
    Returns 0, or 1 if a run's weights are missing."""
    val_dir = REPO_ROOT / "data" / "derived" / "yolo_dataset" / "images" / "val"
    val_ids = sorted(p.stem for p in val_dir.glob("*.png"))
    gt = load_gt(val_ids)
    cam = load_camera()
    intr = Intrinsics.from_dict(cam)
    depth_lookup = {iid: load_depth_m(iid, cam["depth_scale_to_metres"]) for iid in val_ids}
    print(f"Val split: {len(val_ids)} labelled images, {len(gt)} labelled apples.")

    rows = []
    for run_id, experiment, changed, lr0_requested in RUNS:
        weights = REPO_ROOT / "runs" / "yolo" / run_id / "weights" / "best.pt"
        if not weights.exists():
            print(f"Missing {weights} -- cannot evaluate {run_id}.")
            return 1
        args = torch.load(weights, map_location="cpu", weights_only=False)["train_args"]
        segment = args.get("task") == "segment"

        preds, sec_per_image = predict_val(YOLO(str(weights)), val_dir, val_ids, segment)
        matches = match_all(gt, preds)
        summary = summarise(gt, preds, matches, per_match_errors(gt, preds, matches, depth_lookup, intr))
        det, cnt, ovr = summary["detection"], summary["counting"], summary["overall"]

        rows.append({
            "run_id": run_id,
            "experiment": experiment,
            "changed_vs_baseline": changed,
            # settings actually used (from the weights file)
            "model": args.get("model"),
            "pretrained_weights": str(args.get("model", "")).endswith(".pt"),
            "task": args.get("task"),
            "epochs": args.get("epochs"),
            "batch": args.get("batch"),
            "imgsz": args.get("imgsz"),
            "optimizer": args.get("optimizer"),
            "lr0_used": args.get("lr0"),
            "lr0_requested": lr0_requested,
            "cos_lr": args.get("cos_lr"),
            "mosaic": args.get("mosaic"),
            "seed": args.get("seed"),
            "n_train_images": N_TRAIN_IMAGES,
            "n_val_images": len(val_ids),
            "nms_iou": NMS_IOU,
            # val metrics (src/evaluate.py)
            "precision": round(det["precision"], 4),
            "recall": round(det["recall"], 4),
            "f1": round(det["f1"], 4),
            "map50": round(det["mAP@0.5"], 4),
            "tp": det["tp"],
            "fp": det["fp"],
            "fn": det["fn"],
            "count_mae": round(cnt["count_mae"], 3),
            "count_bias": round(cnt["count_bias"], 3),
            "median_error_2d_px": round(ovr["median_error_2d_px"], 3),
            "median_error_3d_proxy_cm": round(ovr["median_error_3d_proxy_cm"], 3),
            "within_2cm_frac": round(ovr["within_2cm_frac"], 4),
            "sec_per_image": round(sec_per_image, 4),
        })
        print(f"{run_id:22} F1={det['f1']:.3f} mAP={det['mAP@0.5']:.3f} countMAE={cnt['count_mae']:.2f}")

    df = pd.DataFrame(rows)
    base = df.set_index("run_id").loc[BASELINE]
    for metric in ("f1", "map50", "count_mae"):
        df[f"delta_{metric}_vs_baseline"] = (df[metric] - base[metric]).round(4)

    results_dir = REPO_ROOT / "results"
    df.to_csv(results_dir / "stage8_comparison.csv", index=False)

    seed_rows = []
    for config, run_ids in SEED_GROUPS.items():
        g = df[df["run_id"].isin(run_ids)]
        seed_rows.append({
            "config": config,
            "runs": " ".join(run_ids),
            "seeds": " ".join(str(s) for s in g["seed"]),
            "f1_mean": round(g["f1"].mean(), 4), "f1_std": round(g["f1"].std(), 4),
            "map50_mean": round(g["map50"].mean(), 4), "map50_std": round(g["map50"].std(), 4),
            "count_mae_mean": round(g["count_mae"].mean(), 3), "count_mae_std": round(g["count_mae"].std(), 3),
        })
    pd.DataFrame(seed_rows).to_csv(results_dir / "stage8_seed_summary.csv", index=False)

    print("\nWrote results/stage8_comparison.csv and results/stage8_seed_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
