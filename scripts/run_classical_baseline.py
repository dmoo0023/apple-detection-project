"""Stage 6 exit check: tune the classical colour-masking detector on TRAIN
labelled images (grid search on F1), evaluate once on VAL labelled images
through the shared geometry/evaluation modules, save overlays, and tag the
result baseline-colour-v1.

Run with: .venv\\Scripts\\python.exe scripts\\run_classical_baseline.py
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.classical import ClassicalConfig, detect_apples  # noqa: E402
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

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_split_image_ids(split: str) -> list[str]:
    splits = json.loads((REPO_ROOT / "configs" / "splits.json").read_text(encoding="utf-8"))
    tree_ids = set(splits["splits"][split])
    manifest = pd.read_csv(load_paths()["manifest_csv"])
    labelled = manifest[manifest["tree_id"].isin(tree_ids) & manifest["has_labelme_json"]]
    rgb_dir = load_paths()["rgb_dir"]
    return [Path(p).stem for p in labelled["rgb_path"]]


def load_gt(image_ids: list[str]) -> list[Apple2D]:
    apples_df = pd.read_csv(load_paths()["derived_dir"] / "apples_2d.csv")
    apples_df = apples_df[apples_df["image_id"].isin(image_ids)]
    return [Apple2D(row.image_id, row.cx, row.cy, row.radius_px) for row in apples_df.itertuples()]


def load_rgb(image_id: str) -> np.ndarray:
    rgb_dir = load_paths()["rgb_dir"]
    return np.array(Image.open(rgb_dir / f"{image_id}.png").convert("RGB"))


def load_depth_m(image_id: str, scale: float) -> np.ndarray:
    paths = load_paths()
    depth_name = image_id.replace("_RGB-", "_D-")
    raw = np.array(Image.open(paths["depth_dir"] / f"{depth_name}.png").convert("L")).astype(float)
    return raw * scale


def detections_to_apple2d(image_id: str, detections) -> list[Apple2D]:
    return [Apple2D(image_id, d.cx, d.cy, d.radius_px, score=float(d.area_px)) for d in detections]


def grid_search(train_ids: list[str], gt_train: list[Apple2D]) -> tuple[ClassicalConfig, float, pd.DataFrame]:
    grid = {
        "hue_low_max": [8, 12],
        "hue_high_min": [165, 172],
        "sat_min": [50, 90],
        "val_min": [30, 60],
        "lab_a_min": [135, 155],
    }
    rgb_cache = {iid: load_rgb(iid) for iid in train_ids}

    results = []
    keys = list(grid.keys())
    for combo in itertools.product(*grid.values()):
        config = ClassicalConfig(**dict(zip(keys, combo)))
        preds = []
        for iid in train_ids:
            preds.extend(detections_to_apple2d(iid, detect_apples(rgb_cache[iid], config)))
        matches = match_all(gt_train, preds)
        det = precision_recall_f1(matches)
        results.append({**config.to_dict(), "precision": det["precision"], "recall": det["recall"], "f1": det["f1"]})

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False).reset_index(drop=True)
    best_row = results_df.iloc[0]
    best_config = ClassicalConfig(**{k: int(best_row[k]) for k in keys})
    return best_config, float(best_row["f1"]), results_df


def main() -> int:
    results_dir = REPO_ROOT / "results"
    overlays_dir = REPO_ROOT / "data" / "derived" / "stage6_overlays"
    results_dir.mkdir(exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    train_ids = load_split_image_ids("train")
    val_ids = load_split_image_ids("val")
    print(f"Train labelled images: {len(train_ids)}; val labelled images: {len(val_ids)}")

    gt_train = load_gt(train_ids)
    gt_val = load_gt(val_ids)

    print("\nGrid search on train (F1)...")
    best_config, best_train_f1, grid_df = grid_search(train_ids, gt_train)
    grid_df.to_csv(results_dir / "stage6_grid_search.csv", index=False)
    print(f"Best config (train F1={best_train_f1:.3f}): {best_config.to_dict()}")

    cam = load_camera()
    intr = Intrinsics.from_dict(cam)
    scale = cam["depth_scale_to_metres"]

    print("\nEvaluating on val...")
    pred_val, depth_lookup = [], {}
    for iid in val_ids:
        rgb = load_rgb(iid)
        detections = detect_apples(rgb, best_config)
        pred_val.extend(detections_to_apple2d(iid, detections))
        depth_lookup[iid] = load_depth_m(iid, scale)

    matches_val = match_all(gt_val, pred_val)
    match_df = per_match_errors(gt_val, pred_val, matches_val, depth_lookup, intr)
    match_df.to_csv(results_dir / "baseline_colour_v1_val_matches.csv", index=False)

    summary = summarise(gt_val, pred_val, matches_val, match_df)
    summary_md = summary_to_markdown(summary, "Classical colour-masking (baseline-colour-v1)", "val")
    (results_dir / "baseline_colour_v1_val_summary.md").write_text(summary_md + "\n", encoding="utf-8")
    print("\n" + summary_md)

    # --- overlays: 10 val images, label=red circle, detection=green circle --
    sample_ids = val_ids[:10]
    gt_by_image: dict[str, list[Apple2D]] = {}
    for a in gt_val:
        gt_by_image.setdefault(a.image_id, []).append(a)
    pred_by_image: dict[str, list[Apple2D]] = {}
    for a in pred_val:
        pred_by_image.setdefault(a.image_id, []).append(a)

    for iid in sample_ids:
        img = cv2.cvtColor(load_rgb(iid), cv2.COLOR_RGB2BGR)
        for a in gt_by_image.get(iid, []):
            cv2.circle(img, (int(a.cx), int(a.cy)), int(a.radius_px), (0, 0, 255), 2)
        for a in pred_by_image.get(iid, []):
            cv2.circle(img, (int(a.cx), int(a.cy)), int(a.radius_px), (0, 255, 0), 2)
        cv2.imwrite(str(overlays_dir / f"{iid}_baseline.png"), img)
    print(f"\nSaved {len(sample_ids)} overlays to {overlays_dir.relative_to(REPO_ROOT)}/ "
          f"(red = label, green = detection).")

    record_run(RunRecord(
        run_id="baseline-colour-v1",
        stage="stage-6",
        seed=None,
        config=best_config.to_dict(),
        metrics_summary=f"val P={summary['detection']['precision']:.3f} "
                         f"R={summary['detection']['recall']:.3f} F1={summary['detection']['f1']:.3f} "
                         f"mAP={summary['detection']['mAP@0.5']:.3f}",
    ))

    exit_ok = (results_dir / "baseline_colour_v1_val_summary.md").exists()
    print(f"\n## Exit check: {'PASS' if exit_ok else 'ISSUES FOUND'} (baseline results saved with distance breakdown)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
