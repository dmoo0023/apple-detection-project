"""Stage 9: compare classical colour-masking and YOLO on the SAME val split
through the SAME evaluation code, by distance band, plus speed and a
robustness note -- the decision table checkpoint C3 asks the student to
confirm before the chosen method ever touches the locked test set.

Does not touch the test split at all (rule 3).

Run with: .venv\\Scripts\\python.exe scripts\\stage9_decision_table.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.classical import ClassicalConfig, detect_apples  # noqa: E402
from src.config import load_paths, load_camera  # noqa: E402
from src.evaluate import Apple2D, match_all, per_match_errors, summarise  # noqa: E402
from src.geometry import Intrinsics  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_val_ids() -> list[str]:
    paths = load_paths()
    manifest = pd.read_csv(paths["manifest_csv"])
    splits = json.loads((REPO_ROOT / "configs" / "splits.json").read_text(encoding="utf-8"))
    val_trees = set(splits["splits"]["val"])
    labelled = manifest[manifest["tree_id"].isin(val_trees) & manifest["has_labelme_json"]]
    return [Path(p).stem for p in labelled["rgb_path"]]


def load_gt(image_ids: list[str]) -> list[Apple2D]:
    apples_df = pd.read_csv(load_paths()["derived_dir"] / "apples_2d.csv")
    apples_df = apples_df[apples_df["image_id"].isin(image_ids)]
    return [Apple2D(r.image_id, r.cx, r.cy, r.radius_px) for r in apples_df.itertuples()]


def load_rgb(image_id: str) -> np.ndarray:
    return np.array(Image.open(load_paths()["rgb_dir"] / f"{image_id}.png").convert("RGB"))


def load_depth_m(image_id: str, scale: float) -> np.ndarray:
    paths = load_paths()
    depth_name = image_id.replace("_RGB-", "_D-")
    raw = np.array(Image.open(paths["depth_dir"] / f"{depth_name}.png").convert("L")).astype(float)
    return raw * scale


def main() -> int:
    val_ids = load_val_ids()
    gt = load_gt(val_ids)
    cam = load_camera()
    intr = Intrinsics.from_dict(cam)
    scale = cam["depth_scale_to_metres"]
    depth_lookup = {iid: load_depth_m(iid, scale) for iid in val_ids}

    rows = []

    # --- classical ------------------------------------------------------
    best_config_path = REPO_ROOT / "results" / "baseline_colour_v1_config.json"
    classical_config = ClassicalConfig(**json.loads(best_config_path.read_text(encoding="utf-8")))
    pred_classical = []
    t0 = time.perf_counter()
    for iid in val_ids:
        rgb = load_rgb(iid)
        for d in detect_apples(rgb, classical_config):
            pred_classical.append(Apple2D(iid, d.cx, d.cy, d.radius_px, score=float(d.area_px)))
    classical_time = (time.perf_counter() - t0) / len(val_ids)

    matches = match_all(gt, pred_classical)
    match_df = per_match_errors(gt, pred_classical, matches, depth_lookup, intr)
    summary = summarise(gt, pred_classical, matches, match_df)
    rows.append({"method": "Classical (colour masking)", **summary["detection"],
                 **{f"overall_{k}": v for k, v in summary["overall"].items()},
                 "sec_per_image": classical_time,
                 "robustness": "fails under shade (recall collapses); undersizes circles from leaf occlusion"})

    # --- YOLO -------------------------------------------------------------
    weights = REPO_ROOT / "runs" / "yolo" / "model-v1-40train" / "weights" / "best.pt"
    model = YOLO(str(weights))
    pred_yolo = []
    t0 = time.perf_counter()
    for iid in val_ids:
        pred = model.predict(str(load_paths()["rgb_dir"] / f"{iid}.png"), verbose=False, device=0, iou=0.3)[0]
        for box in pred.boxes:
            x0, y0, x1, y1 = box.xyxy[0].tolist()
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            r = ((x1 - x0) + (y1 - y0)) / 4
            pred_yolo.append(Apple2D(iid, cx, cy, r, score=float(box.conf[0])))
    yolo_time = (time.perf_counter() - t0) / len(val_ids)

    matches = match_all(gt, pred_yolo)
    match_df = per_match_errors(gt, pred_yolo, matches, depth_lookup, intr)
    summary = summarise(gt, pred_yolo, matches, match_df)
    rows.append({"method": "YOLO (model-v1)", **summary["detection"],
                 **{f"overall_{k}": v for k, v in summary["overall"].items()},
                 "sec_per_image": yolo_time,
                 "robustness": "handles shade much better; dominant failure is missed detections on clustered/touching apples; occasional false positives on background structures"})

    df = pd.DataFrame(rows)
    out_dir = REPO_ROOT / "results"
    df.to_csv(out_dir / "stage9_decision_table.csv", index=False)

    lines = ["# Stage 9 decision table (val split, identical evaluation code)\n"]
    lines.append("| Method | Precision | Recall | F1 | mAP@0.5 | Median 3D proxy (cm) | Within 2cm | Within 3cm | sec/image | Robustness |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['mAP@0.5']:.3f} | "
            f"{r['overall_median_error_3d_proxy_cm']:.2f} | {100*r['overall_within_2cm_frac']:.1f}% | "
            f"{100*r['overall_within_3cm_frac']:.1f}% | {r['sec_per_image']:.3f} | {r['robustness']} |"
        )
    lines.append("\n**3D proxy numbers inherit both the checkpoint-C1 depth-scale approximation and the "
                  "checkpoint-C2 surface-point bias — not literal centimetre accuracy against a physical apple, "
                  "see docs/limitations.md.**")
    report = "\n".join(lines)
    (out_dir / "stage9_decision_table.md").write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
