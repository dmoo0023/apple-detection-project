"""Gather the figures and tables most useful for the written report into one
folder, `report/`, grouped by report section, with a README index explaining
each file.

Everything here is COPIED from existing outputs (nothing is recomputed),
apart from three small summary tables built from existing results: the data
split, val vs test results for the chosen model, and an error-type
breakdown. Re-run after any stage is rerun to refresh the folder.

Run with: .venv\\Scripts\\python.exe scripts\\collect_report_assets.py
"""

import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.config import load_paths  # noqa: E402

OUT = REPO_ROOT / "report"
DERIVED = REPO_ROOT / "data" / "derived"
RESULTS = REPO_ROOT / "results"
CHOSEN_RUN = REPO_ROOT / "runs" / "yolo" / "model-v1-40train"
EXAMPLE_TREES = ["R01U01", "R08U01", "R23U14", "R04U13", "R10U14"]

# section folder -> list of (source file, description). Glob patterns pick the first N sorted matches.
SECTIONS: dict[str, list[tuple[str, str]]] = {
    "01_data": [
        ("data/derived/overlays/*_overlay.png|2", "Depth image over the RGB image, checking that depth is aligned to colour (Stage 1)."),
        ("data/derived/stage2_overlays/*_labels.png|2", "Labelled apples: labelme circle (red), derived YOLO box (green), centre (blue) (Stage 2)."),
    ],
    "02_classical_baseline": [
        ("data/derived/stage6_overlays/R01U01_Summer_RGB-12-40-11_baseline.png", "Classical colour-masking detections on a val image (Stage 6)."),
        ("data/derived/stage6_overlays/R01U04_Summer_RGB-12-41-9_baseline.png", "Classical detections on a denser val image (Stage 6)."),
        ("results/stage6_grid_search.csv", "Threshold grid search on train: every config tried and its F1."),
        ("results/baseline_colour_v1_val_summary.md", "Classical baseline val results (detection, count and error tables)."),
    ],
    "03_yolo_training": [
        ("runs/yolo/model-v1-40train/results.png", "Training/validation loss and metric curves over 100 epochs for the chosen model."),
        ("runs/yolo/model-v1-40train/BoxPR_curve.png", "Precision-recall curve of the chosen model (Ultralytics validation)."),
        ("runs/yolo/model-v1-40train/BoxF1_curve.png", "F1 against confidence threshold for the chosen model."),
        ("runs/yolo/model-v1-40train/confusion_matrix_normalized.png", "Normalised confusion matrix (apple vs background)."),
        ("runs/yolo/model-v1-40train/val_batch0_pred.jpg", "Ultralytics val batch: predictions with confidences."),
        ("runs/yolo/model-v1-40train/val_batch0_labels.jpg", "The same val batch with ground-truth labels, for comparison."),
        ("data/derived/stage8_overlays/model-v1-40train/R01U01_Summer_RGB-12-40-11_yolo.png", "Chosen YOLO model's detections (green) vs labels (red) on a val image."),
        ("data/derived/stage8_overlays/model-v1-40train/R01U04_Summer_RGB-12-41-9_yolo.png", "Chosen YOLO model's detections vs labels on a denser val image."),
        ("results/learning_curve.png", "Learning curve: F1/mAP and count error against number of labelled train images."),
        ("results/learning_curve.csv", "Learning-curve numbers (all points on the same 26-image val set)."),
        ("results/stage8_comparison.csv", "Stage 8 experiments: settings actually used and val metrics for every run."),
        ("results/stage8_seed_summary.csv", "Seed repeats: mean and spread of F1, mAP and count error."),
    ],
    "04_error_analysis": [
        ("data/derived/error_analysis/model-v1-40train/R01U04_Summer_RGB-12-41-9_errors.png", "Chosen model's errors on a val image: labelled apples (thin green), missed apples (red), false positives (yellow)."),
        ("data/derived/error_analysis/model-v1-40train/R08U01_Summer_RGB-13-25-43_errors.png", "Error overlay on a second val image."),
        ("data/derived/error_analysis/model-v1-40train/R09U03_Summer_RGB-13-31-23_errors.png", "Error overlay on a third val image."),
        ("data/derived/error_analysis/model-v1-40train/errors.csv", "Every false negative and false positive of the chosen model on val."),
    ],
    "05_method_comparison": [
        ("results/stage9_decision_table.md", "Classical vs YOLO on val with identical evaluation (Stage 9 decision table)."),
        ("results/stage9_decision_table.csv", "Decision table as CSV."),
    ],
    "06_final_test": [
        ("results/final_test_eval_summary.md", "One-time locked test evaluation of the chosen model."),
    ],
    "07_3d_output": [
        *[(f"results/figures/{t}_3d_overlay.png", f"{t}: photo with numbered apples beside the 3D graph of their front-centre points.") for t in EXAMPLE_TREES],
        *[(f"results/figures/{t}_3d_overlay.pdf", f"{t}: same figure as vector PDF.") for t in EXAMPLE_TREES],
        *[(f"results/figures/{t}_3d.png", f"{t}: 3D graph of the front-centre points on its own.") for t in EXAMPLE_TREES],
        ("results/stage10_histograms.png", "All 544 trees: apples per tree and depth (Z) of every detected apple."),
        ("results/stage10_summary.md", "Summary of the full 544-tree run (counts, missing depth, depth range)."),
        ("docs/stage-4-worked-example.md", "Worked example: one apple's 3D front-centre point calculated by hand."),
    ],
}


def expand(pattern: str) -> list[Path]:
    """Resolve 'path' or 'glob|N' (first N sorted matches) to existing files."""
    glob, _, n = pattern.partition("|")
    matches = sorted(REPO_ROOT.glob(glob))
    return matches[: int(n)] if n else matches


def split_table() -> pd.DataFrame:
    """Train/val/test sizes from the frozen configs/splits.json."""
    splits = json.loads((REPO_ROOT / "configs" / "splits.json").read_text(encoding="utf-8"))["splits"]
    manifest = pd.read_csv(load_paths()["manifest_csv"])
    apples = pd.read_csv(DERIVED / "apples_2d.csv")
    apples["tree_id"] = apples["image_id"].str.extract(r"^(R\d+U\d+)", expand=False)
    rows = []
    for name in ("train", "val", "test"):
        trees = splits[name]
        rows.append({
            "split": name,
            "orchard_rows": " ".join(str(r) for r in sorted({int(re.match(r"R(\d+)", t).group(1)) for t in trees})),
            "trees": len(trees),
            "trees_pct": round(100 * len(trees) / len(manifest), 1),
            "labelled_images": int(manifest[manifest["tree_id"].isin(trees)]["has_labelme_json"].sum()),
            "labelled_apples": int(apples["tree_id"].isin(trees).sum()),
        })
    return pd.DataFrame(rows)


def val_vs_test_table() -> pd.DataFrame:
    """Chosen model's val vs locked-test metrics, parsed from the two summary files."""
    rows = []
    for split, path in (("val", RESULTS / "model-v1-40train_val_summary.md"),
                        ("test", RESULTS / "final_test_eval_summary.md")):
        text = path.read_text(encoding="utf-8")
        get = lambda key: float(re.search(rf"{key}=([-+\d.]+)", text).group(1))  # noqa: E731
        rows.append({"split": split, "precision": get("precision"), "recall": get("recall"), "f1": get("F1"),
                     "map50": get(r"mAP@0\.5"), "count_mae": get("MAE"), "count_bias": get("bias")})
    return pd.DataFrame(rows)


def error_breakdown() -> pd.DataFrame:
    """Count of the chosen model's val errors by type, with mean confidence for false positives."""
    errors = pd.read_csv(DERIVED / "error_analysis" / "model-v1-40train" / "errors.csv")
    out = errors.groupby("type").agg(count=("type", "size"), mean_radius_px=("radius_px", "mean"),
                                      mean_confidence=("score", "mean")).round(3).reset_index()
    return out


def main() -> int:
    """Rebuild report/ from scratch and write its README index. Returns 0, or 1 if a source is missing."""
    if OUT.exists():
        shutil.rmtree(OUT)
    index = ["# Report figures and tables\n",
             "Collected by `scripts/collect_report_assets.py` from the pipeline's outputs. "
             "All 3D numbers inherit the approximate depth scale and camera intrinsics "
             "(see `docs/limitations.md`), and 3D error is a proxy (no physical ground truth).\n"]

    missing = []
    for section, items in SECTIONS.items():
        (OUT / section).mkdir(parents=True)
        index.append(f"\n## {section}\n")
        for pattern, desc in items:
            files = expand(pattern)
            if not files:
                missing.append(pattern)
                continue
            for src in files:
                shutil.copy2(src, OUT / section / src.name)
                index.append(f"- `{section}/{src.name}` — {desc}")

    new_tables = [
        ("01_data", "data_split.csv", split_table(), "Train/val/test split: orchard rows, trees, labelled images and apples per split."),
        ("06_final_test", "val_vs_test.csv", val_vs_test_table(), "Chosen model on val vs the locked test set."),
        ("04_error_analysis", "error_breakdown.csv", error_breakdown(), "Error counts by type, with mean size and confidence."),
    ]
    index.append("\n## Tables built for the report\n")
    for section, name, df, desc in new_tables:
        df.to_csv(OUT / section / name, index=False)
        index.append(f"- `{section}/{name}` — {desc}")

    (OUT / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    n_files = sum(1 for p in OUT.rglob("*") if p.is_file())
    print(f"Wrote {n_files} files to {OUT.relative_to(REPO_ROOT)}/")
    if missing:
        print("Missing sources (skipped):\n  " + "\n  ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
