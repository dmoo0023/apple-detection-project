"""Figures and tables for the report's optimisation section.

Covers the three things that were tuned in this project:
  1. Classical colour thresholds (Stage 6): a 2^5 full-factorial grid search
     on train, shown as a main-effects plot (mean F1 at each level of each
     threshold) from results/stage6_grid_search.csv.
  2. YOLO training settings (Stage 8): each one-change experiment against
     the baseline, with the baseline's seed-to-seed range shown as the
     noise band, from results/stage8_comparison.csv.
  3. Detection thresholds of the chosen model: a sweep of the NMS IoU
     threshold and the confidence threshold on VAL (never test), run here
     through src/evaluate.py. This shows how the fixed settings (NMS IoU
     0.3, confidence 0.25) sit on the curve; it does not change them --
     the test set has already been used, so nothing is retuned.

Writes to results/figures/optimisation_*.png and results/optimisation_*.csv.

Run with: .venv\\Scripts\\python.exe scripts\\optimisation_figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from ultralytics import YOLO  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from scripts.train_yolo import load_gt  # noqa: E402
from src.evaluate import (  # noqa: E402
    Apple2D,
    average_precision_at_iou,
    count_accuracy,
    match_all,
    precision_recall_f1,
)

RESULTS = REPO_ROOT / "results"
FIGURES = RESULTS / "figures"
CHOSEN_RUN = "model-v1-40train"
CHOSEN_NMS_IOU, CHOSEN_CONF = 0.3, 0.25  # 0.25 = Ultralytics' default confidence threshold, never changed
NMS_IOU_VALUES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
CONF_VALUES = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]

POINT = "#2a78d6"       # single-series colour
BASELINE = "#0b0b0b"    # baseline / chosen value marker (ink)
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax) -> None:
    """Recessive axes: light grid, no top/right spines, muted tick labels."""
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.set_facecolor(SURFACE)


def save(fig, stem: str) -> None:
    """Save a figure as PNG (300 dpi) and PDF in results/figures/, then close it."""
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


# --- 1. classical grid search --------------------------------------------------

def classical_main_effects() -> pd.DataFrame:
    """Mean train F1 at each level of each grid-searched threshold, plus a plot."""
    grid = pd.read_csv(RESULTS / "stage6_grid_search.csv")
    varied = [c for c in grid.columns[:-3] if grid[c].nunique() > 1]
    best = grid.loc[grid["f1"].idxmax()]

    rows = []
    for param in varied:
        for level, g in grid.groupby(param):
            rows.append({"parameter": param, "level": level, "mean_f1": round(g["f1"].mean(), 4),
                         "min_f1": round(g["f1"].min(), 4), "max_f1": round(g["f1"].max(), 4),
                         "n_configs": len(g), "best_config_level": level == best[param]})
    effects = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, len(varied), figsize=(2.3 * len(varied), 3.2), sharey=True, facecolor=SURFACE)
    for ax, param in zip(axes, varied):
        sub = effects[effects["parameter"] == param]
        x = range(len(sub))
        for i, (_, lv) in enumerate(grid.groupby(param)):  # each config as a faint dot
            ax.scatter([i] * len(lv), lv["f1"], s=10, color=POINT, alpha=0.25, linewidths=0)
        ax.plot(list(x), sub["mean_f1"], color=POINT, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_xticks(list(x), [str(v) for v in sub["level"]])
        ax.set_xlim(-0.5, len(sub) - 0.5)
        ax.set_title(param, fontsize=9, color=INK)
        style(ax)
    axes[0].set_ylabel("F1 on train", color=INK)
    fig.suptitle(f"Classical colour thresholds: {len(grid)}-config grid search (line = mean F1 at each level)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, "optimisation_classical_grid")
    effects.to_csv(RESULTS / "optimisation_classical_grid_effects.csv", index=False)
    return effects


# --- 2. Stage 8 experiments ------------------------------------------------------

def stage8_plot() -> pd.DataFrame:
    """Dot plot of each Stage 8 experiment's F1 and mAP@0.5, with the baseline's seed range shaded."""
    comp = pd.read_csv(RESULTS / "stage8_comparison.csv")
    seeds = comp[comp["run_id"].isin([CHOSEN_RUN, "seed-baseline-7", "seed-baseline-123"])]
    exps = comp[~comp["run_id"].str.startswith("seed-")].iloc[::-1]  # baseline at the top after reversing
    labels = [("baseline" if r == CHOSEN_RUN else c.split(" (")[0]) for r, c in zip(exps["run_id"], exps["changed_vs_baseline"])]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, facecolor=SURFACE)
    for ax, metric, name in ((axes[0], "f1", "F1"), (axes[1], "map50", "mAP@0.5")):
        lo, hi = seeds[metric].min(), seeds[metric].max()
        ax.axvspan(lo, hi, color=GRID, alpha=0.9, label="baseline seed-to-seed range (3 seeds)")
        is_base = exps["run_id"] == CHOSEN_RUN
        y = range(len(exps))
        ax.scatter(exps[metric][~is_base], [i for i, b in zip(y, is_base) if not b], s=60, color=POINT,
                   edgecolors=SURFACE, linewidths=1.5, zorder=3, label="experiment (one change)")
        ax.scatter(exps[metric][is_base], [i for i, b in zip(y, is_base) if b], s=70, marker="D",
                   color=BASELINE, edgecolors=SURFACE, linewidths=1.5, zorder=3, label="baseline (seed 42)")
        for i, v in zip(y, exps[metric]):
            ax.annotate(f"{v:.3f}", (v, i), xytext=(7, -3), textcoords="offset points", fontsize=7, color=INK_MUTED)
        ax.set_yticks(list(y), labels)
        ax.set_xlabel(f"{name} on val (26 images)", color=INK)
        style(ax)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, fontsize=8, frameon=False)  # below, clear of the data
    fig.suptitle("YOLO training settings: one change at a time vs the baseline", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "optimisation_stage8_experiments")
    return comp


# --- 3. threshold sweep ------------------------------------------------------------

def threshold_sweep() -> pd.DataFrame:
    """Sweep NMS IoU (at conf 0.25) and confidence (at NMS IoU 0.3) for the chosen model on val."""
    val_dir = REPO_ROOT / "data" / "derived" / "yolo_dataset" / "images" / "val"
    val_ids = sorted(p.stem for p in val_dir.glob("*.png"))
    gt = load_gt(val_ids)
    model = YOLO(str(REPO_ROOT / "runs" / "yolo" / CHOSEN_RUN / "weights" / "best.pt"))

    def predict(iou: float) -> list[Apple2D]:
        """All boxes at confidence >= the lowest swept value, after NMS at `iou`."""
        preds = []
        for image_id in val_ids:
            r = model.predict(str(val_dir / f"{image_id}.png"), verbose=False, device=0,
                              iou=iou, conf=min(CONF_VALUES))[0]
            for box in r.boxes:
                x0, y0, x1, y1 = box.xyxy[0].tolist()
                preds.append(Apple2D(image_id, (x0 + x1) / 2, (y0 + y1) / 2, ((x1 - x0) + (y1 - y0)) / 4,
                                     score=float(box.conf[0])))
        return preds

    def score(preds: list[Apple2D]) -> dict:
        """Detection and counting metrics for one set of predictions."""
        det = precision_recall_f1(match_all(gt, preds))
        cnt = count_accuracy(gt, preds)
        return {"precision": round(det["precision"], 4), "recall": round(det["recall"], 4),
                "f1": round(det["f1"], 4), "map50": round(average_precision_at_iou(gt, preds, iou_threshold=0.5), 4),
                "count_mae": round(cnt["count_mae"], 3), "count_bias": round(cnt["count_bias"], 3)}

    rows = []
    for iou in NMS_IOU_VALUES:
        # dropping boxes below a confidence AFTER NMS gives the same result as predicting at that
        # confidence: a box can only be suppressed by a higher-scoring box, which also survives the cut
        preds = [p for p in predict(iou) if p.score >= CHOSEN_CONF]
        rows.append({"sweep": "nms_iou", "nms_iou": iou, "conf": CHOSEN_CONF, **score(preds)})
    base_preds = predict(CHOSEN_NMS_IOU)
    for conf in CONF_VALUES:
        rows.append({"sweep": "conf", "nms_iou": CHOSEN_NMS_IOU, "conf": conf,
                     **score([p for p in base_preds if p.score >= conf])})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(RESULTS / "optimisation_threshold_sweep.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6), sharex="col", facecolor=SURFACE)
    for col, (name, xcol, chosen, xlabel) in enumerate((
            ("nms_iou", "nms_iou", CHOSEN_NMS_IOU, "NMS IoU threshold (confidence 0.25)"),
            ("conf", "conf", CHOSEN_CONF, "Confidence threshold (NMS IoU 0.3)"))):
        sub = sweep[sweep["sweep"] == name]
        for row, (metric, ylabel) in enumerate((("f1", "F1 on val"), ("count_mae", "Count MAE (apples/image)"))):
            ax = axes[row, col]
            ax.plot(sub[xcol], sub[metric], color=POINT, linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
            ax.axvline(chosen, color=BASELINE, linewidth=1, linestyle="--")
            val = sub.loc[(sub[xcol] - chosen).abs().idxmin(), metric]
            ax.annotate(f"used: {chosen} ({val:.3g})", (chosen, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(4, -10), textcoords="offset points", fontsize=7, color=INK)  # top of panel, clear of the line
            ax.set_ylabel(ylabel, color=INK, fontsize=9)
            if row == 1:
                ax.set_xlabel(xlabel, color=INK, fontsize=9)
            style(ax)
    fig.suptitle(f"Detection thresholds for the chosen model ({CHOSEN_RUN}), swept on val", fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, "optimisation_threshold_sweep")
    return sweep


def summary_table(effects: pd.DataFrame, comp: pd.DataFrame, sweep: pd.DataFrame) -> None:
    """One row per optimised quantity: method, search space, criterion, data and result."""
    grid = pd.read_csv(RESULTS / "stage6_grid_search.csv")
    seed_summary = pd.read_csv(RESULTS / "stage8_seed_summary.csv").set_index("config")
    base_s, nomo_s = seed_summary.loc["baseline (mosaic on)"], seed_summary.loc["no mosaic"]
    exps = comp[comp["run_id"].str.startswith("exp-")]
    nms = sweep[sweep["sweep"] == "nms_iou"].set_index("nms_iou")
    conf = sweep[sweep["sweep"] == "conf"].set_index("conf")
    best_conf = conf["f1"].idxmax()
    rows = [
        {"what_was_optimised": "Classical colour thresholds (HSV + Lab)",
         "method": f"Full-factorial grid search, {len(grid)} configs (5 thresholds x 2 levels)",
         "criterion": "F1 on train", "data": "train (labelled images only)",
         "result": f"best train F1 {grid['f1'].max():.3f} (range {grid['f1'].min():.3f}-{grid['f1'].max():.3f}); "
                   f"val F1 0.548 with the best config"},
        {"what_was_optimised": "YOLO training settings (model size, pretraining, LR schedule, batch/epochs, augmentation, boxes vs masks)",
         "method": f"{len(exps)} one-change experiments vs a baseline, plus 3-seed repeats of the 2 best configs",
         "criterion": "F1 and mAP@0.5 on val", "data": "train for fitting, val (26 images) for scoring",
         "result": f"random init and segmentation clearly worse; model size, LR schedule and batch/epochs within "
                   f"seed noise. Over 3 seeds, no mosaic F1 {nomo_s['f1_mean']:.3f}+/-{nomo_s['f1_std']:.3f} vs baseline "
                   f"{base_s['f1_mean']:.3f}+/-{base_s['f1_std']:.3f}, but mAP@0.5 {nomo_s['map50_mean']:.3f} vs "
                   f"{base_s['map50_mean']:.3f}; baseline kept for its higher mAP (a close call)"},
        {"what_was_optimised": "NMS IoU threshold",
         "method": f"Sweep {NMS_IOU_VALUES[0]}-{NMS_IOU_VALUES[-1]} (prediction-time, no retraining)",
         "criterion": "Count MAE and F1 on val", "data": "val",
         "result": f"0.3 used: count MAE {nms.loc[0.3, 'count_mae']:.2f} vs {nms.loc[0.7, 'count_mae']:.2f} at "
                   f"Ultralytics' default 0.7; F1 {nms.loc[0.3, 'f1']:.3f} vs {nms.loc[0.7, 'f1']:.3f}"},
        {"what_was_optimised": "Confidence threshold",
         "method": f"Sweep {CONF_VALUES[0]}-{CONF_VALUES[-1]} (prediction-time, no retraining)",
         "criterion": "F1 on val", "data": "val",
         "result": f"0.25 (Ultralytics default) used: F1 {conf.loc[0.25, 'f1']:.3f}; best on val {best_conf} "
                   f"(F1 {conf.loc[best_conf, 'f1']:.3f}). Not retuned after the locked test evaluation"},
        {"what_was_optimised": "Optimiser and learning rate",
         "method": "Not tuned: Ultralytics optimizer='auto' chose AdamW, lr0=0.002 for every run",
         "criterion": "-", "data": "-",
         "result": "requested lr0 values were overridden; exp-03 therefore tests the cosine schedule only"},
    ]
    pd.DataFrame(rows).to_csv(RESULTS / "optimisation_summary.csv", index=False)


def main() -> int:
    """Build every optimisation figure and table. Returns 0."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    effects = classical_main_effects()
    comp = stage8_plot()
    sweep = threshold_sweep()
    summary_table(effects, comp, sweep)
    print(sweep.to_string(index=False))
    print("\nWrote results/figures/optimisation_*.png/.pdf and results/optimisation_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
