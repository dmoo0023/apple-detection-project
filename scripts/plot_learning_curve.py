"""Stage 8 item 5: plot performance against number of labelled training
images, using every stage-8 (YOLO) run logged in runs/experiments.csv.

Run with: .venv\\Scripts\\python.exe scripts\\plot_learning_curve.py
"""

import ast
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_metrics_summary(s: str) -> dict:
    out = {}
    for key, pattern in [("precision", r"P=([\d.]+)"), ("recall", r"R=([\d.]+)"),
                          ("f1", r"F1=([\d.]+)"), ("map50", r"mAP=([\d.]+)")]:
        m = re.search(pattern, s)
        if m:
            out[key] = float(m.group(1))
    return out


def parse_count_mae(run_id: str) -> float | None:
    summary_path = REPO_ROOT / "results" / f"{run_id}_val_summary.md"
    if not summary_path.exists():
        return None
    text = summary_path.read_text(encoding="utf-8")
    m = re.search(r"MAE=([\d.]+)", text)
    return float(m.group(1)) if m else None


def main() -> int:
    csv_path = REPO_ROOT / "runs" / "experiments.csv"
    df = pd.read_csv(csv_path)
    stage8 = df[df["stage"] == "stage-8"].copy()

    rows = []
    for row in stage8.itertuples():
        try:
            config = ast.literal_eval(row.config)
        except (ValueError, SyntaxError):
            continue
        if "n_train_images" not in config:
            continue
        metrics = parse_metrics_summary(row.metrics_summary)
        metrics["n_train_images"] = config["n_train_images"]
        metrics["run_id"] = row.run_id
        metrics["count_mae"] = parse_count_mae(row.run_id)
        rows.append(metrics)

    if not rows:
        print("No stage-8 runs with n_train_images found in runs/experiments.csv yet.")
        return 1

    curve_df = pd.DataFrame(rows).drop_duplicates(subset="run_id").sort_values("n_train_images")
    out_csv = REPO_ROOT / "results" / "learning_curve.csv"
    curve_df.to_csv(out_csv, index=False)
    print(curve_df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(curve_df["n_train_images"], curve_df["f1"], marker="o", label="F1")
    axes[0].plot(curve_df["n_train_images"], curve_df["map50"], marker="s", label="mAP@0.5")
    axes[0].set_xlabel("Number of labelled train images")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Detection performance vs. train label count")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if curve_df["count_mae"].notna().any():
        axes[1].plot(curve_df["n_train_images"], curve_df["count_mae"], marker="o", color="tab:red")
        axes[1].set_xlabel("Number of labelled train images")
        axes[1].set_ylabel("Count MAE (apples/image)")
        axes[1].set_title("Apple-count error vs. train label count")
        axes[1].grid(alpha=0.3)
    else:
        axes[1].axis("off")

    fig.suptitle(f"Learning curve -- {len(curve_df)} point(s) so far "
                 f"(more labels -> more points; see docs/stage-8-notes.md)")
    fig.tight_layout()
    out_png = REPO_ROOT / "results" / "learning_curve.png"
    fig.savefig(out_png, dpi=120)
    print(f"\nWrote {out_csv.relative_to(REPO_ROOT)} and {out_png.relative_to(REPO_ROOT)}")

    if len(curve_df) < 3:
        print(f"\nNote: only {len(curve_df)} point(s) -- too few to read a trend from yet "
              f"(could be noise). Stage 8's learning curve needs several more label-count "
              f"milestones before drawing conclusions from the shape.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
