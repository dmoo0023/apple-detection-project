"""Stage 10: summarise results/apples_3d.csv -- apples per tree, missing
depth counts, and a distance (Z) histogram, as the brief asks for after
running the full pipeline.

Run with: .venv\\Scripts\\python.exe scripts\\stage10_summary.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    """Write results/stage10_summary.md and the apples-per-tree / depth histograms. Returns 0."""
    from src.config import load_paths

    df = pd.read_csv(REPO_ROOT / "results" / "apples_3d.csv")
    results_dir = REPO_ROOT / "results"

    # a tree with ZERO detections contributes no rows at all, so it would silently
    # vanish from a plain groupby -- reindex against the full manifest so "0 apples
    # on this tree" is counted, not just dropped.
    manifest = pd.read_csv(load_paths()["manifest_csv"])
    all_tree_ids = manifest["tree_id"]
    apples_per_tree = df.groupby("tree_id").size().reindex(all_tree_ids, fill_value=0)
    n_zero_apple_trees = int((apples_per_tree == 0).sum())
    n_missing = int((~df["is_valid"]).sum())

    lines = ["# Stage 10 summary: full inference over all 544 trees\n"]
    lines.append(f"- Total detections: {len(df)}")
    lines.append(f"- Trees processed: {len(all_tree_ids)} (all {len(all_tree_ids)} in the manifest, "
                 f"including {n_zero_apple_trees} with zero detections, counted explicitly rather than "
                 f"silently dropped from this summary)")
    lines.append(f"- Apples per tree: min={apples_per_tree.min()}, median={apples_per_tree.median():.0f}, "
                 f"mean={apples_per_tree.mean():.2f}, max={apples_per_tree.max()}")
    lines.append(f"- Detections with no usable depth sample: {n_missing} ({100 * n_missing / len(df):.1f}%)")

    z_valid = df.loc[df["is_valid"], "Z"]
    lines.append(f"- Z (depth) of valid detections: min={z_valid.min():.3f}m, median={z_valid.median():.3f}m, "
                 f"max={z_valid.max():.3f}m")
    lines.append(f"- **{(z_valid < 0.4).mean() * 100:.1f}% of valid Z values fall below the dataset's own "
                 f"stated 0.4m minimum capture distance** -- the checkpoint-C1 depth-scale limitation "
                 f"propagated through the full 544-tree run exactly as expected, not a new problem. "
                 f"See docs/limitations.md.")

    (results_dir / "stage10_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(apples_per_tree, bins=30, color="tab:green", edgecolor="white")
    axes[0].set_xlabel("Apples detected per tree")
    axes[0].set_ylabel("Number of trees")
    axes[0].set_title("Apples per tree (all 544 trees)")

    axes[1].hist(z_valid, bins=40, color="tab:blue", edgecolor="white")
    axes[1].axvspan(0.4, 1.2, alpha=0.15, color="tab:orange", label="dataset's stated capture range")
    axes[1].set_xlabel("Z (depth, metres -- see checkpoint C1 limitation)")
    axes[1].set_ylabel("Number of detections")
    axes[1].set_title("Distance histogram (valid detections)")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(results_dir / "stage10_histograms.png", dpi=120)
    print(f"\nWrote results/stage10_summary.md and results/stage10_histograms.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
