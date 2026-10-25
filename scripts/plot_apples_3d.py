"""Stage 10 report figure: a 3D graph of each apple's front-centre point.

Reads results/apples_3d.csv (written by run_pipeline.py -- nothing is
recomputed here) and, for each example tree, draws a 3D scatter plot of
that tree's front-centre points in the camera frame:

  - one marker per valid apple, labelled with its apple_index so it can be
    matched to the 2D overlay; apples with no usable depth are left out and
    counted under the plot.
  - axes in metres: X (right), Y (down), Z (depth away from the camera),
    equal scaling on all three, camera marked at the origin.
  - the same viewing angle for every tree.

Each tree has its own camera frame, so trees are never combined in one
graph. Writes to results/figures/, as PNG (300 dpi) and PDF:
  <tree>_3d             the 3D graph on its own
  <tree>_overlay        the RGB image with each apple's 2D centre + index
  <tree>_3d_overlay     the two side by side

Run with: .venv\\Scripts\\python.exe scripts\\plot_apples_3d.py [--tree-id R01U01 ...]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.config import load_paths  # noqa: E402

# Includes one edge case (R10U14, zero detections), not only "nice" examples (rule 9).
EXAMPLE_TREES = ["R01U01", "R08U01", "R10U14", "R23U14", "R04U13"]
VIEW_ELEV, VIEW_AZIM = 18, -62  # one fixed angle for every tree: depth differences read clearly

POINT_COLOUR = "#2a78d6"   # single series -> one hue
SURFACE = "#fcfcfb"        # chart surface; also the ring around each marker
INK = "#0b0b0b"            # primary text
INK_MUTED = "#52514e"      # secondary text, drop lines, camera marker


def _equal_limits(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> tuple[np.ndarray, float]:
    """Centre and half-width of a cube enclosing every point plus the camera at the
    origin, so all three axes share one scale (no distorted distances)."""
    pts = np.vstack([np.column_stack([xs, ys, zs]), np.zeros((1, 3))])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre = (lo + hi) / 2
    half = max((hi - lo).max() / 2, 0.05) * 1.08
    return centre, half


def draw_3d(ax, tree_apples: pd.DataFrame, tree_id: str) -> None:
    valid = tree_apples[tree_apples["is_valid"]]
    n_invalid = int((~tree_apples["is_valid"]).sum())
    xs, ys, zs = valid["X"].to_numpy(float), valid["Y"].to_numpy(float), valid["Z"].to_numpy(float)

    # Matplotlib's vertical axis is "up"; camera Y points down. Plot (X, Z, Y) and invert
    # the vertical axis so the graph reads like the photo: right is right, down is down,
    # and depth recedes into the page.
    centre, half = _equal_limits(xs, ys, zs)
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[2] - half, centre[2] + half)
    ax.set_zlim(centre[1] + half, centre[1] - half)  # inverted: Y down
    ax.set_box_aspect((1, 1, 1))
    floor_y = centre[1] + half

    for x, y, z in zip(xs, ys, zs):  # thin drop line to the floor makes each point's depth readable
        ax.plot([x, x], [z, z], [y, floor_y], color=INK_MUTED, linewidth=0.5, alpha=0.35)
    ax.scatter(xs, zs, ys, s=36, color=POINT_COLOUR, edgecolors=SURFACE, linewidths=1.5,
               depthshade=False, label="apple front-centre point")
    for idx, x, y, z in zip(valid["apple_index"], xs, ys, zs):
        ax.text(x, z, y, f" {idx}", fontsize=6, color=INK_MUTED)

    ax.scatter([0], [0], [0], marker="^", s=60, color=INK_MUTED, depthshade=False, label="camera (origin)")

    ax.set_xlabel("X, right (m)", color=INK, labelpad=6)
    ax.set_ylabel("Z, depth from camera (m)", color=INK, labelpad=6)
    ax.set_zlabel("Y, down (m)", color=INK, labelpad=6)
    ax.tick_params(labelsize=7, colors=INK_MUTED)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(5))  # fewer ticks: labels don't run into each other
        axis.pane.set_facecolor(SURFACE)
        axis.pane.set_edgecolor("#e4e3df")
        axis._axinfo["grid"]["color"] = "#e4e3df"
        axis._axinfo["grid"]["linewidth"] = 0.5
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    ax.set_title(f"{tree_id}: 3D front-centre points ({len(valid)} apples)", color=INK, fontsize=10)
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor=INK)
    note = f"{n_invalid} detection(s) with no usable depth not shown" if n_invalid else "all detections have usable depth"
    if valid.empty:
        note = "no apples detected on this tree"
    ax.text2D(0.5, -0.12, note, transform=ax.transAxes, ha="center", fontsize=7, color=INK_MUTED)


def draw_overlay(ax, rgb: np.ndarray, tree_apples: pd.DataFrame, tree_id: str) -> None:
    ax.imshow(rgb)
    valid = tree_apples[tree_apples["is_valid"]]
    invalid = tree_apples[~tree_apples["is_valid"]]
    ax.scatter(valid["u"], valid["v"], s=22, color=POINT_COLOUR, edgecolors=SURFACE, linewidths=1.2,
               label="front-centre point")
    if len(invalid):
        ax.scatter(invalid["u"], invalid["v"], s=22, marker="x", color=SURFACE, linewidths=1.2,
                   label="no usable depth")
    for a in tree_apples.itertuples():
        ax.text(a.u + 5, a.v - 5, str(a.apple_index), fontsize=6, color=SURFACE,
                bbox=dict(boxstyle="round,pad=0.1", facecolor=INK, alpha=0.55, linewidth=0))
    ax.set_axis_off()
    ax.set_title(f"{tree_id}: detected apples (index matches the 3D graph)", color=INK, fontsize=10)
    if len(tree_apples):
        ax.legend(loc="lower right", fontsize=7, framealpha=0.85)


def save(fig, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree-id", nargs="+", default=EXAMPLE_TREES)
    args = parser.parse_args()

    manifest = pd.read_csv(load_paths()["manifest_csv"])
    apples_df = pd.read_csv(REPO_ROOT / "results" / "apples_3d.csv")
    out_dir = REPO_ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for tree_id in args.tree_id:
        row = manifest[manifest["tree_id"] == tree_id]
        if row.empty:
            print(f"Tree ID {tree_id!r} not found in manifest.")
            return 1
        rgb = np.array(Image.open(row.iloc[0].rgb_path).convert("RGB"))
        tree_apples = apples_df[apples_df["tree_id"] == tree_id]

        fig = plt.figure(figsize=(6, 6), facecolor=SURFACE)
        draw_3d(fig.add_subplot(projection="3d"), tree_apples, tree_id)
        save(fig, out_dir, f"{tree_id}_3d")

        fig, ax = plt.subplots(figsize=(6.4, 4.8), facecolor=SURFACE)
        draw_overlay(ax, rgb, tree_apples, tree_id)
        save(fig, out_dir, f"{tree_id}_overlay")

        fig = plt.figure(figsize=(13, 5.6), facecolor=SURFACE)
        draw_overlay(fig.add_subplot(1, 2, 1), rgb, tree_apples, tree_id)
        draw_3d(fig.add_subplot(1, 2, 2, projection="3d"), tree_apples, tree_id)
        save(fig, out_dir, f"{tree_id}_3d_overlay")

        print(f"{tree_id}: {len(tree_apples)} detections "
              f"({int(tree_apples['is_valid'].sum())} with usable depth) -> results/figures/{tree_id}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
