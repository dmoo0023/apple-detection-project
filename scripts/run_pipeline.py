"""Stage 10: full inference pipeline. For a given tree ID or every tree in
the dataset, detects apples (YOLO, the model-v1 config chosen at checkpoint
C3), samples depth, and reports each apple's 3D surface point (see
src/geometry.py's scope note: a surface point per checkpoint C2's decision,
not a corrected centre -- the column names below are X, Y, Z per the
brief's spec, but they are near-surface positions, not centres).

Writes results/apples_3d.csv: one row per detected apple --
tree_id, apple_index, u, v, radius_px, X, Y, Z, depth_pixels_used,
is_valid, confidence. A detection with no usable depth still gets a row
(X/Y/Z/depth_pixels_used empty), never silently dropped or zeroed (rule 4).

Run with:
  .venv\\Scripts\\python.exe scripts\\run_pipeline.py --tree-id R01U01
  .venv\\Scripts\\python.exe scripts\\run_pipeline.py --all
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_paths, load_camera  # noqa: E402
from src.geometry import Intrinsics, surface_point_3d  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CHOSEN_RUN_ID = "model-v1-40train"  # the config tagged model-v1, confirmed at checkpoint C3


def load_depth_m(depth_path: Path, scale: float) -> np.ndarray:
    raw = np.array(Image.open(depth_path).convert("L")).astype(float)
    return raw * scale


def process_tree(tree_id: str, rgb_path: Path, depth_path: Path, model: YOLO, intr: Intrinsics,
                  scale: float, iou: float) -> list[dict]:
    depth_m = load_depth_m(depth_path, scale)
    pred = model.predict(str(rgb_path), verbose=False, device=0, iou=iou)[0]

    rows = []
    for i, box in enumerate(pred.boxes):
        x0, y0, x1, y1 = box.xyxy[0].tolist()
        u, v = (x0 + x1) / 2, (y0 + y1) / 2
        radius_px = ((x1 - x0) + (y1 - y0)) / 4
        confidence = float(box.conf[0])

        result = surface_point_3d(u, v, radius_px, depth_m, intr)
        row = {
            "tree_id": tree_id,
            "apple_index": i,
            "u": u,
            "v": v,
            "radius_px": radius_px,
            "X": result.point_m[0] if result.is_valid else None,
            "Y": result.point_m[1] if result.is_valid else None,
            "Z": result.point_m[2] if result.is_valid else None,
            "depth_pixels_used": result.depth_sample.valid_pixel_count,
            "is_valid": result.is_valid,
            "confidence": confidence,
        }
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tree-id", help="process a single tree, e.g. R01U01")
    group.add_argument("--all", action="store_true", help="process all 544 trees in the manifest")
    parser.add_argument("--iou", type=float, default=0.3)
    args = parser.parse_args()

    paths = load_paths()
    cam = load_camera()
    intr = Intrinsics.from_dict(cam)
    scale = cam["depth_scale_to_metres"]

    weights = REPO_ROOT / "runs" / "yolo" / CHOSEN_RUN_ID / "weights" / "best.pt"
    model = YOLO(str(weights))

    manifest = pd.read_csv(paths["manifest_csv"])
    if args.tree_id:
        manifest = manifest[manifest["tree_id"] == args.tree_id]
        if manifest.empty:
            print(f"Tree ID {args.tree_id!r} not found in manifest.")
            return 1

    all_rows = []
    for i, row in enumerate(manifest.itertuples()):
        rgb_path, depth_path = Path(row.rgb_path), Path(row.depth_path)
        tree_rows = process_tree(row.tree_id, rgb_path, depth_path, model, intr, scale, args.iou)
        all_rows.extend(tree_rows)
        if args.all and (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(manifest)} trees...")

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_df = pd.DataFrame(all_rows)
    out_path = results_dir / "apples_3d.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(out_df)} apple rows across {manifest['tree_id'].nunique()} trees to "
          f"{out_path.relative_to(REPO_ROOT)}")

    if len(out_df):
        apples_per_tree = out_df.groupby("tree_id").size()
        n_missing_depth = (~out_df["is_valid"]).sum()
        print(f"Apples per tree: min={apples_per_tree.min()}, median={apples_per_tree.median():.0f}, "
              f"max={apples_per_tree.max()}, mean={apples_per_tree.mean():.2f}")
        print(f"Detections with no usable depth: {n_missing_depth} / {len(out_df)} "
              f"({100 * n_missing_depth / len(out_df):.1f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
