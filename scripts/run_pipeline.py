"""Stage 10: full inference pipeline. For a given tree ID or every tree in
the dataset, detects apples (YOLO, the model-v1 config chosen at checkpoint
C3), samples depth, and back-projects each apple's 3D front-centre point
(src/geometry.py's apple_front_point_3d).

Writes results/apples_3d.csv: one row per detected apple --
tree_id, apple_index, u, v, X, Y, Z (front-centre point, metres),
depth_pixels_used, is_valid, invalid_reason, confidence. A detection with no
usable depth still gets a row (X/Y/Z empty, is_valid False, the reason
given), never silently dropped or zeroed (rule 4).

File paths come from the Stage 1 manifest (src/pairing.py). A tree whose
depth file is missing or ambiguous is still detected, and every detection is
marked invalid with reason "no depth file"; a tree with no usable RGB file
cannot be detected at all, so it is skipped and listed at the end. One bad
tree never stops the run.

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
from src.config import load_camera  # noqa: E402
from src.geometry import Intrinsics, apple_front_point_3d  # noqa: E402
from src.pairing import load_manifest  # noqa: E402

NO_DEPTH_FILE = "no depth file"

REPO_ROOT = Path(__file__).resolve().parent.parent
CHOSEN_RUN_ID = "model-v1-40train"  # the config tagged model-v1, confirmed at checkpoint C3


def load_depth_m(depth_path: Path, scale: float) -> np.ndarray:
    """Load a depth PNG and convert it to metres with `scale`."""
    raw = np.array(Image.open(depth_path).convert("L")).astype(float)
    return raw * scale


def process_tree(tree_id: str, rgb_path: Path, depth_path: Path | None, model: YOLO, intr: Intrinsics,
                  scale: float, iou: float) -> list[dict]:
    """Detect apples in one tree's RGB image and compute each one's 3D front-centre point.

    `depth_path` is None when the tree has no usable depth file: apples are still
    detected, and each row is marked invalid with reason NO_DEPTH_FILE.

    Returns one dict per detection: tree_id, apple_index, u, v (pixels), X, Y, Z
    (metres, None if depth is unusable), depth_pixels_used, is_valid,
    invalid_reason, confidence.
    """
    depth_m = load_depth_m(depth_path, scale) if depth_path is not None else None
    pred = model.predict(str(rgb_path), verbose=False, device=0, iou=iou)[0]

    rows = []
    for i, box in enumerate(pred.boxes):
        x0, y0, x1, y1 = box.xyxy[0].tolist()
        u, v = (x0 + x1) / 2, (y0 + y1) / 2
        radius_px = ((x1 - x0) + (y1 - y0)) / 4  # only sizes the depth-sampling disc
        confidence = float(box.conf[0])

        if depth_m is None:
            point, used, valid, reason = None, 0, False, NO_DEPTH_FILE
        else:
            result = apple_front_point_3d(u, v, radius_px, depth_m, intr)
            point, used, valid, reason = (result.point_m, result.depth_sample.valid_pixel_count,
                                          result.is_valid, result.reason)
        rows.append({
            "tree_id": tree_id,
            "apple_index": i,
            "u": u,
            "v": v,
            "X": point[0] if valid else None,
            "Y": point[1] if valid else None,
            "Z": point[2] if valid else None,
            "depth_pixels_used": used,
            "is_valid": valid,
            "invalid_reason": reason or "",
            "confidence": confidence,
        })
    return rows


def main() -> int:
    """Run the pipeline on one tree or all 544 and write results/apples_3d.csv. Returns 0, or 1 for an unknown tree ID."""
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tree-id", help="process a single tree, e.g. R01U01")
    group.add_argument("--all", action="store_true", help="process all 544 trees in the manifest")
    parser.add_argument("--iou", type=float, default=0.3)
    args = parser.parse_args()

    cam = load_camera()
    intr = Intrinsics.from_dict(cam)
    scale = cam["depth_scale_to_metres"]

    weights = REPO_ROOT / "runs" / "yolo" / CHOSEN_RUN_ID / "weights" / "best.pt"
    model = YOLO(str(weights))

    manifest = load_manifest()
    if args.tree_id:
        manifest = manifest[manifest["tree_id"] == args.tree_id]
        if manifest.empty:
            print(f"Tree ID {args.tree_id!r} not found in manifest.")
            return 1

    all_rows, skipped = [], []
    for i, row in enumerate(manifest.itertuples()):
        if not row.rgb_path:  # nothing to detect apples in
            skipped.append(f"{row.tree_id} ({row.pair_status})")
            continue
        depth_path = Path(row.depth_path) if row.depth_path else None
        if depth_path is None:
            print(f"  {row.tree_id}: no usable depth file ({row.pair_status}); detections will be marked invalid")
        tree_rows = process_tree(row.tree_id, Path(row.rgb_path), depth_path, model, intr, scale, args.iou)
        all_rows.extend(tree_rows)
        if args.all and (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(manifest)} trees...")
    if skipped:
        print(f"Skipped {len(skipped)} tree(s) with no usable RGB image: {', '.join(skipped)}")

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_df = pd.DataFrame(all_rows)
    out_path = results_dir / "apples_3d.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(out_df)} apple rows across {manifest['tree_id'].nunique() - len(skipped)} trees to "
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
