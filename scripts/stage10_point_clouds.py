"""Stage 10: save a PLY point cloud (full scene, back-projected from depth,
coloured from RGB) with each detected apple's surface point marked as a
small sphere, for a handful of example trees -- plus a matching 2D overlay.

Chosen trees deliberately include one edge case (R10U14, zero detections)
rather than only "nice" examples, consistent with reporting honestly
(rule 9).

Run with: .venv\\Scripts\\python.exe scripts\\stage10_point_clouds.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.config import load_paths, load_camera  # noqa: E402
from src.geometry import Intrinsics, backproject  # noqa: E402

EXAMPLE_TREES = ["R01U01", "R08U01", "R10U14", "R23U14", "R04U13"]
MARKER_RADIUS_M = 0.01  # 1cm sphere, just to make a point visible/clickable in a viewer


def scene_point_cloud(rgb: np.ndarray, depth_m: np.ndarray, intr: Intrinsics) -> o3d.geometry.PointCloud:
    h, w = depth_m.shape
    vv, uu = np.mgrid[0:h, 0:w]
    valid = depth_m > 0
    us, vs, zs = uu[valid], vv[valid], depth_m[valid]
    xs = (us - intr.cx) * zs / intr.fx
    ys = (vs - intr.cy) * zs / intr.fy
    points = np.stack([xs, ys, zs], axis=1)
    colors = rgb[valid].astype(float) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def main() -> int:
    paths = load_paths()
    cam = load_camera()
    intr = Intrinsics.from_dict(cam)
    scale = cam["depth_scale_to_metres"]

    manifest = pd.read_csv(paths["manifest_csv"])
    apples_df = pd.read_csv(REPO_ROOT / "results" / "apples_3d.csv")

    out_dir = REPO_ROOT / "results" / "point_clouds"
    out_dir.mkdir(parents=True, exist_ok=True)

    for tree_id in EXAMPLE_TREES:
        row = manifest[manifest["tree_id"] == tree_id].iloc[0]
        rgb = np.array(Image.open(row.rgb_path).convert("RGB"))
        depth_m = np.array(Image.open(row.depth_path).convert("L")).astype(float) * scale

        pcd = scene_point_cloud(rgb, depth_m, intr)
        tree_apples = apples_df[apples_df["tree_id"] == tree_id]

        geometries = [pcd]
        n_valid = 0
        for apple in tree_apples.itertuples():
            if not apple.is_valid:
                continue
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=MARKER_RADIUS_M)
            sphere.translate([apple.X, apple.Y, apple.Z])
            sphere.paint_uniform_color([1.0, 0.0, 0.0])  # red marker, stands out against the scene
            geometries.append(sphere)
            n_valid += 1

        combined = geometries[0]
        for g in geometries[1:]:
            combined += g.sample_points_uniformly(200) if isinstance(g, o3d.geometry.TriangleMesh) else g

        ply_path = out_dir / f"{tree_id}.ply"
        o3d.io.write_point_cloud(str(ply_path), combined)

        # matching 2D overlay
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for apple in tree_apples.itertuples():
            colour = (0, 255, 0) if apple.is_valid else (0, 0, 255)
            cv2.circle(img, (int(apple.u), int(apple.v)), int(apple.radius_px), colour, 2)
        cv2.imwrite(str(out_dir / f"{tree_id}_overlay.png"), img)

        print(f"{tree_id}: {len(tree_apples)} detections ({n_valid} with usable depth) "
              f"-> {ply_path.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
