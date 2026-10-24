"""Stage 8: save a handful of YOLO val overlays in the same red=label,
green=detection style as Stage 6's classical-baseline overlays, for
direct visual comparison between methods.

Run with: .venv\\Scripts\\python.exe scripts\\make_yolo_overlays.py [--run-id ID] [-n N]
"""

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_paths  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="model-v1-first-fit")
    parser.add_argument("-n", type=int, default=5)
    args = parser.parse_args()

    weights = REPO_ROOT / "runs" / "yolo" / args.run_id / "weights" / "best.pt"
    model = YOLO(str(weights))

    val_dir = REPO_ROOT / "data" / "derived" / "yolo_dataset" / "images" / "val"
    val_ids = sorted(p.stem for p in val_dir.glob("*.png"))[: args.n]

    apples_df = pd.read_csv(load_paths()["derived_dir"] / "apples_2d.csv")

    out_dir = REPO_ROOT / "data" / "derived" / "stage8_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    for image_id in val_ids:
        img_path = val_dir / f"{image_id}.png"
        img = cv2.imread(str(img_path))

        gt = apples_df[apples_df["image_id"] == image_id]
        for row in gt.itertuples():
            cv2.circle(img, (int(row.cx), int(row.cy)), int(row.radius_px), (0, 0, 255), 2)

        pred = model.predict(str(img_path), verbose=False, device=0)[0]
        for box in pred.boxes:
            x0, y0, x1, y1 = box.xyxy[0].tolist()
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            r = ((x1 - x0) + (y1 - y0)) / 4
            cv2.circle(img, (int(cx), int(cy)), int(r), (0, 255, 0), 2)

        out_path = out_dir / f"{image_id}_yolo.png"
        cv2.imwrite(str(out_path), img)
        print(f"wrote {out_path.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
