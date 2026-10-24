"""Stage 8 item 6: save false positives and false negatives for a given run,
so they can be inspected and grouped by cause -- the per-image overlays this
writes are the input to that grouping (done by eye, summarised in
docs/stage-8-notes.md), not an automatic classifier.

Colour key: red = false negative (missed apple), yellow = false positive
(spurious detection), green = true positive (for context).

Run with: .venv\\Scripts\\python.exe scripts\\error_analysis.py [--run-id ID]
"""

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_paths  # noqa: E402
from src.evaluate import Apple2D, match_all  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="model-v1-40train")
    parser.add_argument("--iou", type=float, default=0.3)
    args = parser.parse_args()

    weights = REPO_ROOT / "runs" / "yolo" / args.run_id / "weights" / "best.pt"
    model = YOLO(str(weights))

    val_dir = REPO_ROOT / "data" / "derived" / "yolo_dataset" / "images" / "val"
    val_ids = sorted(p.stem for p in val_dir.glob("*.png"))

    apples_df = pd.read_csv(load_paths()["derived_dir"] / "apples_2d.csv")
    apples_df = apples_df[apples_df["image_id"].isin(val_ids)]
    gt = [Apple2D(r.image_id, r.cx, r.cy, r.radius_px) for r in apples_df.itertuples()]

    pred, pred_by_image = [], {}
    for image_id in val_ids:
        result = model.predict(str(val_dir / f"{image_id}.png"), verbose=False, device=0, iou=args.iou)[0]
        image_preds = []
        for box in result.boxes:
            x0, y0, x1, y1 = box.xyxy[0].tolist()
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            r = ((x1 - x0) + (y1 - y0)) / 4
            apple = Apple2D(image_id, cx, cy, r, score=float(box.conf[0]))
            pred.append(apple)
            image_preds.append(apple)
        pred_by_image[image_id] = image_preds

    matches = match_all(gt, pred)

    gt_by_image: dict[str, list[Apple2D]] = {}
    for a in gt:
        gt_by_image.setdefault(a.image_id, []).append(a)

    out_dir = REPO_ROOT / "data" / "derived" / "error_analysis" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_id, result in matches.items():
        g_list = gt_by_image.get(image_id, [])
        p_list = pred_by_image.get(image_id, [])
        if not result.unmatched_gt and not result.unmatched_pred:
            continue  # no errors in this image -- skip, nothing to inspect

        img = cv2.imread(str(val_dir / f"{image_id}.png"))
        for gi, pi, _ in result.matched:
            a = g_list[gi]
            cv2.circle(img, (int(a.cx), int(a.cy)), int(a.radius_px), (0, 200, 0), 1)
        for gi in result.unmatched_gt:
            a = g_list[gi]
            cv2.circle(img, (int(a.cx), int(a.cy)), int(a.radius_px), (0, 0, 255), 2)
            rows.append({"image_id": image_id, "type": "false_negative", "cx": a.cx, "cy": a.cy, "radius_px": a.radius_px})
        for pi in result.unmatched_pred:
            a = p_list[pi]
            cv2.circle(img, (int(a.cx), int(a.cy)), int(a.radius_px), (0, 255, 255), 2)
            rows.append({"image_id": image_id, "type": "false_positive", "cx": a.cx, "cy": a.cy, "radius_px": a.radius_px, "score": a.score})

        cv2.imwrite(str(out_dir / f"{image_id}_errors.png"), img)

    errors_df = pd.DataFrame(rows)
    errors_df.to_csv(out_dir / "errors.csv", index=False)

    n_fn = (errors_df["type"] == "false_negative").sum() if len(errors_df) else 0
    n_fp = (errors_df["type"] == "false_positive").sum() if len(errors_df) else 0
    print(f"{n_fn} false negatives, {n_fp} false positives, saved to {out_dir.relative_to(REPO_ROOT)}/")
    print("(green = true positive, red = false negative/missed, yellow = false positive/spurious)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
