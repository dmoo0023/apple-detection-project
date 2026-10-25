"""Stage 9: the ONE-TIME final evaluation of the chosen method (YOLO,
`model-v1`) on the locked TEST split. Per Section 3 rule 5 of the brief,
refuses to run on a dirty git tree, so the commit hash logged with this
run is a true, unambiguous record of exactly what code produced it.

This is the only script in the whole project allowed to touch test-split
metrics. Run it once. Per the brief: do not tune anything after seeing
this result -- if the result is disappointing, that is itself the answer,
not a cue to go back and adjust thresholds until test looks better.

Run with: .venv\\Scripts\\python.exe scripts\\run_final_test_eval.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_paths, load_camera  # noqa: E402
from src.evaluate import (  # noqa: E402
    Apple2D,
    match_all,
    per_match_errors,
    summarise,
    summary_to_markdown,
)
from src.geometry import Intrinsics  # noqa: E402
from src.tracking import RunRecord, record_run, refuse_if_dirty  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CHOSEN_RUN_ID = "model-v1-40train"  # the config tagged model-v1, confirmed at checkpoint C3


def load_test_ids() -> list[str]:
    """Return the image IDs of the labelled images in the locked test split."""
    paths = load_paths()
    manifest = pd.read_csv(paths["manifest_csv"])
    splits = json.loads((REPO_ROOT / "configs" / "splits.json").read_text(encoding="utf-8"))
    test_trees = set(splits["splits"]["test"])
    labelled = manifest[manifest["tree_id"].isin(test_trees) & manifest["has_labelme_json"]]
    return [Path(p).stem for p in labelled["rgb_path"]]


def load_gt(image_ids: list[str]) -> list[Apple2D]:
    """Load the labelled circles for the given images from apples_2d.csv."""
    apples_df = pd.read_csv(load_paths()["derived_dir"] / "apples_2d.csv")
    apples_df = apples_df[apples_df["image_id"].isin(image_ids)]
    return [Apple2D(r.image_id, r.cx, r.cy, r.radius_px) for r in apples_df.itertuples()]


def load_depth_m(image_id: str, scale: float) -> np.ndarray:
    """Load the depth image paired with `image_id`, converted to metres with `scale`."""
    paths = load_paths()
    depth_name = image_id.replace("_RGB-", "_D-")
    raw = np.array(Image.open(paths["depth_dir"] / f"{depth_name}.png").convert("L")).astype(float)
    return raw * scale


def main() -> int:
    """Evaluate the chosen YOLO model once on the test split (clean git tree only), save and log the results. Returns 0."""
    refuse_if_dirty("Stage 9 final test evaluation")

    test_ids = load_test_ids()
    print(f"Test split: {len(test_ids)} labelled images (rows 4, 21 by configs/splits.json).")

    gt_test = load_gt(test_ids)
    cam = load_camera()
    intr = Intrinsics.from_dict(cam)
    scale = cam["depth_scale_to_metres"]

    weights = REPO_ROOT / "runs" / "yolo" / CHOSEN_RUN_ID / "weights" / "best.pt"
    model = YOLO(str(weights))

    rgb_dir = load_paths()["rgb_dir"]
    pred_test, depth_lookup = [], {}
    for image_id in test_ids:
        pred = model.predict(str(rgb_dir / f"{image_id}.png"), verbose=False, device=0, iou=0.3)[0]
        for box in pred.boxes:
            x0, y0, x1, y1 = box.xyxy[0].tolist()
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            r = ((x1 - x0) + (y1 - y0)) / 4
            pred_test.append(Apple2D(image_id, cx, cy, r, score=float(box.conf[0])))
        depth_lookup[image_id] = load_depth_m(image_id, scale)

    matches_test = match_all(gt_test, pred_test)
    match_df = per_match_errors(gt_test, pred_test, matches_test, depth_lookup, intr)

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    match_df.to_csv(results_dir / "final_test_eval_matches.csv", index=False)

    summary = summarise(gt_test, pred_test, matches_test, match_df)
    summary_md = summary_to_markdown(summary, "YOLO (model-v1) -- FINAL TEST EVALUATION", "test")
    (results_dir / "final_test_eval_summary.md").write_text(summary_md + "\n", encoding="utf-8")
    print("\n" + summary_md)

    det = summary["detection"]
    cnt = summary["counting"]
    record_run(RunRecord(
        run_id="final-test-eval",
        stage="stage-9",
        seed=42,
        config={"model": CHOSEN_RUN_ID, "split": "test", "n_test_images": len(test_ids), "nms_iou": 0.3},
        metrics_summary=f"TEST P={det['precision']:.3f} R={det['recall']:.3f} F1={det['f1']:.3f} "
                         f"mAP={det['mAP@0.5']:.3f} countMAE={cnt['count_mae']:.2f} countBias={cnt['count_bias']:+.2f}",
    ))

    print("\n## FINAL TEST EVALUATION COMPLETE. Do not tune further -- this is the answer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
