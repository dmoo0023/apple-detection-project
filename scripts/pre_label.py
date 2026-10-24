"""Stage 7: pre-label unlabelled TRAIN images with the current best detector
(Stage 6's classical baseline), as labelme-compatible circle JSON in a
SEPARATE review folder -- never the raw data root (rule 1). The student
opens these in labelme, corrects them, and copies the corrected files into
the real dataset folder; scripts/convert_labels.py and scripts/audit_data.py
are then re-run to pick them up as real labels.

Only TRAIN images are pre-labelled. Val and test need genuine manual
labels, not detector-assisted ones -- pre-labelling them would let the
detector's own blind spots quietly become "ground truth" for the numbers
used to judge it.

Run with: .venv\\Scripts\\python.exe scripts\\pre_label.py [--batch-size N]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.classical import ClassicalConfig, detect_apples  # noqa: E402
from src.config import load_paths  # noqa: E402
from src.labels import write_labelme_json  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BATCH_SIZE = 50


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    config_path = REPO_ROOT / "results" / "baseline_colour_v1_config.json"
    if not config_path.exists():
        print(f"Missing {config_path} -- run scripts/run_classical_baseline.py first.")
        return 1
    config = ClassicalConfig(**json.loads(config_path.read_text(encoding="utf-8")))

    paths = load_paths()
    manifest = pd.read_csv(paths["manifest_csv"])
    splits = json.loads((REPO_ROOT / "configs" / "splits.json").read_text(encoding="utf-8"))
    train_ids = set(splits["splits"]["train"])

    unlabelled_train = manifest[
        manifest["tree_id"].isin(train_ids) & ~manifest["has_labelme_json"]
    ].sort_values("tree_id")

    batch = unlabelled_train.head(args.batch_size)
    print(f"Unlabelled train trees: {len(unlabelled_train)}. Pre-labelling {len(batch)} of them.")

    out_dir = paths["derived_dir"] / "prelabels"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for row in batch.itertuples():
        rgb_path = Path(row.rgb_path)
        image_id = rgb_path.stem
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        detections = detect_apples(rgb, config)
        apples = [(d.cx, d.cy, d.radius_px) for d in detections]

        json_path = out_dir / f"{image_id}.json"
        write_labelme_json(json_path, rgb_path.name, apples, img_w=640, img_h=480)
        rows.append({"tree_id": row.tree_id, "image_id": image_id, "n_prelabelled_apples": len(apples)})

    report_df = pd.DataFrame(rows)
    report_df.to_csv(out_dir / "prelabel_manifest.csv", index=False)

    print(f"\nWrote {len(rows)} pre-label JSON files to {out_dir.relative_to(REPO_ROOT)}/")
    print(f"Detections per image: min={report_df['n_prelabelled_apples'].min()}, "
          f"median={report_df['n_prelabelled_apples'].median():.0f}, "
          f"max={report_df['n_prelabelled_apples'].max()}, "
          f"{int((report_df['n_prelabelled_apples'] == 0).sum())} images with ZERO detections "
          f"(worth checking by hand -- the detector may have simply missed every apple).")
    print(f"\nNEXT STEP (manual, waiting on the student):")
    print(f"  1. Open each image+JSON pair in {out_dir.relative_to(REPO_ROOT)}/ with labelme.")
    print(f"  2. Correct the circles (move/resize/delete wrong ones, add missed apples).")
    print(f"  3. Copy the corrected .json into the SAME folder as its .png in the raw "
          f"RGB folder (never edit the raw folder directly from here, rule 1).")
    print(f"  4. Tell the agent to re-run scripts/audit_data.py and scripts/convert_labels.py "
          f"to pick up the newly labelled images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
