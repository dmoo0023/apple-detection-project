"""Stage 8 exp-06: build a YOLO SEGMENTATION dataset (polygon masks, from
Stage 2's labels_seg/) in its own folder, parallel to prepare_yolo_dataset.py's
detection-task one -- same images, different label format, kept separate so
building one task's dataset never risks corrupting the other's.

Run with: .venv\\Scripts\\python.exe scripts\\prepare_yolo_seg_dataset.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_paths  # noqa: E402
from scripts.prepare_yolo_dataset import build_split  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    paths = load_paths()
    manifest = pd.read_csv(paths["manifest_csv"])
    splits = json.loads((REPO_ROOT / "configs" / "splits.json").read_text(encoding="utf-8"))

    out_root = paths["derived_dir"] / "yolo_dataset_seg"
    labels_seg_dir = paths["derived_dir"] / "labels_seg"

    counts = {}
    for split in ("train", "val"):
        tree_ids = set(splits["splits"][split])
        labelled = manifest[manifest["tree_id"].isin(tree_ids) & manifest["has_labelme_json"]]
        image_ids = [Path(p).stem for p in labelled["rgb_path"]]
        counts[split] = build_split(split, image_ids, paths["rgb_dir"], labels_seg_dir, out_root)
        print(f"{split}: {counts[split]} image/label pairs -> {out_root / 'images' / split}")

    data_yaml = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "apple"},
    }
    yaml_path = REPO_ROOT / "configs" / "yolo_data_seg.yaml"
    yaml_path.write_text(yaml.dump(data_yaml, sort_keys=False), encoding="utf-8")
    print(f"\nWrote {yaml_path.relative_to(REPO_ROOT)}")
    print(f"train={counts['train']} images, val={counts['val']} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
