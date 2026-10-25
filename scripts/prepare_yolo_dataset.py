"""Stage 8: build a YOLO-compatible dataset folder from the currently
labelled train/val images.

Ultralytics expects images and labels in parallel 'images/<split>' and
'labels/<split>' folders (it maps one to the other by replacing
'/images/' with '/labels/' in the path). Our raw images live in OneDrive
and our converted labels in data/derived/labels_det/ (flat), so this copies
both into that expected layout under data/derived/yolo_dataset/ -- small
enough (currently ~30 labelled images) that copying, not symlinking, is
simplest and most reliable on Windows.

Run with: .venv\\Scripts\\python.exe scripts\\prepare_yolo_dataset.py
"""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_paths  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_split(split: str, image_ids: list[str], rgb_dir: Path, labels_det_dir: Path, out_root: Path) -> int:
    """Copy one split's currently-labelled images+labels into Ultralytics' expected
    images/<split>/ + labels/<split>/ layout. Clears the folder first and rebuilds
    from scratch every run, rather than only adding new files, so a label that gets
    corrected or a tree that moves between splits can never leave a stale copy behind.
    """
    img_out = out_root / "images" / split
    lbl_out = out_root / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for f in img_out.glob("*"):
        f.unlink()
    for f in lbl_out.glob("*"):
        f.unlink()

    n = 0
    for image_id in image_ids:
        src_img = rgb_dir / f"{image_id}.png"
        src_lbl = labels_det_dir / f"{image_id}.txt"
        if not src_img.exists() or not src_lbl.exists():
            continue
        shutil.copy2(src_img, img_out / f"{image_id}.png")
        shutil.copy2(src_lbl, lbl_out / f"{image_id}.txt")
        n += 1
    return n


def main() -> int:
    """Build the Ultralytics dataset folder (labelled train/val images + labels) and configs/yolo_data.yaml. Returns 0."""
    paths = load_paths()
    manifest = pd.read_csv(paths["manifest_csv"])
    splits = json.loads((REPO_ROOT / "configs" / "splits.json").read_text(encoding="utf-8"))

    out_root = paths["derived_dir"] / "yolo_dataset"
    labels_det_dir = paths["derived_dir"] / "labels_det"

    counts = {}
    for split in ("train", "val"):
        tree_ids = set(splits["splits"][split])
        labelled = manifest[manifest["tree_id"].isin(tree_ids) & manifest["has_labelme_json"]]
        image_ids = [Path(p).stem for p in labelled["rgb_path"]]
        counts[split] = build_split(split, image_ids, paths["rgb_dir"], labels_det_dir, out_root)
        print(f"{split}: {counts[split]} image/label pairs -> {out_root / 'images' / split}")

    data_yaml = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "apple"},
    }
    yaml_path = REPO_ROOT / "configs" / "yolo_data.yaml"
    yaml_path.write_text(yaml.dump(data_yaml, sort_keys=False), encoding="utf-8")
    print(f"\nWrote {yaml_path.relative_to(REPO_ROOT)}")
    print(f"train={counts['train']} images, val={counts['val']} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
