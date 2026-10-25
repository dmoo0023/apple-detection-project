"""Pair each tree's RGB image with its depth image, and read that pairing back.

This is the one place file names are interpreted. Stage 1 (audit_data.py)
uses `pair_files` to build data/derived/manifest.csv; every later stage
reads file paths from the manifest with `load_manifest` / `depth_path_for`,
so no script ever rebuilds a file name itself.

Every tree gets a pair status:
  ok             one RGB file and one depth file
  missing_rgb    a depth file but no RGB file
  missing_depth  an RGB file but no depth file
  ambiguous      two or more files claim the same tree in one modality
                 (e.g. a typo in the row/unit number). The clashing path is
                 left blank rather than guessing which file is right.

A file whose name does not match NAME_PATTERN cannot be assigned to a tree
at all; it is returned in `unmatched` so the audit can report it, and its
tree then shows up as missing that file.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.config import load_paths

NAME_PATTERN = re.compile(
    r"^R(?P<row>\d+)U(?P<unit>\d+)_(?P<season>[A-Za-z]+)_(?P<kind>RGB|Depth|D)"
    r"-(?P<hh>\d{1,2})-(?P<mm>\d{1,2})-(?P<ss>\d{1,2})$"
)

PAIR_OK = "ok"
MISSING_RGB = "missing_rgb"
MISSING_DEPTH = "missing_depth"
AMBIGUOUS = "ambiguous"


def tree_id(row: int, unit: int) -> str:
    """Tree ID string for a row and unit number, e.g. (1, 1) -> 'R01U01'."""
    return f"R{row:02d}U{unit:02d}"


def pair_files(rgb_files: list[Path], depth_files: list[Path]) -> tuple[list[dict], list[Path]]:
    """Group RGB and depth files by tree (row + unit) and pair them.

    Files are matched on row and unit only; the capture time in the name may
    differ between the two images of one tree.

    Inputs:
      rgb_files, depth_files: image paths (only their names are used).
    Output:
      (records, unmatched):
        records: one dict per tree, sorted by row then unit, with keys
          tree_id, row, unit, rgb_path, depth_path (Path or None) and
          pair_status (PAIR_OK, MISSING_RGB, MISSING_DEPTH or AMBIGUOUS).
        unmatched: files whose name does not match NAME_PATTERN.
    """
    by_tree: dict[tuple[int, int], dict[str, list[Path]]] = {}
    unmatched = []
    for modality, files in (("rgb", rgb_files), ("depth", depth_files)):
        for p in files:
            m = NAME_PATTERN.match(p.stem)
            if not m:
                unmatched.append(p)
                continue
            key = (int(m["row"]), int(m["unit"]))
            by_tree.setdefault(key, {"rgb": [], "depth": []})[modality].append(p)

    records = []
    for (row, unit), found in sorted(by_tree.items()):
        rgb, depth = found["rgb"], found["depth"]
        if len(rgb) > 1 or len(depth) > 1:
            status = AMBIGUOUS
        elif not rgb:
            status = MISSING_RGB
        elif not depth:
            status = MISSING_DEPTH
        else:
            status = PAIR_OK
        records.append({
            "tree_id": tree_id(row, unit),
            "row": row,
            "unit": unit,
            # a clashing modality is left blank: never silently pick one of several files
            "rgb_path": rgb[0] if len(rgb) == 1 else None,
            "depth_path": depth[0] if len(depth) == 1 else None,
            "pair_status": status,
        })
    return records, unmatched


def load_manifest(path: Path | None = None) -> pd.DataFrame:
    """Read the Stage 1 manifest with blank file paths kept as "" (pandas would
    otherwise turn them into NaN, which crashes Path()). Older manifests without a
    pair_status column get one derived from which paths are present."""
    manifest = pd.read_csv(path or load_paths()["manifest_csv"])
    for col in ("rgb_path", "depth_path"):
        # an all-blank column is read as float NaN; swap NaN for "" before converting to text
        manifest[col] = manifest[col].astype(object).where(manifest[col].notna(), "").astype(str)
    if "pair_status" not in manifest.columns:
        manifest["pair_status"] = [
            PAIR_OK if r and d else MISSING_RGB if not r else MISSING_DEPTH
            for r, d in zip(manifest["rgb_path"], manifest["depth_path"])
        ]
    return manifest


def depth_path_for(image_id: str, manifest: pd.DataFrame) -> Path | None:
    """The depth image paired with an RGB image, looked up in the manifest.

    Inputs:
      image_id: RGB file stem, e.g. 'R01U01_Summer_RGB-12-40-11'.
      manifest: from load_manifest().
    Output:
      Path of the paired depth image, or None if the tree has no usable depth
      file (missing, or ambiguous).
    """
    stems = manifest["rgb_path"].map(lambda p: Path(p).stem if p else "")
    match = manifest[stems == image_id]
    if match.empty or not match.iloc[0]["depth_path"]:
        return None
    return Path(match.iloc[0]["depth_path"])


def load_depth_m(image_id: str, scale: float, manifest: pd.DataFrame | None = None) -> np.ndarray | None:
    """Load the depth image paired with an RGB image, converted to metres.

    Inputs:
      image_id: RGB file stem, e.g. 'R01U01_Summer_RGB-12-40-11'.
      scale: metres per raw depth unit (configs/camera.yaml).
      manifest: from load_manifest(); read from disk if not given.
    Output:
      (H, W) depth array in metres, or None (with a printed warning) if the image
      has no paired depth file. Evaluation code treats None as "no 3D error for
      this image", so a missing file never stops a run.
    """
    path = depth_path_for(image_id, manifest if manifest is not None else load_manifest())
    if path is None:
        print(f"  {image_id}: no paired depth file in the manifest; its 3D errors are left empty")
        return None
    return np.array(Image.open(path).convert("L")).astype(float) * scale
