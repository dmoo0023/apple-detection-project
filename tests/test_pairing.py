"""Tests for RGB/depth pairing (src/pairing.py) and how the pipeline copes with
missing or wrongly named files. Uses made-up file names only: no raw data needed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.run_pipeline import NO_DEPTH_FILE, process_tree  # noqa: E402
from src.geometry import Intrinsics  # noqa: E402
from src.pairing import (  # noqa: E402
    AMBIGUOUS,
    MISSING_DEPTH,
    MISSING_RGB,
    PAIR_OK,
    depth_path_for,
    load_manifest,
    pair_files,
)

RGB = Path("rgb")
DEPTH = Path("depth")


def rgb(name: str) -> Path:
    """A made-up RGB file path."""
    return RGB / f"{name}.png"


def depth(name: str) -> Path:
    """A made-up depth file path."""
    return DEPTH / f"{name}.png"


def by_tree(records: list[dict]) -> dict[str, dict]:
    """Index pair_files' records by tree ID."""
    return {r["tree_id"]: r for r in records}


# --- pairing ---------------------------------------------------------------------

def test_clean_pair() -> None:
    """One RGB and one depth file for a tree pair up as 'ok'."""
    records, unmatched = pair_files([rgb("R01U01_Summer_RGB-12-40-11")], [depth("R01U01_Summer_D-12-40-11")])
    r = by_tree(records)["R01U01"]
    assert r["pair_status"] == PAIR_OK
    assert r["rgb_path"] == rgb("R01U01_Summer_RGB-12-40-11")
    assert r["depth_path"] == depth("R01U01_Summer_D-12-40-11")
    assert unmatched == []


def test_missing_depth_file() -> None:
    """An RGB file with no depth file is reported as missing_depth, with no depth path."""
    records, _ = pair_files([rgb("R01U01_Summer_RGB-12-40-11")], [])
    r = by_tree(records)["R01U01"]
    assert r["pair_status"] == MISSING_DEPTH
    assert r["depth_path"] is None


def test_missing_rgb_file() -> None:
    """A depth file with no RGB file is reported as missing_rgb, with no RGB path."""
    records, _ = pair_files([], [depth("R01U01_Summer_D-12-40-11")])
    r = by_tree(records)["R01U01"]
    assert r["pair_status"] == MISSING_RGB
    assert r["rgb_path"] is None


def test_timestamp_difference_still_pairs() -> None:
    """RGB and depth are paired on row + unit, so a different capture time still pairs."""
    records, _ = pair_files([rgb("R01U01_Summer_RGB-12-40-11")], [depth("R01U01_Summer_D-12-40-12")])
    assert by_tree(records)["R01U01"]["pair_status"] == PAIR_OK


def test_name_breaking_the_pattern_is_unmatched() -> None:
    """A file whose name breaks the pattern is listed as unmatched, and its tree shows the file missing."""
    bad = rgb("R01U01_Summer_rgb-12-40-11")  # lower-case 'rgb'
    records, unmatched = pair_files([bad], [depth("R01U01_Summer_D-12-40-11")])
    assert unmatched == [bad]
    assert by_tree(records)["R01U01"]["pair_status"] == MISSING_RGB


def test_typo_in_tree_id_is_ambiguous_not_guessed() -> None:
    """Two RGB files claiming one tree (a row/unit typo) is ambiguous: no RGB file is picked."""
    records, _ = pair_files(
        [rgb("R01U01_Summer_RGB-12-40-11"), rgb("R01U02_Summer_RGB-12-40-26"),
         rgb("R01U02_Summer_RGB-12-40-30")],  # this one should have been R01U03
        [depth("R01U01_Summer_D-12-40-11"), depth("R01U02_Summer_D-12-40-26"),
         depth("R01U03_Summer_D-12-40-30")],
    )
    trees = by_tree(records)
    assert trees["R01U02"]["pair_status"] == AMBIGUOUS
    assert trees["R01U02"]["rgb_path"] is None  # never silently pick one of the two
    assert trees["R01U03"]["pair_status"] == MISSING_RGB
    assert trees["R01U01"]["pair_status"] == PAIR_OK


# --- reading the manifest back ---------------------------------------------------

def test_load_manifest_keeps_blank_paths_as_empty_strings(tmp_path: Path) -> None:
    """A blank path in the manifest reads back as "" (not NaN), and has no paired depth file."""
    csv = tmp_path / "manifest.csv"
    pd.DataFrame([
        {"tree_id": "R01U01", "rgb_path": "rgb/R01U01_Summer_RGB-12-40-11.png",
         "depth_path": "depth/R01U01_Summer_D-12-40-11.png", "pair_status": PAIR_OK},
        {"tree_id": "R01U02", "rgb_path": "rgb/R01U02_Summer_RGB-12-40-26.png",
         "depth_path": "", "pair_status": MISSING_DEPTH},
    ]).to_csv(csv, index=False)
    manifest = load_manifest(csv)
    assert manifest.loc[1, "depth_path"] == ""
    assert depth_path_for("R01U02_Summer_RGB-12-40-26", manifest) is None
    assert depth_path_for("R01U01_Summer_RGB-12-40-11", manifest) == Path("depth/R01U01_Summer_D-12-40-11.png")
    assert depth_path_for("R09U99_Summer_RGB-00-00-00", manifest) is None  # not in the manifest at all


def test_load_manifest_derives_status_for_old_manifests(tmp_path: Path) -> None:
    """A manifest written before pair_status existed gets the status from which paths are present."""
    csv = tmp_path / "manifest.csv"
    pd.DataFrame([{"tree_id": "R01U01", "rgb_path": "a.png", "depth_path": ""}]).to_csv(csv, index=False)
    assert load_manifest(csv).loc[0, "pair_status"] == MISSING_DEPTH


# --- the pipeline with a missing depth file --------------------------------------

class _Boxes:
    """Stand-in for Ultralytics' Boxes: one box per (x0, y0, x1, y1, conf)."""

    def __init__(self, boxes: list[tuple[float, float, float, float, float]]) -> None:
        self._boxes = boxes

    def __iter__(self):
        """Yield objects with .xyxy and .conf tensors, like Ultralytics' per-box view."""
        for x0, y0, x1, y1, c in self._boxes:
            yield type("Box", (), {"xyxy": torch.tensor([[x0, y0, x1, y1]]), "conf": torch.tensor([c])})()


class _StubModel:
    """Stand-in for a YOLO model that always returns the same detections."""

    def __init__(self, boxes: list[tuple[float, float, float, float, float]]) -> None:
        self._boxes = boxes

    def predict(self, *args, **kwargs) -> list:
        """Return one result whose .boxes are the fixed detections."""
        return [type("Result", (), {"boxes": _Boxes(self._boxes)})()]


def test_pipeline_marks_detections_invalid_when_depth_file_missing() -> None:
    """With no depth file the tree is still processed: each detection gets a row, marked invalid."""
    model = _StubModel([(100, 100, 140, 140, 0.9), (300, 200, 330, 230, 0.8)])
    rows = process_tree("R01U01", Path("unused.png"), None, model,
                        Intrinsics(600.0, 600.0, 320.0, 240.0), scale=0.001, iou=0.3)
    assert len(rows) == 2
    for r in rows:
        assert r["is_valid"] is False
        assert r["invalid_reason"] == NO_DEPTH_FILE
        assert r["X"] is None and r["Y"] is None and r["Z"] is None
    assert rows[0]["u"] == pytest.approx(120.0)


def test_pipeline_uses_depth_when_present(tmp_path: Path) -> None:
    """With a depth file the same detections get valid 3D points."""
    from PIL import Image
    depth_png = tmp_path / "d.png"
    Image.fromarray(np.full((480, 640), 150, dtype=np.uint8)).save(depth_png)  # 150 raw -> 0.15 m
    model = _StubModel([(100, 100, 140, 140, 0.9)])
    rows = process_tree("R01U01", Path("unused.png"), depth_png, model,
                        Intrinsics(600.0, 600.0, 320.0, 240.0), scale=0.001, iou=0.3)
    assert rows[0]["is_valid"] is True
    assert rows[0]["Z"] == pytest.approx(0.15)
    assert rows[0]["invalid_reason"] == ""
