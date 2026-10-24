"""Stage 1 tests: the manifest must only point to files that really exist,
and every labelled RGB image must have a structurally valid labelme JSON."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_paths  # noqa: E402
from src.labels import validate_labelme_file  # noqa: E402


@pytest.fixture(scope="module")
def manifest() -> pd.DataFrame:
    paths = load_paths()
    manifest_path = paths["manifest_csv"]
    if not manifest_path.exists():
        pytest.skip("manifest.csv not generated yet - run scripts/audit_data.py first")
    return pd.read_csv(manifest_path)


@pytest.fixture(scope="module")
def rgb_dir() -> Path:
    return load_paths()["rgb_dir"]


def test_manifest_not_empty(manifest: pd.DataFrame) -> None:
    assert len(manifest) == 544, f"expected 544 trees, got {len(manifest)}"


def test_every_rgb_path_exists(manifest: pd.DataFrame) -> None:
    missing = [p for p in manifest["rgb_path"] if p and not Path(p).is_file()]
    assert not missing, f"manifest rows point to missing RGB files: {missing[:5]}"


def test_every_depth_path_exists(manifest: pd.DataFrame) -> None:
    missing = [p for p in manifest["depth_path"] if p and not Path(p).is_file()]
    assert not missing, f"manifest rows point to missing depth files: {missing[:5]}"


def test_labelled_images_have_valid_json(manifest: pd.DataFrame, rgb_dir: Path) -> None:
    labelled = manifest[manifest["has_labelme_json"] == True]  # noqa: E712
    assert len(labelled) > 0, "expected at least one labelled image"
    failures = {}
    for rgb_path in labelled["rgb_path"]:
        json_path = Path(rgb_path).with_suffix(".json")
        problems = validate_labelme_file(json_path, rgb_dir)
        if problems:
            failures[json_path.name] = problems
    assert not failures, f"labelme JSON problems: {failures}"


def test_no_duplicate_tree_ids(manifest: pd.DataFrame) -> None:
    dupes = manifest["tree_id"][manifest["tree_id"].duplicated()]
    assert dupes.empty, f"duplicate tree IDs in manifest: {dupes.tolist()}"
