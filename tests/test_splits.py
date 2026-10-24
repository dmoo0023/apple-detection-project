"""Stage 3 tests: every tree is assigned exactly once, whole rows stay together."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_paths  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def splits() -> dict:
    path = REPO_ROOT / "configs" / "splits.json"
    if not path.exists():
        pytest.skip("configs/splits.json not generated yet - run scripts/make_splits.py first")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest() -> pd.DataFrame:
    return pd.read_csv(load_paths()["manifest_csv"])


def test_every_tree_assigned_exactly_once(splits: dict, manifest: pd.DataFrame) -> None:
    all_ids = splits["splits"]["train"] + splits["splits"]["val"] + splits["splits"]["test"]
    assert len(all_ids) == len(manifest), f"expected {len(manifest)} trees, got {len(all_ids)}"
    assert len(set(all_ids)) == len(all_ids), "a tree_id appears in more than one split"
    assert set(all_ids) == set(manifest["tree_id"]), "split trees don't match the manifest exactly"


def test_splits_are_roughly_70_15_15(splits: dict) -> None:
    total = sum(len(splits["splits"][s]) for s in ("train", "val", "test"))
    train_frac = len(splits["splits"]["train"]) / total
    val_frac = len(splits["splits"]["val"]) / total
    test_frac = len(splits["splits"]["test"]) / total
    assert 0.60 <= train_frac <= 0.80
    assert 0.08 <= val_frac <= 0.22
    assert 0.08 <= test_frac <= 0.22


def test_whole_rows_stay_together(splits: dict, manifest: pd.DataFrame) -> None:
    id_to_split = {}
    for split in ("train", "val", "test"):
        for tid in splits["splits"][split]:
            id_to_split[tid] = split

    manifest = manifest.copy()
    manifest["split"] = manifest["tree_id"].map(id_to_split)
    splits_per_row = manifest.groupby("row")["split"].nunique()
    bad_rows = splits_per_row[splits_per_row > 1]
    assert bad_rows.empty, f"rows split across more than one train/val/test set: {bad_rows.index.tolist()}"


def test_seed_is_recorded(splits: dict) -> None:
    assert "seed" in splits and isinstance(splits["seed"], int)
