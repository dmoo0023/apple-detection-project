"""Stage 3 exit check: assign all 544 trees to train/val/test, whole rows at a time.

Writes configs/splits.json (refuses to overwrite without --overwrite) and
data/derived/stage3_report.md with the per-split counts and a recommendation
for which images to label next.

Run with: .venv\\Scripts\\python.exe scripts\\make_splits.py [--overwrite]
"""

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_paths  # noqa: E402

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test gets the remainder (~0.15)
TEST_LABEL_TARGET_LOW = 50
TEST_LABEL_TARGET_HIGH = 80


def assign_rows_to_splits(rows_with_counts: list[tuple[int, int]], seed: int) -> dict[int, str]:
    """Shuffle rows with a fixed seed, then greedily fill train, then val, then test,
    by cumulative tree count, so whole rows never get split across train/val/test.
    """
    rows = list(rows_with_counts)
    random.Random(seed).shuffle(rows)
    total = sum(c for _, c in rows)
    train_target = TRAIN_FRAC * total
    val_target = (TRAIN_FRAC + VAL_FRAC) * total

    assignment: dict[int, str] = {}
    cumulative = 0
    for row, count in rows:
        if cumulative < train_target:
            assignment[row] = "train"
        elif cumulative < val_target:
            assignment[row] = "val"
        else:
            assignment[row] = "test"
        cumulative += count
    return assignment


def main() -> int:
    """Create configs/splits.json (whole rows kept together, fixed seed); refuses to overwrite
    without --overwrite. Returns 0 on success, 1 if the file already exists.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="overwrite an existing configs/splits.json")
    args = parser.parse_args()

    paths = load_paths()
    splits_path = Path(__file__).resolve().parent.parent / "configs" / "splits.json"

    if splits_path.exists() and not args.overwrite:
        print(f"REFUSING to overwrite existing {splits_path} (per rule: splits are frozen once committed).")
        print("Pass --overwrite if you really mean to regenerate it.")
        return 1

    manifest_path = paths["manifest_csv"]
    df = pd.read_csv(manifest_path)
    assert len(df) == 544, f"expected 544 trees in manifest, found {len(df)}"

    rows_with_counts = df.groupby("row").size().items()
    row_assignment = assign_rows_to_splits(list(rows_with_counts), SEED)

    df["split"] = df["row"].map(row_assignment)

    splits = {
        split: sorted(df.loc[df["split"] == split, "tree_id"].tolist())
        for split in ("train", "val", "test")
    }
    splits_payload = {
        "seed": SEED,
        "method": "whole rows shuffled with a fixed seed, then greedily filled train->val->test "
                  "by cumulative tree count, targeting 70/15/15",
        "train_frac_target": TRAIN_FRAC,
        "val_frac_target": VAL_FRAC,
        "rows": {str(k): v for k, v in sorted(row_assignment.items())},
        "splits": splits,
    }
    splits_path.write_text(json.dumps(splits_payload, indent=2), encoding="utf-8")

    report_lines = ["# Stage 3 split report\n"]

    def log(line: str = "") -> None:
        """Print a line and also keep it for the markdown report."""
        print(line)
        report_lines.append(line)

    log(f"Wrote {splits_path.relative_to(Path.cwd())} (seed={SEED}).\n")
    log("## Row assignment")
    for split in ("train", "val", "test"):
        split_rows = sorted(r for r, s in row_assignment.items() if s == split)
        log(f"- {split}: rows {split_rows}")

    log("\n## Counts per split")
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        n_trees = len(sub)
        n_labelled = int(sub["has_labelme_json"].sum())
        pct = 100 * n_trees / len(df)
        log(f"- **{split}**: {n_trees} trees ({pct:.1f}%), {n_labelled} labelled")

    test_sub = df[df["split"] == "test"]
    test_labelled = int(test_sub["has_labelme_json"].sum())
    log(f"\n## Labelling recommendation (target: {TEST_LABEL_TARGET_LOW}-{TEST_LABEL_TARGET_HIGH} labelled test images)")
    if test_labelled >= TEST_LABEL_TARGET_LOW:
        log(f"- test split already has {test_labelled} labelled images, within/above target. No action needed yet.")
    else:
        needed = TEST_LABEL_TARGET_LOW - test_labelled
        candidates = test_sub[~test_sub["has_labelme_json"]].sort_values("tree_id")["tree_id"].tolist()
        recommended = candidates[:needed]
        log(f"- test split currently has only {test_labelled} labelled images. "
            f"Recommend labelling at least {needed} more from the test split to reach "
            f"the low end of the target ({TEST_LABEL_TARGET_LOW}).")
        log("- suggested trees to label next (from test split, unlabelled, in tree-ID order):")
        log(f"  {recommended}")
        log("- note: these are **test** images — per rule 3, do not look at their predictions "
            "or metrics until the Stage 9 final evaluation, only label them.")

    val_sub = df[df["split"] == "val"]
    val_labelled = int(val_sub["has_labelme_json"].sum())
    log(f"\n- val split has {val_labelled} labelled images. Stage 5/6 need at least a handful of "
        f"labelled val images to compute meaningful metrics; if this is 0, prioritise labelling "
        f"some val-split images too.")

    (paths["derived_dir"] / "stage3_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    log(f"\n## Exit check: PASS (544/544 trees assigned, splits frozen at {splits_path.relative_to(Path.cwd())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
