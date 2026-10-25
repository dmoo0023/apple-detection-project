"""Reproducibility bookkeeping (rule 4 / Section 3 item 5): every evaluation
run records the git commit hash, whether the working tree was dirty
(uncommitted changes), the config used, and the seed, in runs/experiments.csv.
"""

import csv
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_CSV = REPO_ROOT / "runs" / "experiments.csv"

FIELDS = ["timestamp", "run_id", "stage", "commit_hash", "dirty", "seed", "config", "metrics_summary"]


def git_commit_hash() -> str:
    """The full SHA of the current HEAD commit -- recorded with every run so results can
    always be traced back to the exact code that produced them."""
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def git_is_dirty() -> bool:
    """True if there are uncommitted changes in the working tree. A run logged while dirty
    means its commit hash alone doesn't fully pin down what code actually ran."""
    result = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return bool(result.stdout.strip())


def refuse_if_dirty(context: str) -> None:
    """Call before a run that must not happen on a dirty tree (e.g. the Stage 9 final
    test evaluation, per Section 3 rule 5: 'Refuse to run the final test evaluation
    on a dirty tree.')."""
    if git_is_dirty():
        raise RuntimeError(
            f"Refusing to run {context}: the git working tree has uncommitted changes. "
            f"Commit or stash them first, so this run's commit hash truly reflects what produced it."
        )


@dataclass
class RunRecord:
    """One row of runs/experiments.csv.

    Attributes:
      run_id: experiment name, e.g. 'model-v1-40train'.
      stage: workflow stage, e.g. 'stage-8'.
      seed: random seed used, or None if not applicable.
      config: the settings that define the run.
      metrics_summary: one-line human-readable result.
    """
    run_id: str
    stage: str
    seed: int | None
    config: dict = field(default_factory=dict)
    metrics_summary: str = ""


def record_run(record: RunRecord) -> None:
    """Append one row to runs/experiments.csv, creating it with a header if needed."""
    import datetime

    EXPERIMENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    is_new = not EXPERIMENTS_CSV.exists()

    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "run_id": record.run_id,
        "stage": record.stage,
        "commit_hash": git_commit_hash(),
        "dirty": git_is_dirty(),
        "seed": record.seed,
        "config": str(record.config),
        "metrics_summary": record.metrics_summary,
    }

    with open(EXPERIMENTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
