# 3D Apple Localisation Pipeline

Detects apples in RGB-D images of orchard trees and estimates each
apple's 3D position, for an automated picking application. Built
following `agent_workflow.md`, the project brief this repo implements.

**Status: Stages 0-8 of 10 complete.** Stage 8 (model training: learning
curve, experiment series, seed repeats, error analysis) is done and
tagged (`stage-8-done`, `model-v1`). Stages 9 (final test evaluation,
gated on checkpoint C3 — the student confirming the chosen method) and 10
(full inference + deliverables) have not started. See
[Current results](#current-results) below for where things stand, and
[`docs/limitations.md`](docs/limitations.md) for what isn't resolved yet.

## Setup

Requires Windows, Python 3.11+, and an NVIDIA GPU (CUDA 12.8+ driver).

```powershell
# from the repo root
python -m venv .venv
.\.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\pip install -r requirements.txt

# confirm the GPU is actually usable (not just "available")
.\.venv\Scripts\python.exe scripts\check_env.py
```

The repo is deliberately kept **outside** any OneDrive-synced folder —
OneDrive's background syncing conflicts with git and with a Python venv's
many small files. Raw data, by contrast, lives wherever it already is;
point `configs/paths.yaml`'s `raw_data_root` at it (see
[Data](#data) below). The repo never copies raw images into itself.

## Data

Raw data (RGB + depth PNGs, labelme JSON labels) is read from the path
set in `configs/paths.yaml` → `raw_data_root`, and is **never written to**
by anything in this repo (rule 1 of the brief). Update that one path if
running on a different machine; nothing else in the code should need a
path edit (rule 8: no paths in code, everything comes from
`configs/paths.yaml`).

Camera intrinsics and the depth scale are in `configs/camera.yaml`. Read
the comments in that file before trusting any 3D number this project
produces — both are **documented approximations**, not measurements (see
checkpoint C1 in `docs/stage-1-notes.md`): the original camera and
capture setup were unavailable, so intrinsics are derived from Intel's
published D435i field-of-view spec, and the depth scale is the RealSense
mm convention applied directly to 8-bit depth PNGs that turned out not to
be the expected 16-bit format.

## Running the pipeline

Each stage has its own script(s), run in order. All paths/config come
from `configs/`; nothing needs editing between runs except adding new
labels to the raw data folder.

```powershell
# Stage 1: audit the raw dataset (paths, pairing, integrity, labels)
.\.venv\Scripts\python.exe scripts\audit_data.py

# Stage 2: convert labelme circles to YOLO detection/segmentation labels
.\.venv\Scripts\python.exe scripts\convert_labels.py

# Stage 3: freeze train/val/test splits (ALREADY DONE -- refuses to
# overwrite configs/splits.json without --overwrite; do not pass that
# flag unless you specifically mean to change the frozen split)
.\.venv\Scripts\python.exe scripts\make_splits.py

# Stage 6: classical colour-masking baseline (tunes on train, evaluates on val)
.\.venv\Scripts\python.exe scripts\run_classical_baseline.py

# Stage 7: pre-label unlabelled train images with the current best detector,
# for a human to correct in labelme (writes to data/derived/prelabels/,
# never the raw data folder)
.\.venv\Scripts\python.exe scripts\pre_label.py [--batch-size N]

# Stage 8: build the YOLO dataset folder, then train and evaluate
.\.venv\Scripts\python.exe scripts\prepare_yolo_dataset.py
.\.venv\Scripts\python.exe scripts\train_yolo.py --run-id <name> [--epochs N] [--batch N]

# Stage 8: plot the learning curve from every logged run
.\.venv\Scripts\python.exe scripts\plot_learning_curve.py

# Save visual overlays (red = label, green = detection) for a given run
.\.venv\Scripts\python.exe scripts\make_yolo_overlays.py --run-id <name> [-n N]
```

Run `.\.venv\Scripts\python.exe -m pytest tests\ -v` to run the full test
suite (50 tests as of this writing — unit tests for every geometry
function, a synthetic-sphere accuracy check, the evaluation module's own
exit check, and more).

### After adding more labels

Whenever more images get labelled in labelme (in the raw RGB folder,
alongside their `.png`): re-run `audit_data.py` then `convert_labels.py`
to pick them up, then (if train or val changed)
`prepare_yolo_dataset.py` and re-run `train_yolo.py` for a new point on
the learning curve. `run_classical_baseline.py` can be re-run any time to
re-tune the classical baseline on the current labels.

## Current results

**Stage 8 exit check is complete**: logged experiments, learning curve,
best config with seed spread, and error analysis are all done (tagged
`model-v1`, `stage-8-done`). Two methods compared on the same val split
through the one shared evaluation module (`src/evaluate.py`), at 86
labelled images (40 train / 26 val / 20 test):

| | Classical (`baseline-colour-v1`) | YOLO (`model-v1-40train`, standing best) |
|---|---|---|
| F1 | 0.560 | **0.812** |
| mAP@0.5 | 0.200 | **0.830** |
| Count MAE (apples/image) | 5.00 | **2.96** |

YOLO's config was chosen against a 6-way one-change experiment series
(model size, pretrained-vs-random, LR/schedule, batch/epochs,
augmentation, boxes-vs-segmentation) and seed-repeat testing on the two
closest contenders — none beat it convincingly, and the margin against
the runner-up is honestly reported as close rather than overstated. Full
tables in `results/`; the complete narrative (including a tried-and-
reverted experiment, two corrected mistakes, and the error analysis) is
in `EXPERIMENTS.md` and `docs/stage-6-notes.md` / `docs/stage-8-notes.md`.
Stages 9 (final test evaluation) and 10 (full inference + deliverables)
have not started.

## Known limitations

See [`docs/limitations.md`](docs/limitations.md) for the full list.
Headline items: 3D depth scale and camera intrinsics are documented
approximations, not measurements (checkpoint C1); reported 3D positions
are the apple's visible near-surface point, not a corrected centre, by
deliberate scope decision (checkpoint C2) — so the brief's "2-3cm of the
true centre" target cannot be read literally against any 3D number here;
the dataset is red apples, single viewpoint, summer only.

## Repository structure

```text
configs/        camera.yaml, paths.yaml, splits.json (frozen), yolo_data.yaml
data/derived/   generated manifests, converted labels, overlays (raw data stays
                at raw_data_root, never copied in)
docs/           one plain-language note per stage, the worked 3D example,
                limitations.md, ai_use_log.md
EXPERIMENTS.md  narrative log of every experiment, including failed ones
results/        metric tables, the learning curve, per-run markdown summaries
runs/           training run logs/plots (model weights are git-ignored) and
                experiments.csv (commit hash + config + metrics per run)
scripts/        one thin script per stage/task -- see "Running the pipeline" above
src/            the shared modules every script/stage calls into:
                  geometry.py   - pixel+depth -> 3D (the one geometry code path)
                  evaluate.py   - matching + every metric (the one evaluation code path)
                  labels.py     - labelme <-> YOLO label conversion
                  classical.py  - the colour-masking detector
                  config.py     - loads configs/*.yaml (no path is ever hard-coded)
                  tracking.py   - git commit hash / dirty-flag logging for reproducibility
tests/          one test file per src/ module, run with pytest
```

## Reproducibility

Every training/evaluation run is logged in `runs/experiments.csv` with
its git commit hash, whether the tree was dirty when it ran, its config,
and its seed (rule 4/5 of the brief) — `src/tracking.py`'s `record_run()`.
`configs/splits.json` is frozen (committed once, refuses to be silently
overwritten) so train/val/test membership never drifts between runs.
Stage tags (`stage-1-done` through `stage-7-done`, `baseline-colour-v1`,
`checkpoint-C2-resolved`) mark points where a stage's exit check passed —
`git log --oneline --all` or `git tag` to see them all.
