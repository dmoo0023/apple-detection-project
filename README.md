# 3D Apple Localisation Pipeline

Detects apples in RGB-D images of orchard trees and gives the 3D
position of each apple's front-centre point (the point on its
camera-facing surface at its 2D centre), for an automated picking
application.

**Status: all stages complete.** (Stage 7, detector-assisted
pre-labelling, was not used: every label was drawn by hand in labelme, so
the stage numbering skips from 6 to 8.) YOLO (`model-v1`) was chosen after
the full Stage 8 experiment series, scored **F1=0.757, mAP@0.5=0.755** on
the one-time locked test evaluation, and was then run over **all 544
trees**: 19,294 apples detected. See [Results](#results) and
[Known limitations](#known-limitations) below.

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
produces — both are **documented approximations**, not measurements: the original camera and
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

# Stage 8: build the YOLO dataset folder, then train and evaluate
.\.venv\Scripts\python.exe scripts\prepare_yolo_dataset.py
.\.venv\Scripts\python.exe scripts\train_yolo.py --run-id <name> [--epochs N] [--batch N]
# re-evaluate an existing run's saved weights on the current val set, no retraining
.\.venv\Scripts\python.exe scripts\train_yolo.py --run-id <name> --eval-only

# Stage 8: plot the learning curve from every logged run
.\.venv\Scripts\python.exe scripts\plot_learning_curve.py

# Save visual overlays (red = label, green = detection) for a given run
.\.venv\Scripts\python.exe scripts\make_yolo_overlays.py --run-id <name> [-n N]

# Stage 9: decision table comparing classical vs YOLO on val (checkpoint C3)
.\.venv\Scripts\python.exe scripts\stage9_decision_table.py

# Stage 9: the ONE-TIME locked test evaluation -- ALREADY DONE, tagged
# final-test-eval. Refuses to run on a dirty git tree. Do not re-run this
# unless you are genuinely starting a new, separate final evaluation (e.g.
# a materially different chosen method) -- it is not meant to be repeated
# or tuned against.
.\.venv\Scripts\python.exe scripts\run_final_test_eval.py

# Stage 10: full inference over one tree or all 544 (-> results/apples_3d.csv)
.\.venv\Scripts\python.exe scripts\run_pipeline.py --tree-id R01U01
.\.venv\Scripts\python.exe scripts\run_pipeline.py --all

# Stage 10: summary stats + histograms from results/apples_3d.csv
.\.venv\Scripts\python.exe scripts\stage10_summary.py

# Stage 10: 3D graph of the apples' front-centre points (plus photo
# overlay, and the two side by side) for example trees -> results/figures/
.\.venv\Scripts\python.exe scripts\plot_apples_3d.py [--tree-id R01U01 ...]

# Stage 8: comparison table of every experiment run (settings actually
# used + val metrics) -> results/stage8_comparison.csv
.\.venv\Scripts\python.exe scripts\stage8_comparison.py
```

Generated outputs (`results/`, `runs/yolo/`, `data/derived/`) are not
tracked in git; the scripts above recreate them.

## Tests and test report

Run `.\.venv\Scripts\python.exe -m pytest tests\ -v` to run the full test
suite (49 tests as of this writing — unit tests for every geometry
function, a synthetic depth-image accuracy check, the evaluation module's
own exit check, and more).

The latest test report is in
[`docs/test_report.html`](docs/test_report.html) (open it in a browser).
Regenerate it with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v --html=docs\test_report.html --self-contained-html
```

## Code documentation

HTML documentation of every module, class and function in `src/` and
`scripts/` (built from their docstrings with [pdoc](https://pdoc.dev)) is
in [`docs/api/`](docs/api/index.html) — open `docs/api/index.html` in a
browser. Regenerate it after changing any docstring with:

```powershell
.\.venv\Scripts\pdoc.exe src scripts -o docs\api --footer-text "3D apple localisation pipeline"
```

### After adding more labels

Whenever more images get labelled in labelme (in the raw RGB folder,
alongside their `.png`): re-run `audit_data.py` then `convert_labels.py`
to pick them up, then (if train or val changed)
`prepare_yolo_dataset.py` and re-run `train_yolo.py` for a new point on
the learning curve. `run_classical_baseline.py` can be re-run any time to
re-tune the classical baseline on the current labels.

## Results

YOLO was chosen over the classical baseline on the validation split, then
evaluated once on the locked test split (orchard rows never seen in
training or validation):

| | Classical (val) | YOLO val (`model-v1`) | **YOLO test (`final-test-eval`, locked)** |
|---|---|---|---|
| F1 | 0.548 | 0.812 | **0.757** |
| mAP@0.5 | 0.204 | 0.830 | **0.755** |
| Count MAE (apples/image) | 4.92 | 2.96 | 4.55 |

Test scores are lower than val on every metric — reported as the real,
honest generalisation gap (some val rows were seen repeatedly during
development; test rows never were), not retuned away, per the brief's
"no tuning after the final test evaluation" rule.

**Full inference, all 544 trees:** 19,294 apples detected, 0-94 per tree
(median 36), 0.7% with no usable depth sample (flagged as invalid, never
given a made-up position). Every valid depth falls below the dataset's
stated 0.4 m minimum capture distance — a direct consequence of the
approximate depth scale (see below).

## Known limitations

- Depth scale and camera intrinsics are documented approximations, not
  measurements, so the "2-3 cm" accuracy target cannot be verified in
  absolute terms against any 3D number here.
- 3D accuracy is a proxy: there is no physical ground truth, so "3D error"
  is the distance between front-centre points computed from the predicted
  and the labelled circle through the same geometry code.
- The 3D output is each visible apple's front-centre point; apples hidden
  from the single camera viewpoint are not detected at all.
- The dataset is red apples, one viewpoint per tree, summer only.
- Ultralytics' default `optimizer='auto'` (AdamW, lr0=0.002) overrode the
  requested learning rate in every training run.

## Repository structure

```text
configs/        camera.yaml, paths.yaml, splits.json (frozen), yolo_data.yaml
data/derived/   manifest.csv, apples_2d.csv, checksums.txt (everything else
                here is generated; raw data stays at raw_data_root)
docs/           api/ (HTML code documentation), test_report.html (pytest report)
results/        baseline_colour_v1_config.json (tuned classical detector);
                all other outputs are generated here by the scripts
runs/           experiments.csv (commit hash + config + metrics per run);
                training outputs are generated here by train_yolo.py
scripts/        one thin script per stage/task -- see "Running the pipeline" above
src/            the shared modules every script/stage calls into:
                  geometry.py   - 2D centre + depth -> 3D front-centre point (the one geometry code path)
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
overwritten) so train/val/test membership never drifts between runs. The
final test evaluation (`scripts/run_final_test_eval.py`) refuses outright
to run on an uncommitted (dirty) working tree, so its logged commit hash
is an unambiguous record of exactly what code produced it. Stage tags
(`stage-1-done` through `stage-10-done`, `baseline-colour-v1`,
`checkpoint-C2-resolved`, `model-v1`, `final-test-eval`, and
`final-test-eval-front-point` for the identical rerun after the
front-centre-point cleanup) mark points where a stage's exit check passed
or a checkpoint was confirmed — `git log --oneline --all` or `git tag` to
see them all.

## Licence

Copyright (C) 2026 moodyyyyyy (dmoo0023, Monash University).

This project is licensed under the **GNU Affero General Public License
v3.0** — see [`LICENSE`](LICENSE). AGPL-3.0 was chosen because the
pipeline is built on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics),
which is itself AGPL-3.0; code that uses it is distributed under the same
terms. The raw dataset is not part of this repository and is not covered
by this licence.
