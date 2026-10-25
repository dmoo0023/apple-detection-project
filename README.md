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

## Running everything, top to bottom

Run every command from the repo root in PowerShell. Set `$py` once per
session:

```powershell
$py = ".\.venv\Scripts\python.exe"
```

All paths and settings come from `configs/`. Generated outputs
(`results/`, `runs/yolo/`, `data/derived/`) are not tracked in git; the
scripts below recreate them.

### 1. Data audit (Stage 1)

```powershell
& $py scripts\audit_data.py
```

Checks every RGB/depth pair, file name and label file, and writes
`data/derived/manifest.csv` and `path_report.md`. Should end with
"Exit check: PASS"; the script exits with code 1 if it doesn't.

Every tree gets a `pair_status` in the manifest (`src/pairing.py`): `ok`,
`missing_rgb`, `missing_depth`, or `ambiguous` (two files claim the same
tree, e.g. a typo in the row/unit number; the clashing path is left blank
rather than guessed). Files whose names break the naming pattern, duplicate
tree IDs, case clashes and double extensions all fail the exit check. Every
later stage takes file paths from the manifest, never by rebuilding names,
so after a failed audit nothing crashes or pairs the wrong files:

- a tree with no usable depth file is still detected, and each detection is
  marked invalid with `invalid_reason = "no depth file"`;
- a tree with no usable RGB file is skipped and listed at the end of the run;
- in evaluation, an image with no depth file simply has no 3D error.

Fix the file names and rerun the audit before trusting results.

### 2. Convert labels (Stage 2)

```powershell
& $py scripts\convert_labels.py
```

Converts the labelme circles to YOLO labels and
`data/derived/apples_2d.csv`.

### 3. Train/val/test split (Stage 3, already frozen)

```powershell
& $py scripts\make_splits.py
```

Reports the existing split in `configs/splits.json` and refuses to
overwrite it. Never pass `--overwrite`: the split is fixed.

### 4. Classical baseline (Stage 6)

```powershell
& $py scripts\run_classical_baseline.py
```

Grid-searches the colour thresholds on train and scores the best set on
val (a few minutes).

### 5. YOLO (Stage 8)

Build the dataset folder, then train the chosen model (about 10 minutes):

```powershell
& $py scripts\prepare_yolo_dataset.py
& $py scripts\train_yolo.py --run-id model-v1-40train --batch 8
```

The experiment series, one change each against that baseline:

```powershell
& $py scripts\train_yolo.py --run-id exp-01-yolov8s       --batch 8 --model yolov8s.pt
& $py scripts\train_yolo.py --run-id exp-02-random-init   --batch 8 --model yolov8n.yaml
& $py scripts\train_yolo.py --run-id exp-03-lr-cosine     --batch 8 --cos-lr
& $py scripts\train_yolo.py --run-id exp-04-batch16-ep150 --batch 16 --epochs 150
& $py scripts\train_yolo.py --run-id exp-05-no-mosaic     --batch 8 --mosaic 0
& $py scripts\train_yolo.py --run-id seed-baseline-7      --batch 8 --seed 7
& $py scripts\train_yolo.py --run-id seed-baseline-123    --batch 8 --seed 123
& $py scripts\train_yolo.py --run-id seed-nomosaic-7      --batch 8 --mosaic 0 --seed 7
& $py scripts\train_yolo.py --run-id seed-nomosaic-123    --batch 8 --mosaic 0 --seed 123
```

The segmentation experiment (exp-06) uses `scripts/train_yolo_seg.py` on
the `exp/06-boxes-vs-segmentation` branch.

Re-score a trained model on the current val set without retraining:

```powershell
& $py scripts\train_yolo.py --run-id model-v1-40train --eval-only
```

Analysis of the trained runs:

```powershell
& $py scripts\plot_learning_curve.py                         # results/learning_curve.png
& $py scripts\stage8_comparison.py                           # results/stage8_comparison.csv
& $py scripts\make_yolo_overlays.py --run-id model-v1-40train   # label vs detection overlays
& $py scripts\error_analysis.py                              # missed apples / false positives
```

### 6. Choose the method (Stage 9)

```powershell
& $py scripts\stage9_decision_table.py
```

Compares classical and YOLO on val with identical evaluation code.

The one-time locked test evaluation (already done, tagged
`final-test-eval`):

```powershell
& $py scripts\run_final_test_eval.py
```

It refuses to run with uncommitted changes. Only rerun it if the chosen
method changes, and never tune anything after seeing its result.

### 7. Full run and 3D output (Stage 10)

```powershell
& $py scripts\run_pipeline.py --all     # all 544 trees -> results/apples_3d.csv (~2 min)
& $py scripts\stage10_summary.py        # apples per tree + depth histograms
& $py scripts\plot_apples_3d.py         # 3D graphs of the front-centre points -> results/figures/
```

Use `run_pipeline.py --tree-id R01U01` for a single tree, and
`plot_apples_3d.py --tree-id R01U01 R08U01` to choose which trees get
graphs.

### 8. Tests, test report and code docs

```powershell
& $py -m pytest tests -v --html=docs\test_report.html --self-contained-html
.\.venv\Scripts\pdoc.exe src scripts '!scripts.collect_report_assets' '!scripts.optimisation_figures' -o docs\api --footer-text "3D apple localisation pipeline"
```

### Shortcut: regenerate outputs without retraining

Run steps 1, 2, 4 and 7, plus `train_yolo.py --run-id <name> --eval-only`
for each model you need. Everything else reads the saved weights in
`runs/yolo/<run-id>/weights/`.

### After adding more labels

Label new images in labelme (save the `.json` next to its `.png` in the
raw RGB folder), then rerun steps 1 and 2. If the train or val labels
changed, rerun `prepare_yolo_dataset.py` and `train_yolo.py` for a new
point on the learning curve; `run_classical_baseline.py` re-tunes the
classical baseline on the current labels.

Every training run is logged automatically to `runs/experiments.csv` with
its git commit hash, so commit code changes before training or running the
final test evaluation.

## Tests and test report

The test suite has 60 tests: unit tests for every geometry function, a
synthetic depth-image accuracy check, the evaluation module's own exit
check, and more. Run it with `& $py -m pytest tests -v`. The latest
report is in [`docs/test_report.html`](docs/test_report.html) (open it in
a browser); step 8 regenerates it.

## Code documentation

HTML documentation of every module, class and function in `src/` and
`scripts/`, built from their docstrings with [pdoc](https://pdoc.dev), is
in [`docs/api/`](docs/api/index.html): open `docs/api/index.html` in a
browser. Step 8 regenerates it.

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
                  pairing.py    - file-name parsing, RGB/depth pairing, manifest reading
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
