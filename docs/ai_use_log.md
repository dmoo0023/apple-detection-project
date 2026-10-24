# AI use log

One entry per stage: date, what the agent was asked to do, what it produced.
The "Student review notes" line under each entry is for the student to fill
in themselves; the agent never writes on that line.

## Stage 0: Environment — 2026-10-24

**Asked:** Set up a Python environment outside OneDrive with GPU-enabled
PyTorch and the project's other dependencies, write a GPU check script, and
initialise git with a `.gitignore` and first commit.

**Produced:** `.venv` at `C:\Users\melbo\dev\apple-detection-project\.venv`,
`requirements.txt`, `scripts/check_env.py`, `.gitignore`, git repo
initialised with `main` branch, `docs/stage-0-notes.md`.

**Student review notes:**

## Stage 1: Data audit and path checks — 2026-10-24

**Asked:** Audit the raw dataset (paths, naming, pairing, integrity,
labelme labels, depth sanity), without modifying any raw files, and verify
camera intrinsics and depth scale rather than assuming RealSense defaults.

**Produced:** `configs/paths.yaml`, `scripts/audit_data.py`,
`data/derived/manifest.csv`, `data/derived/path_report.md`,
`tests/test_paths.py`, depth/RGB overlays, `docs/stage-1-notes.md`. Camera
intrinsics and depth scale could **not** be verified from the data — see
checkpoint C1 below.

**Student review notes:**

## Checkpoint C1 resolution — 2026-10-24

**Asked:** Student reviewed the overlays (confirmed alignment correct) and
decided, after being shown the concrete numeric consequences, how to handle
the missing depth scale and camera intrinsics (no camera/capture setup
available to measure either).

**Produced:** Updated `configs/camera.yaml` with documented approximations
(depth scale = RealSense mm convention applied directly to the 8-bit
values; intrinsics derived from Intel's published D435i FOV spec, cx/cy at
image centre), both flagged with long comments and a stated error range.
Updated `scripts/audit_data.py` to run the 0.4-1.2 m depth sanity check
honestly against the confirmed scale (it fails, as expected, and the
failure is printed rather than hidden). Updated `docs/stage-1-notes.md`
with the full reasoning and the practical implication: absolute 3D
accuracy cannot be rigorously verified with this setup. Stage 1 exit check
now passes; tagging `stage-1-done` and merging to `main`.

**Student review notes:**

## Stage 2: Label conversion — 2026-10-24

**Asked:** Convert the 25 labelme circle labels into YOLO detection and
segmentation formats plus a flat per-apple CSV, in one shared module, with
warnings for bad circles and a round-trip test.

**Produced:** Conversion functions added to `src/labels.py`
(`extract_apples`, `yolo_detection_line`, `yolo_segmentation_line`, and
their inverses), `scripts/convert_labels.py`, `data/derived/apples_2d.csv`,
`data/derived/labels_det/`, `data/derived/labels_seg/`,
`data/derived/stage2_overlays/`, `tests/test_labels.py` (11 new tests, all
passing), `docs/stage-2-notes.md`. All 25 images converted with zero
warnings (781 apples).

**Student review notes:**

## Stage 3: Splits — 2026-10-24

**Asked:** Assign all 544 trees to train/val/test (~70/15/15), keeping
whole orchard rows together to avoid background leakage, with a fixed
seed, refusing to silently overwrite an existing split, and recommending
which images to label next so the test set reaches 50-80 labelled images.

**Produced:** `scripts/make_splits.py`, `configs/splits.json` (frozen,
seed=42), `data/derived/stage3_report.md`, `tests/test_splits.py` (4 new
tests, all passing), `docs/stage-3-notes.md`. Result: 383/84/77 trees
(train/val/test), with the 25 currently labelled images split 5/20/0 by
chance of the row shuffle. Test split has 0 labelled images right now;
50 specific tree IDs recommended to label next.

**Student review notes:**

## Stage 4: Shared 3D module — 2026-10-24

**Asked:** Build `src/geometry.py` (back-projection, robust depth
sampling, surface-to-centre correction, optional sphere fit), test it on a
synthetic sphere (within 1mm with no noise, with noise too), and write a
hand-calculated worked example for one real labelled apple that the code
must reproduce exactly.

**Produced:** `src/geometry.py` (`backproject`, `robust_depth`,
`apple_centre_3d` with `project_and_offset` and `sphere_fit` methods),
`tests/test_geometry.py` (9 tests: synthetic-sphere accuracy with/without
noise, missing-depth-returns-None behaviour, and exact reproduction of the
worked example), `docs/stage-4-worked-example.md`, `docs/stage-4-notes.md`.
All 29 project tests pass. The worked example surfaced a concrete
consequence of the checkpoint-C1 depth-scale limitation: the computed
apple radius (4.8mm) is not physically realistic.

**Student review notes:**

## Stage 5: Evaluation module — 2026-10-24

**Asked:** Build `src/evaluate.py` (Hungarian matching, precision/recall/
F1, mAP@0.5, 2D and 3D-proxy error, distance-band breakdown, CSV/markdown
output), with the module's own exit check: a trivial predictor (the labels
themselves) must score perfectly, and a deliberately shifted copy must
give a specific, predictable error.

**Produced:** `src/evaluate.py`, `tests/test_evaluate.py` (5 tests, all
passing), `docs/stage-5-notes.md`. Mid-stage, noticed the student had
labelled 2 more images (27 total now, was 25) — validated them, re-ran
Stage 1/2's generation scripts to pick them up, confirmed `splits.json`
regenerates byte-identical (row structure unchanged), and loosened a
hard-coded "25 images / 781 apples" test assertion to check against
whatever's currently labelled instead. 34/34 project tests pass.

**Student review notes:**

## Checkpoint C2 scope decision — 2026-10-24

**Asked:** At the C2 code walkthrough, student judged the surface-to-centre
correction in `src/geometry.py` as overcomplicating the problem given the
depth scale is already an accepted approximation, and asked to remove it
entirely, treating the apple's near-surface point as the point of interest.

**Produced:** Simplified `src/geometry.py`: removed `apple_centre_3d`
(both the `project_and_offset` and `sphere_fit` methods) and its
`AppleCentreResult`/`_fit_sphere`/`_backproject_patch` helpers; added a
plain `surface_point_3d()` (robust depth + backproject, no correction).
Updated `src/evaluate.py` to call it, with docstrings/markdown-caveat text
now stating the resulting ~3-4.5cm systematic surface-vs-centre bias
explicitly. Rewrote `tests/test_geometry.py` (dropped the offset/sphere-fit
tests, repurposed the synthetic-sphere test to validate the surface point
against the sphere's true near-surface intersection, with tolerances
loosened from 1mm/5mm to 3mm/5mm to reflect real curvature-averaging
effects of the disc-median depth sample). Rewrote
`docs/stage-4-worked-example.md` and `docs/stage-4-notes.md` to match, and
added an addendum to `docs/stage-5-notes.md`. 33/33 tests pass.
**Consequence stated plainly for the final report:** every 3D position in
this project now carries a known, deliberate, un-corrected ~3-4.5cm bias
toward the camera versus the apple's true centre, on top of the
checkpoint-C1 depth-scale approximation — the brief's "2-3cm of the true
centre" target cannot be read literally against these numbers.

**Student review notes:**

## Stage 6: Classical baseline — 2026-10-24

**Asked:** Build `src/classical.py` (HSV+Lab red thresholding, morphology,
watershed splitting, least-squares circle fit), tune thresholds by grid
search on train only, evaluate once on val through the shared geometry/
evaluation modules, save overlays, list failure modes, tag
`baseline-colour-v1`.

**Produced:** `src/classical.py`, `src/tracking.py` (new: git commit
hash/dirty-flag logging to `runs/experiments.csv`), `EXPERIMENTS.md` (new),
`scripts/run_classical_baseline.py`, `tests/test_classical.py` (7 tests,
all passing), `results/baseline_colour_v1_val_summary.md` + `..._matches.csv`,
`data/derived/stage6_overlays/`, `docs/stage-6-notes.md`. Val: precision
0.660, recall 0.404, F1 0.501, mAP@0.5 0.103. While building this, found
and fixed a real bug in `src/evaluate.py`'s distance-band reporting (valid
depths outside every named band were silently dropped instead of reported
as "out_of_range") — added a regression test. 41/41 project tests pass.

**Student review notes:**

## Stage 7: Pre-labelling loop — 2026-10-24

**Asked:** Use the current best detector to pre-label unlabelled train
images as labelme-compatible circle JSON in a separate folder, tell the
student which to review, and wait.

**Produced:** `write_labelme_json()` in `src/labels.py` (round-trip
tested), `scripts/pre_label.py`, 50 pre-label JSON files in
`data/derived/prelabels/` (first batch, of 376 unlabelled train images),
`docs/stage-7-notes.md`. 42/42 project tests pass. **Waiting on the
student** to review/correct these in labelme and copy them into the real
dataset folder before any further label-dependent work continues.

**Student review notes:**
