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

## Stage 8 (first model only): YOLO fine-tune on 30 labels — 2026-10-24

**Asked:** Student added 3 more corrected labels (30 total: 10 train / 20
val / 0 test) and asked to proceed with training now, planning to add more
and retrain later.

**Produced:** `scripts/prepare_yolo_dataset.py`, `scripts/train_yolo.py`,
`configs/yolo_data.yaml`, `runs/yolo/model-v1-first-fit/` (metrics/plots;
weights git-ignored), `results/model-v1-first-fit_val_summary.md` +
`..._val_matches.csv`, `docs/stage-8-notes.md`, an `EXPERIMENTS.md` entry.
First YOLO fit (yolov8n, 100 epochs, 10 train images) scored F1=0.744,
mAP@0.5=0.756 on val, well above the classical baseline (F1=0.501,
mAP@0.5=0.103). Flagged clearly as a first data point, not a finished
result: only 10 training images, and train/val being adjacent orchard rows
means some of the apparent strength could be domain similarity rather than
general apple-detection ability — both stated explicitly in
`EXPERIMENTS.md` so it isn't mistaken for more than it is. 42/42 project
tests pass (unchanged; no test code touched this round).

**Student review notes:**

## Evaluation: added apple-count accuracy metric — 2026-10-24

**Asked:** Add a metric summarising model accuracy in terms of apple
*count* (not just per-apple match quality) — relevant for yield estimation.

**Produced:** `count_accuracy()` in `src/evaluate.py` (MAE, RMSE, signed
bias, correlation of predicted vs. true apple count per image), wired into
`summarise()`/`summary_to_markdown()`, 3 new tests (45/45 project tests
pass). Re-ran classical baseline (which also picked up 3 more train labels
added earlier, 10 total now) and re-evaluated the existing YOLO weights
(no retrain) to populate the new metric for both. Found a genuinely
interesting result: YOLO wins every per-apple metric but classical has
better count accuracy (MAE 5.15 vs 7.00, bias +0.45 vs +6.40) — YOLO's
higher recall comes from far more total predictions, which pushes raw
per-image counts further from truth despite being individually more often
correct. Documented in `EXPERIMENTS.md` and both `docs/stage-6-notes.md`
and `docs/stage-8-notes.md`.

**Student review notes:**

## Stage 8: second YOLO fit, learning curve started — 2026-10-24

**Asked:** Student added 10 more labels (40 total: 20 train / 20 val / 0
test) and asked to proceed; separately asked (side note) whether trying
different segmentation models would be worth it.

**Produced:** Picked up and validated the 10 new labels, refreshed
manifest/conversion artifacts, rebuilt the YOLO dataset, retrained
(`model-v1-20train`, 20 train images). Result: F1 0.770, mAP@0.5 0.786,
count MAE 4.20 (all improved over the 10-image run; count accuracy now
beats classical too). Added `scripts/plot_learning_curve.py` — 2 points
so far (10, 20 images), explicitly flagged as too few for a trend.
Re-checked the epoch-30 training-loss spike noted in the first run: it
recurred (worse, not better) despite doubling batch size and train
images, which weakens the earlier "small batch" guess — corrected the
note to "cause not yet identified" rather than leave a wrong explanation
standing. Answered the segmentation-model question as a brief
recommendation (not implemented): worth trying once the learning curve
shows signs of levelling off, since Stage 2 already exports YOLO
segmentation labels and Stage 8 item 3 already plans a boxes-vs-
segmentation comparison. 45/45 project tests pass (unchanged).

**Student review notes:**

## Stage 8: third YOLO fit, test labels arrive — 2026-10-24

**Asked:** Student added 21 more labels (61 total: 31 train / 20 val / 10
test — first test labels). "For your perusal."

**Produced:** Validated all 21, refreshed manifest/conversion artifacts.
Confirmed test images are structurally excluded from the YOLO dataset
build (not just a promise — `prepare_yolo_dataset.py` only ever touches
train/val), consistent with rule 3. Retrained (`model-v1-31train`, 31
train images): mAP@0.5 improved to 0.814 (best yet), but count accuracy
got *worse* (MAE 4.20→6.00), reported plainly as a non-monotonic result
rather than smoothed over — flagged as evidence that seed repeats
(Stage 8 item 4) are now a higher priority than previously stated.
Re-checked the recurring training-loss spike: now 3/3 runs, getting
worse and earlier each time (not better, contradicting the "small
dataset noise, will settle" framing). Computed the LR-warmup-schedule
hypothesis numerically against each run's actual results.csv and ruled
it out with real numbers rather than leaving it as an unchecked guess.
Root cause still open. 45/45 tests pass (unchanged).

**Student review notes:**

## Training-spike investigation — 2026-10-24

**Asked:** Dig into the recurring training-loss spike properly (previous
notes had flagged it 3x without reaching a real explanation).

**Produced:** Read Ultralytics' own installed source
(`.venv/Lib/site-packages/ultralytics/engine/trainer.py`) instead of
guessing further. Three verified findings: (1) validation runs every
epoch unconditionally in our setup, ruling out a "skipped validation"
theory; (2) `results.csv` logs repeat bit-for-bit identical values across
groups of 2-4 epochs (confirmed: only 19/39/51 distinct values across 100
logged epochs per run) — the visual "spike width" in the plots overstates
how many distinct measurements are involved; (3) checked saved training
logs for Ultralytics' own built-in NaN/Inf recovery warning — never
fired in any run, meaning the spikes stay below the threshold Ultralytics
itself treats as a genuine training failure. Also corrected an
overstatement from earlier notes: comparing actual minimum precision/
recall/mAP (not raw loss magnitude, which is unbounded) shows the three
runs' worst points are comparably bad, not escalating as previously
claimed — fixed that claim in both `docs/stage-8-notes.md` and
`EXPERIMENTS.md` rather than let it stand. Also fixed an unrelated
document-structure bug in `EXPERIMENTS.md` from an earlier edit (part of
the `model-v1-first-fit` entry had been accidentally displaced to the end
of the file by a prior insertion). Conclusion: consistent with the
YOLO classification head's known rapid re-specialisation from COCO's 80
classes to this project's single class, on a very small/noisy dataset —
not a bug, not currently worth more investigation at this scale. 45/45
tests pass (unchanged).

**Student review notes:**

## NMS IoU threshold fix (over-counting) — 2026-10-24

**Asked:** Would adding an overlap penalty to the loss function help the
over-counting problem? Then: go ahead and generate overlays, explain NMS,
then "yes" to making the fix standing default.

**Produced:** Found YOLO already has this mechanism (NMS), just set
loosely (IoU 0.7 default). Verified empirically that tightening to 0.3
cuts count MAE from 6.00 to 2.60 on the latest run with no F1 cost, before
recommending it (not just asserted). Made `--iou 0.3` the default in
`scripts/train_yolo.py` and `scripts/make_yolo_overlays.py`. Re-evaluated
all 3 logged YOLO runs with the new default, keeping the old (iou=0.7)
results/overlays in `*_iou0.7`-suffixed files/folders rather than
discarding them. This substantially resolved the "non-monotonic count
regression" flagged in the previous entry — with the fix applied
consistently, count MAE is now monotonically improving across all 3 runs,
and YOLO now beats classical on every metric including count accuracy
(previously classical's one advantage). Fixed a real bug found along the
way: `plot_learning_curve.py`'s `drop_duplicates(keep="first")` would have
kept each run's stale pre-fix metrics instead of the re-evaluated ones —
changed to `keep="last"`. Updated `docs/stage-8-notes.md` and
`EXPERIMENTS.md` with corrected tables rather than leaving the old
numbers standing. 45/45 tests pass (unchanged).

**Student review notes:**

## Stage 8: fourth YOLO fit, deliberate row diversity — 2026-10-24

**Asked:** Student asked for labelling-strategy advice (3-4 images per
row across many rows vs. deep in few), then added 16 more labels spread
across 6 new rows and said "I have added some more."

**Produced:** Validated all 16, refreshed manifest/conversion artifacts
(77 total: 40 train / 25 val / 12 test), rebuilt the YOLO dataset,
retrained (`model-v1-40train`) and re-tuned classical on the same new
split for a fair comparison. Result: YOLO F1 0.813, mAP@0.5 0.827
(biggest jump yet), count MAE 3.00. Flagged that val also grew (20→25,
2 new rows) alongside train, so not a clean isolated comparison — but
noted this is itself informative: F1/mAP *improved* on a val set with
rows never seen before, real evidence against the domain-similarity
concern raised at the first run. Classical re-tuned on the same data
moved only slightly (F1 0.588→0.560) despite 4x the tuning images,
suggesting it's near its ceiling. Corrected a now-stale paragraph in
`docs/stage-6-notes.md` that still claimed classical wins on count
accuracy (true before the NMS fix, not after). 45/45 tests pass
(unchanged).

**Student review notes:**

## Multi-colour classical detector — tried, reverted — 2026-10-24

**Asked:** Modify the classical detector to use multiple colours for the
apple. Clarified first: broaden single-colour red detection to catch
shaded/dark-red and bright/orange-highlight apple skin, not add genuinely
different colour families.

**Produced:** Two extra HSV+Lab colour variants in `src/classical.py`,
OR'd with the existing grid-searched primary variant, with 5 new passing
synthetic unit tests. On real data: a severe regression (F1 0.560→0.366,
false positives 332→1134, count MAE 5.00→33.68) — the "highlight" variant
matched a background wall almost exactly and both variants picked up far
more real orchard clutter than synthetic test colours anticipated.
Reported the regression with full numbers and overlay evidence before
taking any action; student chose to revert. `use_shadow_variant`/
`use_highlight_variant` now default `False`; verified the revert
reproduces the known-good baseline numbers bit-for-bit. Code kept (not
deleted) for future retightening against real data. Documented as a
negative result in `docs/stage-6-notes.md` and `EXPERIMENTS.md` rather
than quietly undone, per rule 9. 50/50 tests pass.

**Student review notes:**

## Stage 8 completion: full experiment series, seed repeats, error analysis — 2026-10-24

**Asked:** "Continue with the stages" — proceeded autonomously through
Stage 8's remaining exit-check items (one branch per experiment, per the
git convention), since the workflow only requires stopping at checkpoints
C1-C4, none of which apply here.

**Produced:** Picked up and validated 9 more labels first (row 4/test to
20/31; one more val label). Rebuilt the YOLO dataset and re-evaluated the
standing baseline on the now-26-image val set for a consistent series.
Extended `scripts/train_yolo.py` with `--seed/--lr0/--cos-lr/--mosaic`.
Ran 6 one-change experiments against the baseline (model size, pretrained
vs. random init, LR/schedule, batch/epochs, augmentation, boxes vs.
segmentation — the last needed new scaffolding,
`scripts/prepare_yolo_seg_dataset.py` + `scripts/train_yolo_seg.py`, to
answer the brief's specific "compare centre accuracy" question), each on
its own `exp/0N-...` branch, merged only if it beat the baseline (none
did outright). Caught and fixed a real gitignore gap along the way (the
new segmentation dataset's image copies and Ultralytics cache files were
about to get committed — excluded them, same treatment as the existing
detection dataset). Ran seed repeats (3 seeds each) on the two most
promising configs, reporting a genuinely nuanced result rather than
forcing a winner. Built `scripts/error_analysis.py` and ran it on the
standing best model: 120 FN / 175 FP, grouped by cause from inspecting
the saved overlays (clustered apples dominate misses; background
structures, fallen fruit, and out-of-focus background rows dominate false
positives). All four Stage 8 exit-check items now complete — tagged
`model-v1` and `stage-8-done`. Full numbers and reasoning in
`EXPERIMENTS.md`; plain-language summary in `docs/stage-8-notes.md`.
50/50 tests pass throughout.

**Student review notes:**

## Stage 9: decision table, checkpoint C3 confirmed — 2026-10-24

**Asked:** "Yes, confirm YOLO, proceed to Stage 9."

**Produced:** `scripts/stage9_decision_table.py`, `docs/stage-9-notes.md`,
`results/stage9_decision_table.{csv,md}` — classical vs YOLO on val
(never test), same evaluation code, plus speed and robustness. YOLO wins
every accuracy metric by a wide margin (F1 0.812 vs 0.548, mAP 0.830 vs
0.204); classical's only edge is a small speed advantage. Declined to
build a hybrid method, reasoning explicitly: exp-06 already answered the
closest related question (box-centre beats mask-centroid), and YOLO's
lead is wide enough that a hybrid's only plausible upside isn't worth the
complexity here. Checkpoint C3 resolved. Proceeding to the one-time
locked test evaluation next.

**Student review notes:**

## Stage 9: FINAL TEST EVALUATION — 2026-10-24

**Asked:** Continuing Stage 9 after checkpoint C3 confirmation.

**Produced:** `scripts/run_final_test_eval.py` (refuses to run on a dirty
tree, committed before running so the logged commit hash is accurate),
run once on the locked test split: F1=0.757, mAP@0.5=0.755, count
MAE=4.55 — noticeably lower than val's 0.812/0.830/2.96, reported plainly
as the real generalisation gap between rows seen during development and
rows never seen until this run, not explained away or retuned. One apple
had no usable depth sample and was correctly reported as unknown rather
than fabricated. Tagged `final-test-eval`. No tuning after this result,
per the brief. `docs/stage-9-notes.md` and `EXPERIMENTS.md` updated with
the full val-vs-test comparison.

**Student review notes:**
