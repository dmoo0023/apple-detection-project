# Experiment log

One entry per experiment (rule 5: one change per experiment, against a
stated baseline). Full machine-readable detail — commit hash, dirty flag,
config, seed — is in `runs/experiments.csv`; this file is the human-
readable narrative: what was tried, why, and what happened.

## baseline-colour-v1 — Stage 6 classical colour-masking baseline

**Date:** 2026-10-24
**What:** First baseline for the whole project. HSV + Lab red thresholding,
morphological clean-up, distance-transform + watershed splitting of
touching apples, least-squares circle fit — `src/classical.py`. Thresholds
(hue ranges, saturation/value minimums, Lab 'a' minimum) grid-searched
(32 combinations) on the 7 currently-labelled **train** images only,
selecting by F1; morphology kernel size, minimum blob area, and the
watershed seed threshold were left fixed (not searched) to keep the grid a
reasonable size given only 7 train images to tune against.

**Result (val, 20 labelled images, through `src/evaluate.py`):**
precision 0.660, recall 0.404, F1 0.501, mAP@0.5 0.103 (TP=231, FP=119,
FN=341). Train F1 was 0.545 — close to val's 0.501, so no dramatic
overfitting to the 7 train images, though 7 is a very small tuning set and
this should be revisited once more labels exist (Stage 7).

**The mAP@0.5 vs F1 gap (0.103 vs 0.501) is itself a finding, not a bug:**
F1 comes from the lenient centre-distance match (within 0.5x the labelled
radius); mAP@0.5 requires box IoU >= 0.5, which the classical detector's
systematically undersized circles (see failure modes below) frequently
fail even when the centre is well-placed. The two metrics are measuring
different things and are expected to disagree this much for a detector
whose main weakness is getting the *size* wrong, not just the position.

**Failure modes** (from inspecting `data/derived/stage6_overlays/`):
- **Shadowed / dense-shade canopy images** (e.g. `R01U01`): recall drops
  sharply — most labelled apples get no detection at all, because the red
  skin colour is too dark/desaturated under shade to pass the HSV/Lab
  thresholds at all.
- **Leaf occlusion → undersized circles:** in well-lit, dense-fruit images
  (e.g. `R01U04`, `R01U08`), most apples ARE detected, but many detected
  circles are visibly smaller than the labelled circle — the watershed
  region only covers the patch of apple skin visible through gaps in the
  leaves, not the apple's true extent, so the circle fit undersizes it.
  This is the main driver of the low mAP@0.5 despite reasonable recall in
  these images.
- **Ground-level false positives:** a handful of small detections appear
  in the grass at the bottom of some frames, with no matching label —
  most likely fallen apples that weren't labelled (not clearly wrong
  detections), but counted as false positives under the current metric
  either way.
- **No clear false positives from red leaves or soil were found** in the
  10 sampled overlays, though the HSV+Lab-AND combination was specifically
  chosen to guard against exactly that (see `src/classical.py`).

**Tag:** `baseline-colour-v1`

**Limitation inherited from checkpoints C1/C2:** the 3D proxy-error numbers
in `results/baseline_colour_v1_val_summary.md` carry both the depth-scale
approximation (C1) and the surface-point bias (C2) — every matched apple's
depth falls in the `out_of_range` band (below the dataset's own stated
0.4-1.2m capture range), which the per-band table reports honestly rather
than hiding. The 2D pixel/cm numbers and detection metrics (P/R/F1/mAP) are
not affected by either limitation and are this baseline's most trustworthy
numbers.

**Update 2026-10-24 (same day, later):** two more train labels arrived
(10 train images total now, was 7); re-ran the grid search, which picked a
different threshold combination. New result: precision 0.583, recall
0.593, F1 0.588, mAP@0.5 0.211 — all moved, F1/mAP both up. Also added a
new metric this update, **apple-count accuracy** (MAE/RMSE/bias/
correlation of predicted vs. true apple count per image), requested
because P/R/F1/mAP all score individual-apple matches, not whether the
*total count* per image is right — relevant for yield estimation.
Classical's count MAE is 5.15 apples/image, bias +0.45 (near-neutral).
See `model-v1-first-fit`'s entry below for why this makes classical look
*better* than YOLO on counting despite losing on every other metric.

## model-v1-first-fit — Stage 8 first YOLO fine-tune

**Date:** 2026-10-24
**What:** First learned-model fit, against the `baseline-colour-v1`
classical baseline. `yolov8n.pt` (pretrained), fine-tuned 100 epochs,
imgsz=640, batch=4, seed=42, on the **10** currently-labelled train images
(572 GT apples in val). One change from the baseline: the detection method
itself (learned vs. classical) — everything else (split, evaluation code,
geometry module) held fixed, per rule 5.

**Result (val, through `src/evaluate.py`):** precision 0.676, recall
0.827, F1 **0.744**, mAP@0.5 **0.756** (TP=473, FP=227, FN=99) — a clear
win over the classical baseline's F1 0.588 / mAP@0.5 0.211 on every
per-apple metric.

**But classical wins on apple count accuracy** (metric added 2026-10-24,
same day, after a student request — see the `baseline-colour-v1` update
note above): YOLO's count MAE is **7.00** apples/image with a **+6.40**
over-counting bias, worse than classical's 5.15 / +0.45. Not a
contradiction: YOLO makes substantially more total predictions than
classical (700 vs 581 across the 20 val images, for 572 true apples) to
achieve its much higher recall, and that extra volume — even though a
higher *fraction* of it is individually correct (precision 0.676 vs
0.583) — pushes per-image counts further from the true count on average.
Precision/recall/F1/mAP score whether *individual* apples are correctly
localised; this new metric scores whether the *total count* per image is
right. They can and do disagree, and which one matters depends on the
application (a picking arm needs individual positions; a yield estimate
needs the total right) — both are kept in the report, not just the
flattering one.

**Caveat, stated plainly:** this is trained on only 10 images. A result
this strong from so little data is encouraging but should be treated
cautiously, for two reasons neither of which this run can rule out on its
own:
1. **No learning curve yet** — Stage 8 calls for performance plotted
   against the number of labelled training images; with one data point
   there's no way to tell whether 10 images is already enough or whether
   this number will move a lot as more labels are added (the student is
   actively adding more and plans to retrain — the natural next step is
   exactly this learning-curve comparison once there's a second point).
2. **Possible domain similarity between train and val**, not data leakage.
   The 70/15/15 split keeps individual trees strictly separated (Stage 3),
   but train's 10 labelled images are all row 2, and val's 20 are all row
   1 — two adjacent rows captured in the same session, so they likely
   share lighting, tree variety and camera conditions closely. That's not
   a split-correctness problem (different physical trees, properly held
   out), but it does mean this first result may not predict performance on
   a visually different row as well as a more row-diverse train/val split
   eventually will once more rows get labelled.

**Not yet done (deferred until more labels exist, per the student's own
plan to retrain):** the full Stage 8 series (model size, pretrained vs.
random init, LR/batch/epoch sweep, augmentation, boxes vs. segmentation),
seed-repeated runs, and the learning-curve plot. This entry is the first
point on that curve, not the finished stage.

**Tag:** none yet — `model-v1` is reserved for Stage 8's full exit check
(logged experiments, learning curve, seed spread, error analysis), not a
single first fit.
