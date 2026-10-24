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
