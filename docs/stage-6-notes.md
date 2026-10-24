# Stage 6: Classical baseline (colour masking)

## What this stage does

Before training any machine-learning model, it's worth having a simple,
understandable baseline: if a one-afternoon colour-thresholding pipeline
already finds most apples, that tells you something about how hard the
problem is; if a learned model (Stage 8) can't beat this simple baseline,
that's a red flag worth investigating before trusting the fancier method.

## The pipeline (`src/classical.py`)

1. **Colour segmentation.** Apples are red; most background (leaves,
   branches, sky, soil) isn't. Two independent colour tests are combined
   with AND: an **HSV** test (hue near red — note red wraps around 0/180 in
   OpenCV's hue scale, so this needs *two* hue ranges, one near 0 and one
   near 180, not one — plus minimum saturation/brightness so dark shadows
   and washed-out highlights don't pass just because their hue happens to
   be reddish), and a **Lab** test (the 'a' channel directly measures
   position on the green–red axis). Requiring *both* tests to agree cuts
   down on false positives a single colour space alone would let through
   (e.g. a reddish-brown branch that passes HSV's hue test but not Lab's).
2. **Morphological clean-up.** "Open" (erode then dilate) removes tiny
   speckle noise; "close" (dilate then erode) fills small holes inside a
   real blob (e.g. a bright highlight on the apple's skin).
3. **Watershed splitting.** Two touching apples often merge into one
   connected blob after segmentation. The **distance transform** (how far
   each pixel is from the nearest non-apple pixel) peaks at each apple's
   own centre and dips where two apples touch; thresholding it *per
   connected blob* (not with one global threshold, which could miss a
   smaller apple's peak entirely) gives one "seed" per apple, and
   **watershed** grows each seed outward to the blob's true edges —
   splitting the merged blob back into separate apples.
4. **Circle fit.** Each separated region's outline is irregular (leaf
   gaps, lighting). A least-squares circle fit (the same linear algebraic
   technique as Stage 4's sphere fit, in 2D) turns that irregular outline
   into a clean (centre, radius).

## Tuning (grid search on train only)

The five colour thresholds (two hue bounds, saturation minimum, value
minimum, Lab 'a' minimum) were grid-searched — 32 combinations — scored by
**F1 on the train split's currently-labelled images only** (10, as of the
2026-10-24 re-tune — was 7 at first run), never val or
test (rule 3). The best combination (train F1 = 0.545) was then evaluated
**once** on the 20 labelled val images through `src/evaluate.py`.

## Result

**Updated 2026-10-24** after the train split grew from 7 to 10 labelled
images (Stage 7/8's ongoing labelling) — the grid search was re-run and
picked a different threshold combination; numbers below are current.

| Metric | Value |
|---|---|
| Precision | 0.583 |
| Recall | 0.593 |
| F1 | 0.588 |
| mAP@0.5 | 0.211 |
| Count MAE (apples/image) | 5.15 |
| Count bias | +0.45 (slight over-counting) |
| Count correlation | 0.902 |

Full breakdown: `results/baseline_colour_v1_val_summary.md` and
`results/baseline_colour_v1_val_matches.csv`. Train F1 (0.540) and val F1
(0.588) stay close, so the (now 10-image) tuning set isn't *wildly*
overfit, but 10 images is still a small basis for picking thresholds —
worth re-checking again as more labels arrive.

**Why mAP@0.5 (0.211) is so much lower than F1 (0.588):** F1 uses the
lenient centre-distance match (within half the labelled radius); mAP@0.5
requires 50% box overlap (IoU), which the classical pipeline's circles
often miss even when well-centred, because of the next finding.

**The count-accuracy metric (added after a student request, see
`EXPERIMENTS.md`) is a different question from F1/mAP**: it asks whether
the *total number* of apples predicted per image is close to the true
count, regardless of whether individual apples were correctly localised.
Classical's count bias here (+0.45, barely over-counting) looks good in
isolation — see Stage 8's notes for why this is actually a case where a
weaker per-apple detector produces a *more* accurate count than a
stronger one.

## Failure modes (from `data/derived/stage6_overlays/`)

- **Shade kills recall.** In images with a dense, shaded canopy, most
  labelled apples get no detection at all — the red skin is too dark to
  pass the colour thresholds under shade.
- **Leaf occlusion undersizes circles.** In well-lit images, most apples
  *are* found, but the detected circle is often visibly smaller than the
  label — the watershed region only covers the patch of skin visible
  through gaps in leaves, and the circle fit follows that, not the apple's
  true extent. This is the main reason mAP@0.5 is so much lower than F1.
- **A few ground-level false positives** — small detections in the grass,
  most plausibly fallen (unlabelled) apples rather than a colour-mistake.
- **No clear false positives from red leaves or soil** were seen in the 10
  sampled overlays — some evidence the HSV+Lab combination is doing its
  job, though 10 images is a small sample to be confident in that.

## Reproducibility

Tagged `baseline-colour-v1`. Logged in `EXPERIMENTS.md` (narrative) and
`runs/experiments.csv` (commit hash, dirty flag, config, metrics) via the
new `src/tracking.py` helper, which every later experiment (Stage 8
onward) will reuse.

## A bug caught and fixed while building this stage

The per-distance-band table in `src/evaluate.py`'s `summarise()` only
recognised the three named bands (0.4–0.7m, 0.7–1.0m, 1.0–1.2m) plus
"unknown" (no depth). It had no label for a valid depth that simply falls
*outside* every named band — which, given this project's real depths
(typically ~0.1–0.2m, the checkpoint-C1 depth-scale limitation), is
**every single matched apple** in this run. The table was silently coming
out empty instead of reporting that. Fixed by adding an explicit
`"out_of_range"` band so this is reported, not hidden (rule 9: never
weaken a check to make it pass) — a regression test
(`test_summarise_reports_out_of_range_band_not_silently`) now guards it.

## What could go wrong later

- The grid search's 5-parameter space was deliberately narrow (morphology
  kernel, minimum blob area, and the watershed seed threshold were fixed,
  not searched) to keep runtime reasonable with so few train images. If
  more labels arrive (Stage 7), widening the search is worth revisiting.
- `area_px` is used as a makeshift detection "confidence" for mAP's
  ranking, since the classical pipeline has no real probability score —
  a bigger blob isn't necessarily a more confident apple (it could be two
  overlapping apples the watershed failed to split), which likely also
  contributes to the low mAP@0.5.

## Terms used

- **Distance transform:** for each foreground pixel, its distance to the
  nearest background pixel — highest in the middle of a blob, near zero at
  its edge.
- **Watershed:** an algorithm that treats the (inverted) distance
  transform like a landscape and "floods" it from seed points outward,
  stopping where two floods would meet — a standard way to split touching
  blobs into separate objects.
