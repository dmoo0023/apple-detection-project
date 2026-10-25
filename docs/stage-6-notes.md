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
   gaps, lighting). A 2D least-squares circle fit turns that irregular outline
   into a clean (centre, radius).

## Tuning (grid search on train only)

The five colour thresholds (two hue bounds, saturation minimum, value
minimum, Lab 'a' minimum) were grid-searched — 32 combinations — scored by
**F1 on the train split's currently-labelled images only** (40, as of the
latest re-tune — 7 at first run, 10, then 40), never val or test (rule 3).
The grid search is re-run from scratch each time (never incrementally
tuned), so the best combination found also changes each time. Evaluated
**once** per re-tune on the current labelled val images through
`src/evaluate.py`.

## Result

**Updated 2026-10-24 (latest)**, now on 40 train / 25 val labelled images
(up from the 10/20 of the previous update) — the grid search was re-run
again and picked yet another threshold combination; numbers below are
current.

| Metric | Value |
|---|---|
| Precision | 0.548 |
| Recall | 0.572 |
| F1 | 0.560 |
| mAP@0.5 | 0.200 |
| Count MAE (apples/image) | 5.00 |
| Count bias | +1.24 (over-counting) |
| Count correlation | 0.905 |

Full breakdown: `results/baseline_colour_v1_val_summary.md` and
`results/baseline_colour_v1_val_matches.csv`. F1 moved only slightly
(0.588→0.560) despite 4x the tuning images — classical's threshold-search
approach appears to be close to its ceiling for this problem; contrast
with YOLO, which kept improving noticeably over the same growth in
labels (`docs/stage-8-notes.md`'s learning curve). A reasonable sign
that colour thresholding alone cannot keep closing the gap just by
tuning on more examples, which is useful context for the final Stage 9
method comparison.

**Why mAP@0.5 (0.200) is so much lower than F1 (0.560):** F1 uses the
lenient centre-distance match (within half the labelled radius); mAP@0.5
requires 50% box overlap (IoU), which the classical pipeline's circles
often miss even when well-centred, because of the next finding.

**The count-accuracy metric (added after a student request, see
`EXPERIMENTS.md`) is a different question from F1/mAP**: it asks whether
the *total number* of apples predicted per image is close to the true
count, regardless of whether individual apples were correctly localised.
Early on, classical briefly had *better* count accuracy than YOLO despite
losing on every per-apple metric — but that gap closed once a YOLO-side
evaluation bug (an overly loose NMS threshold letting near-duplicate boxes
through) was found and fixed; see `docs/stage-8-notes.md` for the full
story. As of the latest numbers, YOLO leads on count accuracy too.

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

## A failed experiment, kept in the writeup rather than deleted: multiple colour variants

The "shade kills recall" failure mode above is this detector's biggest
weakness, so the obvious next thing to try was recognising more than one
shade of red: a well-lit primary red (the original, grid-searched
detector), plus a darker/less-saturated "shadow" variant and a
brighter/more-orange "highlight" variant, OR'd together in
`segment_red_mask()`. Synthetic unit tests confirmed the *intent* worked
(caught hand-picked dark-red and orange-red test colours, still rejected
plain green and one soil colour) — but on the real dataset it was a
**severe regression**, not an improvement:

| Metric | Single colour (kept) | Multi-colour (tried, reverted) |
|---|---|---|
| Precision | 0.548 | 0.266 |
| F1 | 0.560 | 0.366 |
| False positives | 332 | 1134 |
| Count MAE | 5.00 | 33.68 |

Looking at the overlays explained why: the "highlight" variant's hue/
brightness range matches a tan/beige background wall visible in several
images almost exactly, and the broadened ranges generally pick up far
more soil/bark/dead-leaf clutter than the hand-picked synthetic test
colours anticipated. **Reverted** — `use_shadow_variant` and
`use_highlight_variant` now default to `False` in `ClassicalConfig`, and
`baseline-colour-v1`'s numbers above are back to the single-colour
result. The variant code itself is kept in `src/classical.py` (tested,
not deleted) rather than thrown away, in case someone wants to retighten
the thresholds against real images instead of guessed colours, or add
the toggles to the grid search so it's chosen empirically rather than
assumed. Recorded here and in `EXPERIMENTS.md` because a negative result
investigated properly is still a real finding worth keeping, not
something to quietly undo and forget (rule 9).

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
  more labels arrive, widening the search is worth revisiting.
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
