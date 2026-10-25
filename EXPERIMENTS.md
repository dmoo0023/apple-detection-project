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
this should be revisited once more labels exist.

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

## model-v1-20train — Stage 8 second YOLO fit (learning curve, point 2)

**Date:** 2026-10-24 (same day)
**What:** Retrained after the student added 10 more train labels (20 train
images total, was 10). Same setup as `model-v1-first-fit` except
batch=8 (was 4, since there's now more data to batch) — one change
(label count, the thing the learning curve is tracking), plus the batch
size bump which is noted rather than hidden, since it technically means
this isn't a perfectly isolated single-variable comparison (rule 5 in
spirit: flag it rather than pretend it's cleaner than it is).

**Result (val, through `src/evaluate.py`):** precision 0.736, recall
0.806, F1 **0.770**, mAP@0.5 **0.786**, count MAE **4.20**, count bias
**+2.70** — improved on every metric versus the 10-image run (F1 0.744,
mAP 0.756, count MAE 7.00, count bias +6.40). The count-accuracy
improvement is the standout: more training data specifically closed the
gap that made classical look better on counting at 10 images, and now
YOLO's count MAE (4.20) beats classical's (5.15) too.

**Learning curve:** `results/learning_curve.png` /
`results/learning_curve.csv`, generated by `scripts/plot_learning_curve.py`
from every stage-8 run logged in `runs/experiments.csv`. **Only 2 points
— not yet a trend**, but consistent and encouraging: nothing got worse
going from 10 to 20 images.

**Training-curve anomaly, corrected:** `model-v1-first-fit`'s stage-8 note
guessed a validation-loss spike partway through training was caused by
small batch size + few val images. This run doubled both the batch size
and train images, and the spike recurred (worse, not better) — see
`docs/stage-8-notes.md` for the corrected, honest "cause not yet
identified" status.

**Tag:** none yet (same reasoning as `model-v1-first-fit` — `model-v1` is
reserved for the full Stage 8 exit check).

## model-v1-31train — Stage 8 third YOLO fit (learning curve, point 3)

**Date:** 2026-10-24 (same day)
**What:** Retrained after the student added 21 more labels (61 total: 31
train / 20 val / **10 test — first test labels**, not used here or
anywhere, per rule 3). Same setup as `model-v1-20train` (batch=8).

**Result (val):** precision 0.704, recall 0.837, F1 0.765, mAP@0.5
**0.814** (best yet), but count MAE **6.00** and bias **+5.40** — both
*worse* than the 20-image run (4.20 / +2.70). **Non-monotonic, reported
honestly, not smoothed over**: detection quality kept improving, count
accuracy didn't. With one run per label-count point, there's no way yet to
tell a genuine effect (e.g. the new row-3 images being harder in some
way) from plain run-to-run noise — this is precisely the gap Stage 8 item
4 (seed repeats) exists to close, and it's now a higher priority than it
was.

**Training-curve spike, now 3/3 runs — investigated properly, not just
flagged again.** Read Ultralytics' own installed source
(`trainer.py`): confirmed validation runs every epoch unconditionally
(`args.val=True`), ruled out the LR-warmup-schedule hypothesis with real
numbers (predicted warmup-end stays ~25-33 across all three runs; actual
spike keeps moving earlier), and checked saved training logs for
Ultralytics' own built-in NaN/Inf recovery warning — **never fired in any
run**, meaning these spikes stay below the threshold Ultralytics itself
treats as a real training failure. Also corrected an overstatement from
the first pass: comparing actual minimum precision/recall/mAP (not raw
loss magnitude, which is unbounded) shows the three runs' worst points are
comparably bad, not escalating — the "~100 → ~230 → ~400" trend was partly
an artifact of unbounded cross-entropy loss values, not genuinely
worsening model behaviour. Full writeup: `docs/stage-8-notes.md`.

**Tag:** none yet.

## NMS IoU threshold tightened (0.7 -> 0.3) — applied to all 3 YOLO runs

**Date:** 2026-10-24 (same day)
**What:** Student asked whether a loss-function overlap penalty would help
the over-counting problem. Investigated and found YOLO already has that
mechanism — NMS (non-max suppression), which discards a predicted box that
overlaps a higher-confidence box by more than a threshold. Ultralytics'
default (0.7) is loose enough that near-duplicate boxes on the same apple
both survive. Tightening to 0.3 (pure prediction-time setting, no
retraining) cut `model-v1-31train`'s count MAE from 6.00 to 2.60 with F1
unchanged or slightly better. Made the standing default in
`scripts/train_yolo.py` and `scripts/make_yolo_overlays.py`.

**Re-evaluated all 3 logged runs with it, for a clean comparison:**

| Run | Train images | F1 | mAP@0.5 | Count MAE (was, iou=0.7) | Count MAE (now, iou=0.3) |
|---|---|---|---|---|---|
| `model-v1-first-fit` | 10 | 0.767 | 0.744 | 7.00 | 3.25 |
| `model-v1-20train` | 20 | 0.785 | 0.780 | 4.20 | 3.00 |
| `model-v1-31train` | 31 | 0.779 | 0.811 | 6.00 | 2.60 |

**This substantially resolves the "non-monotonic count regression"
reported in the `model-v1-31train` entry above.** With the NMS fix
applied consistently, count MAE is now monotonically improving
(3.25 → 3.00 → 2.60) rather than dipping and bouncing back — the earlier
"regression" was largely an evaluation-side artifact (inconsistent
over-counting severity across runs), not a genuine step backwards in the
model. Old (iou=0.7) results/overlays kept, not deleted, in
`*_iou0.7`-suffixed files — see `docs/stage-8-notes.md` for the full
writeup, including the updated classical-vs-YOLO table (YOLO now wins on
every metric, including count accuracy, which it previously didn't).

Fixed a related bug while doing this: `scripts/plot_learning_curve.py`
used `drop_duplicates(keep="first")`, which would have kept each run's
*original* (stale, pre-fix) logged metrics instead of the re-evaluated
ones. Changed to `keep="last"`.

## Multi-colour classical detector — tried, reverted (negative result kept)

**Date:** 2026-10-24 (same day)
**What:** Student asked to make the classical detector recognise multiple
colours of apple skin, clarified as: broaden the single "well-lit red"
threshold to also catch shaded/dark-red and bright/orange-highlight
apple skin (the direct fix for Stage 6's biggest failure mode, "shade
kills recall"). Implemented as two extra HSV+Lab colour variants in
`src/classical.py`, OR'd with the existing grid-searched primary variant.
Validated with synthetic unit tests (12/12 passing): caught hand-picked
dark-red and orange-red test apples, still rejected plain green leaf and
one soil colour.

**Result on real data: a severe regression, not an improvement.**
Same train/val split (40/25), re-tuned grid search:

| Metric | Single colour | Multi-colour |
|---|---|---|
| F1 | 0.560 | **0.366** |
| Precision | 0.548 | **0.266** |
| False positives | 332 | **1134** |
| Count MAE | 5.00 | **33.68** |

Inspecting overlays showed why: the "highlight" variant's hue/brightness
range matches a tan/beige background wall in several images almost
exactly, and both new variants pick up far more real-world soil/bark/
dead-leaf clutter than the hand-picked synthetic test colours anticipated
— the synthetic tests validated intent, not real-world specificity.

**Reverted**, with the student's explicit go-ahead after seeing the
numbers: `use_shadow_variant`/`use_highlight_variant` now default to
`False`, restoring `baseline-colour-v1`'s known-good numbers exactly
(verified: re-running after the revert reproduced F1=0.560, count
MAE=5.00 bit-for-bit). Code kept, not deleted — available for future
retightening against real data, or for adding the toggles to the grid
search so it's chosen empirically. A process gap also worth noting: the
grid search never tested "variants off" as an option in the first place
(the new toggles weren't added to its search space), so this result
needed a manual before/after rather than the grid search itself flagging
the regression — worth fixing if this is revisited.

**Tag:** none (reverted before being adopted).

## model-v1-40train — Stage 8 fourth YOLO fit (learning curve, point 4)

**Date:** 2026-10-24 (same day)
**What:** Retrained after the student added 16 more labels across 6 new
rows (77 total: 40 train / **25 val** / 12 test), deliberately spread
widely (3-4 images per row) rather than concentrated, specifically to
test the domain-similarity concern from `model-v1-first-fit`'s entry.
Also re-tuned and re-ran `baseline-colour-v1` on the same new train/val
sets, for a fair comparison.

**Not a clean single-variable change**: val grew too (20→25, rows 8 and 9
added), so this isn't purely "train count changed, everything else held."
Flagged rather than hidden — see `docs/stage-8-notes.md`.

**Result (val, through `src/evaluate.py`, iou=0.3):** precision 0.785,
recall 0.844, F1 **0.813**, mAP@0.5 **0.827**, count MAE 3.00, count bias
+2.12 — F1 and mAP both took their **biggest jump yet** (from 0.779/0.811
at 31 images), on a val set that now includes 2 rows the model had never
seen in training or validation before. **This is real evidence against
the domain-similarity concern**, not just hope that it isn't a problem:
if the earlier strong numbers were mostly the model recognising this
orchard's early rows rather than apples in general, performance should
have dropped on the new rows, not improved.

**Classical, re-tuned on the same 40/25 split:** F1 0.560, mAP@0.5 0.200,
count MAE 5.00 — essentially unchanged from its 10-image numbers (F1
0.588) despite 4x the tuning data, suggesting classical's threshold-search
approach is close to its ceiling for this problem, unlike YOLO which kept
improving. YOLO now leads classical on every single metric at this point.

**Count MAE ticked up slightly** (2.60→3.00) alongside the val-set change
— not read as a new regression, since there's no clean "only one thing
changed" comparison here (unlike the 31-image dip, which was isolated to
an NMS artifact). Worth checking again once val holds steady.

**Tag:** none yet.

## Stage 8 experiment series (exp-01 through exp-07) — Stage 8 exit check complete

**Date:** 2026-10-24 (same day)
**Baseline for every experiment below:** `model-v1-40train` (yolov8n.pt,
pretrained, batch=8, epochs=100, lr0=0.01, no cosine schedule, mosaic=1.0),
re-evaluated first on a val set grown to 26 images (one more label arrived,
R09U03) so every comparison below uses a consistent val set: **F1=0.812,
mAP@0.5=0.830, count MAE=2.96**. Each experiment changes exactly one
thing from this baseline (rule 5), on its own branch (`exp/01-...` through
`exp/07-...`), kept unmerged since none became the new standing default
(rule: merge only if it becomes the new baseline). Full run artifacts
(weights-excluded logs/plots/configs) are on those branches; summarised
here.

| # | Change | F1 | mAP@0.5 | Count MAE | Verdict |
|---|---|---|---|---|---|
| — | baseline | 0.812 | 0.830 | 2.96 | — |
| 01 | model size: yolov8s vs yolov8n | 0.813 | 0.826 | 2.85 | no meaningful difference |
| 02 | random init vs pretrained | 0.724 | 0.656 | 4.04 | pretrained clearly wins |
| 03 | lr0=0.001 + cosine schedule | 0.801 | 0.824 | 3.42 | slightly worse, keep default |
| 04 | batch=16, epochs=150 | 0.814 | 0.801 | 2.73 | mixed (F1 ties, mAP worse) |
| 05 | mosaic augmentation off | 0.818 | 0.800 | 2.62 | mixed (F1 up, mAP worse) |
| 06 | segmentation vs detection | 0.675 | 0.654 | 2.12 | mask-centroid centre worse than box-centre |

**exp-01 (model size):** yolov8s has ~3.5x yolov8n's parameters; no payoff
at 40 train images — more capacity needs more data to show benefit. Keep
nano: same performance, smaller, faster.

**exp-02 (pretrained vs random init):** the clearest result in the whole
series. Random-initialised weights lose on every metric, confirming
COCO pretraining is doing real work transferring general visual features
to this tiny dataset — not a close call.

**exp-03 (LR/schedule):** a lower LR with cosine decay converges more
cautiously; on only 100 epochs and 40 images, that caution cost accuracy
rather than helping it. Ultralytics' defaults (tuned on large datasets,
admittedly not this one) still won here.

**exp-04 (batch/epochs) and exp-05 (augmentation):** grouped in this
summary because they produced the *same pattern* independently — F1 ticks
up, count MAE improves, but mAP@0.5 drops, and precision rises while
recall falls (the model gets more conservative: fewer false positives,
more misses). Seeing this pattern twice from two unrelated changes
suggests it may be a general property of training on this small a dataset
(e.g. less exposure to hard/varied examples per epoch), not a coincidence
specific to either change — worth retesting once the dataset is bigger.
Neither counts as a clean win given mAP (the standard detection metric)
got worse both times.

**exp-06 (boxes vs segmentation):** answers the brief's specific
"compare centre accuracy" question directly. Ultralytics' own box metrics
for the segmentation model (mAP50=0.884) show its underlying detection
quality is fine, comparable to the box-only baseline — the problem is
specifically the **mask-to-circle conversion**: centroid of a predicted
mask polygon is a worse centre estimate than a box's centre (2D error
3.16px vs 1.78px) when the visible apple shape is irregular, as it often
is under leaf occlusion. **Finding: use box-based detection for centre
position, not segmentation**, at least with this simple centroid
conversion — a smarter mask-to-circle method (e.g. a circle fit to the
mask contour, matching the classical pipeline's own technique) might close
this gap and would be worth trying if segmentation is revisited.

## exp-07: seed repeats (Stage 8 item 4)

Per the brief, repeated the best two configurations across 3 seeds each
(42, 123, 7) and compared mean ± std, rather than trusting single runs:

| Config | F1 (mean±std) | mAP@0.5 (mean±std) | Count MAE (mean±std) |
|---|---|---|---|
| baseline (mosaic on) | 0.810 ± 0.006 | 0.820 ± 0.017 | 2.80 ± 0.46 |
| no-mosaic | 0.823 ± 0.008 | 0.802 ± 0.004 | 2.72 ± 0.21 |

**Genuinely nuanced, reported as such rather than forced into a winner:**
no-mosaic has a modest, fairly consistent F1 edge (difference > either
std) and a *more stable* mAP (lower variance) — but baseline's *mean* mAP
is still higher. Baseline's own mAP varies a lot across seeds (one seed
scored 0.801, another 0.832), which on its own is worth knowing: a single
baseline run could look better or worse than it "typically" is just from
seed luck. **Decision: keep `mosaic=1.0` (the baseline) as the standing
config**, since mAP@0.5 is the more standard detection metric and its
mean is still ahead — but this is a closer call than any single-seed
comparison suggested, and documented as such rather than overstated.

## Error analysis (Stage 8 item 6)

`scripts/error_analysis.py` on `model-v1-40train`'s val predictions:
**120 false negatives, 175 false positives** across 25 of 26 val images
(overlays in `data/derived/error_analysis/model-v1-40train/`). False
positives' confidence scores are mostly low-to-moderate (mean 0.44, median
0.43, max 0.78) — none are confidently wrong, suggesting a higher
confidence threshold at inference time would trade some recall for
meaningfully fewer false positives, a cheap lever not yet tried.

Grouped by cause, from inspecting the overlays:
- **Clustered/touching apples (false negatives)** — the dominant missed-
  detection pattern. Where multiple apples overlap or sit very close
  together, the model often finds only one of the group or none at all.
- **Background structures (false positives)** — a tan/beige background
  wall, visible in several images, occasionally gets detected as an apple
  (same root cause as the classical baseline's worse version of this
  problem, just far less frequent here).
- **Ground-level false positives** — a handful of detections in grass/soil
  near the tree base, most plausibly fallen (unlabelled) apples, same
  pattern noted for the classical baseline.
- **Distant background rows (false positives)** — a failure mode not seen
  in the classical baseline's overlays: out-of-focus apple-like colour in
  a different orchard row visible far behind the main tree, occasionally
  triggers a detection.

## Stage 8 exit check: COMPLETE

Logged experiments (6 one-change comparisons + seed repeats, this file
and `runs/experiments.csv`), the learning-curve plot (4 points,
`results/learning_curve.png`), best configuration with seed spread (above),
and error analysis (above) are all done. **Standing best configuration:
`model-v1-40train`'s settings** (yolov8n.pt, pretrained, batch=8,
epochs=100, default LR/schedule/augmentation) — no experiment in the
series beat it convincingly enough to replace it, and the ones that came
close involve real tradeoffs (precision vs recall, F1 vs mAP) rather than
a clean win. Tagging `model-v1` and `stage-8-done`.

## Stage 9 decision table — checkpoint C3

**Date:** 2026-10-24 (same day)
**What:** Classical vs YOLO on val, identical evaluation code, plus speed
and robustness (`scripts/stage9_decision_table.py`, `docs/stage-9-notes.md`).

| Method | F1 | mAP@0.5 | sec/image |
|---|---|---|---|
| Classical | 0.548 | 0.204 | 0.021 |
| YOLO (`model-v1`) | **0.812** | **0.830** | 0.055 |

Not a close call on accuracy; classical's only edge is a small absolute
speed advantage, both fast enough in practice. A hybrid (detector box +
classical circle fit) was considered and declined — exp-06 already found
box-centre beats mask-centroid under occlusion, and YOLO's accuracy lead
is wide enough that a hybrid's only plausible benefit (speed) isn't worth
the added complexity here. **Checkpoint C3: student confirmed YOLO.**
Proceeding to the one-time locked test evaluation.

## FINAL TEST EVALUATION — `final-test-eval`

**Date:** 2026-10-24 (same day)
**What:** The one and only evaluation of YOLO (`model-v1`) on the locked
test split (20 images, rows 4+21 — never seen in training or validation).
Run on a clean git tree (`refuse_if_dirty()`), `scripts/run_final_test_eval.py`.

| | Val | **Test (final)** |
|---|---|---|
| F1 | 0.812 | **0.757** |
| mAP@0.5 | 0.830 | **0.755** |
| Count MAE | 2.96 | 4.55 |

Lower than val on every metric, reported as-is — a real generalisation
gap between rows seen during development and rows never looked at until
this run, not something to retune away. Still well above collapse (F1,
mAP both > 0.75). One apple (`R04U13`) had no usable depth sample;
correctly reported as `None`/unknown rather than fabricated. Full
reasoning in `docs/stage-9-notes.md`. **No tuning after this result, per
the brief.** Tagged `final-test-eval`.

## Stage 10: full inference, all 544 trees

**Date:** 2026-10-24 (same day)
**What:** `scripts/run_pipeline.py` run with `--all` — the chosen method
(`model-v1`) over every tree, not just the labelled subset.
`scripts/stage10_summary.py` and `scripts/stage10_point_clouds.py` for
the required summary/histograms and 5 example-tree deliverables.

**Result:** 19,294 apples across all 544 trees (0-94 per tree, median
36). 0.7% of detections had no usable depth, correctly flagged rather
than fabricated. **100% of valid Z values fall below the dataset's
stated 0.4m minimum** — the checkpoint-C1 limitation, now visible across
the whole dataset. A real bug caught while building the summary script:
a plain `groupby` on tree_id silently drops any tree with zero
detections from the "apples per tree" count — fixed by reindexing against
the full manifest, surfacing one such tree (`R10U14`) explicitly rather
than undercounting "trees processed" by one. That tree was deliberately
kept as one of the 5 example trees (checked by eye: genuinely no visible
apples) rather than swapped for a cleaner-looking example. Full writeup:
`docs/stage-10-notes.md`. Tagging `stage-10-done`.

## Front-point cleanup and full rerun

**Date:** 2026-10-25
**What:** Geometry wording, tests and outputs now describe only the 3D
front-centre point of each apple (no size/shape reconstruction anywhere);
PLY point clouds replaced by a 3D graph report figure
(`scripts/plot_apples_3d.py`). Everything downstream of training was
rerun; no model was retrained (detection does not use depth).

**Result:** 3D outputs unchanged (all 19,294 X/Y/Z values identical). The
learning curve is now measured on one fixed 26-image val set for every
point (previously a mix of 20- and 25-image val sets):
F1 0.753 / 0.781 / 0.778 / 0.812 and mAP 0.742 / 0.779 / 0.806 / 0.830 at
10 / 20 / 31 / 40 train images. Classical baseline on the same val set:
F1 0.548, mAP 0.204. Stage 9 decision table numbers unchanged (it was
already on the 26-image val set). See `docs/stage-8-notes.md` and
`docs/stage-10-notes.md`.

## Stage 7 removed (not used)

**Date:** 2026-10-25
**What:** The pre-labelling stage was removed from the repository. Its
pre-labels were never used: every label was drawn by hand (verified: 0 of
the hand-drawn circles on the 27 overlapping images match a pre-label
circle). No results change.
