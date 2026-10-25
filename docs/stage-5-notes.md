# Stage 5: Evaluation module

## What this stage does

Stage 4 answers "where is this apple in 3D?" for one apple. This stage
answers "how good are a whole image's (or dataset's) predictions?" —
turning a list of predicted apples and a list of labelled apples into
precision/recall/F1, detection mAP, and both 2D and 3D position error,
broken down by how far the apple was from the camera. Built once in
`src/evaluate.py` so colour masking (Stage 6), YOLO (Stage 8) and any
hybrid (Stage 9) are graded identically (rule 7).

## What was built

1. **Matching (`match_image` / `match_all`).** Before you can say "this
   predicted apple corresponds to that labelled apple," you need a rule for
   pairing them up. This uses the **Hungarian algorithm**
   (`scipy.optimize.linear_sum_assignment`) on pixel centre distance,
   computed separately for each image — it finds the *globally* cheapest
   one-to-one pairing between all predicted and labelled apples in an
   image, rather than each prediction just grabbing whichever labelled
   apple is nearest to it (which can produce avoidably bad pairings when
   two apples are close together, e.g. two detections both near two
   labels, greedily matched in the wrong order). A candidate pairing only
   counts as a real match if the distance is under `0.5 x the labelled
   apple's radius` — a predicted centre further off than that is treated
   as simply wrong, not "close enough."
2. **Detection metrics.** From the matches: true positives (matched),
   false positives (predictions nothing matched), false negatives (labels
   nothing matched) → precision, recall, F1. Separately, **mAP@0.5**: the
   standard object-detection metric, computed by ranking every prediction
   across the dataset by confidence score, greedily assigning each to its
   best same-image labelled box by **IoU** (intersection-over-union of the
   square detection boxes) if IoU ≥ 0.5, then taking the area under the
   resulting precision-recall curve.
3. **2D error.** For each matched pair: pixel distance between centres,
   then converted to centimetres using that apple's depth (`error_cm =
   pixel_error_in_x,y x depth / fx,fy`, summed in quadrature) — so the
   same pixel error is reported as a bigger real-world distance for a
   far-away apple than a close one, which is physically correct.
4. **3D error — explicitly a PROXY.** There's no physical measurement of
   where any apple actually is, so "ground truth" 3D position here means:
   take the *labelled* circle's centre, run it through the exact same
   `apple_front_point_3d` as the prediction, and measure the distance
   between the two resulting 3D front-centre points. This measures how much
   a prediction's 2D centre disagrees with the label's 2D centre, expressed
   in 3D — not distance to a physically measured apple.
   `summary_to_markdown()` prints this caveat in every single table it
   produces, so it can't be reported without the caveat by accident later.
5. **Breakdowns.** Every error metric is also reported split into the
   three capture-distance bands from the brief (0.4–0.7m, 0.7–1.0m,
   1.0–1.2m — using the labelled apple's own measured depth to decide
   which band it falls in), plus the share of matched apples within 2cm
   and within 3cm of the proxy 3D target.
6. **Output.** `per_match_errors()` returns one row per matched apple
   (CSV-able); `summarise()` + `summary_to_markdown()` produce the overall
   and per-band numbers as a short markdown table, ready to drop into a
   results file per method/split.

## Tests (the workflow's own exit check for this stage)

`tests/test_evaluate.py`, 5 tests:

- **Trivial predictor (predictions = the labels themselves) scores
  perfectly:** precision = recall = F1 = mAP@0.5 = 1.0, every 2D and 3D
  proxy error is (numerically) zero, 100% within 2cm and 3cm.
- **A deliberately shifted copy gives the expected error, not just "some"
  error:** every predicted apple shifted by a fixed (3, 4) pixel offset
  (5px, a 3-4-5 triangle so the expected distance is an exact, checkable
  number) — the test asserts the measured 2D pixel error equals `hypot(3,4)
  = 5.0` exactly, the 2D cm error equals the hand-computed
  `pixel_error x depth / focal_length` conversion, and the 3D proxy error
  is close to that same number (expected, since at a constant depth a
  2D shift maps straight onto the same sideways shift of the 3D point).
- A shift larger than the matching threshold correctly produces zero
  matches (one false positive, one false negative) rather than a
  nonsensically large "matched" error.
- Missing a prediction / adding a spurious extra prediction move precision
  and recall in the expected direction, with exact expected values.

## An interruption, handled mid-stage

While building this stage, the raw dataset's labelme file count changed
from 25 to **27** (the student labelled `R02U06` and `R02U07` during this
session). Per rule 1, raw data is read-only and was not touched — but
Stage 1 and Stage 2's *generated* files (`manifest.csv`, `apples_2d.csv`,
label files) were built before those 2 images existed, so they were
re-run to pick the new labels up (both new files validated cleanly; totals
are now 27 labelled images, 816 apples). `configs/splits.json` was
**not** regenerated in the sense of changing — it was re-run with
`--overwrite` specifically to confirm it reproduces byte-identical output
(the split only depends on row structure, which didn't change), and that
was verified with `git diff` showing no change, before trusting the
re-run. The 2 new labels landed in row 2, which is in the **train** split
(train now has 7 labelled images, up from 5; val and test unchanged at 20
and 0). Tests that had hard-coded "25 images, 781 apples" were changed to
check "every labelled image that currently exists converts cleanly,"
since that count is expected to keep growing — a hard-coded number would
have kept breaking the test suite for a reason that isn't actually a bug.

## Result

All project tests pass (50/50 as of the latest rerun). Evaluation module built, tested against its own
exit check (trivial predictor perfect, shifted copy predictable), ready for
Stage 6 onward to call.

## What could go wrong later

- mAP@0.5 uses IoU on the *square* detection boxes (same convention as
  Stage 2's YOLO export), not the circular labels directly — consistent
  with how a YOLO detector's output will be evaluated, but means a
  "perfect" circular match and a "perfect" square-box match aren't quite
  the same geometric statement. Worth remembering when comparing a
  segmentation-based method against a box-based one.
- The per-band depth used for binning comes from the **labelled** apple's
  robust depth, not the prediction's — intentional (so the binning doesn't
  shift around depending on how wrong a given method's detection is), but
  worth knowing if band counts look odd for a poorly-performing method.

## Terms used

- **Hungarian algorithm:** an algorithm that finds the minimum-total-cost
  way to pair up two equal-ish-sized groups of things one-to-one (here:
  predicted vs. labelled apples by distance) — globally optimal, not just
  "each one picks its nearest."
- **IoU (intersection over union):** the overlap area of two boxes divided
  by their combined area; 1.0 = identical boxes, 0.0 = no overlap.
- **mAP (mean Average Precision):** the standard object-detection accuracy
  metric — area under the precision-recall curve as the detector's
  confidence threshold is swept from strict to loose. "mAP@0.5" means a
  prediction only counts as correct if its IoU with the true box is at
  least 0.5.
