# Limitations

A consolidated list, pulled from the checkpoint decisions, experiments and
bugs found across Stages 0-8, for use when writing the final report
(Section 6's "Definition of done" calls for limitations to be stated
plainly). Each point is explained in full, with numbers and reasoning, in
the stage notes and `EXPERIMENTS.md` linked below — this file is the
compact version for pulling into a report, not a replacement for them.

## Depth & geometry (checkpoint C1)

- Depth scale (`0.001`, i.e. 1 raw unit = 1mm) is an accepted
  approximation, not a measured value — the original 16-bit depth source
  and the camera/capture setup were unavailable to verify it.
  (`docs/stage-1-notes.md`, `configs/camera.yaml`)
- Camera intrinsics (fx, fy, cx, cy) are derived from Intel's published
  D435i field-of-view spec, not measured on this specific unit — roughly
  ±5-8% error, from Intel's own stated ±3° manufacturing tolerance.
  (`docs/stage-1-notes.md`)
- Because of the above, real computed depths come out far below the
  dataset's own stated 0.4-1.2m capture range (every matched apple
  currently falls in an "out of range" distance band in the evaluation
  tables). (`docs/stage-6-notes.md`, `docs/stage-8-notes.md`)
- The 3D output is each apple's *front-centre point* (the point on its
  camera-facing surface at its 2D centre), so it only exists for apples
  the camera can see, and its accuracy is bounded by the depth scale and
  intrinsics above. (`docs/stage-4-notes.md`)
- Combined, the brief's "2-3cm" target cannot be verified in absolute
  terms against any 3D number this project reports.
- 3D error throughout is a **proxy**: no physical ground truth exists, so
  "3D error" means the distance between two points computed through the
  same geometry code (predicted circle vs. labelled circle) — not
  distance to a physically measured apple. (`src/evaluate.py`)

## Data & generalisation

- Dataset is red apples, summer only, single viewpoint per tree —
  findings don't generalise to other varieties, seasons, or multi-view
  capture.
- The method can only ever detect/count apples visible from that single
  viewpoint; anything occluded by leaves, branches, or other apples is
  invisible to the whole pipeline, which matters for both detection
  recall and any yield-estimation use.
- Early train/val were concentrated in just 2 adjacent orchard rows (same
  capture session); later runs added more rows and performance held
  up/improved, which is reassuring but still only 4 points on the
  learning curve — not firm proof of generalisation across the full
  23-row orchard. (`EXPERIMENTS.md`, `docs/stage-8-notes.md`)
- Test split, by design of the frozen split, can only ever be rows 4 and
  21 — no option to diversify further; final test-set size was
  constrained by labelling time.

## Model/experiment maturity

- Learning curve has only 4 points (10/20/31/40 train images); F1/mAP
  still rising, not clearly plateaued. (`results/learning_curve.png`)
- No seed-repeat runs yet — every YOLO run used a single fixed seed
  (42), so genuine trend vs. run-to-run noise can't yet be fully
  separated (Stage 8 item 4, not yet done).
- The 4th learning-curve point also changed the val set size (20→25
  images) alongside train count, so it isn't a perfectly isolated
  single-variable comparison against the earlier 3 points.
- Full Stage 8 experiment series (model size, pretrained vs. random
  init, LR/batch/epoch sweep, augmentation, boxes vs. segmentation) not
  yet run — only label count has been systematically varied so far.
- A recurring training-loss spike appears in every YOLO run; investigated
  at length (ruled out LR warmup schedule, confirmed it stays below
  Ultralytics' own NaN/Inf failure threshold) but the root cause isn't
  fully confirmed, only a plausible explanation (classification-head
  re-specialisation on a small dataset). (`docs/stage-8-notes.md`)

## Method-specific

- Classical colour-thresholding baseline appears close to its ceiling —
  F1 moved only slightly despite 4x the tuning data, while YOLO kept
  improving over the same growth. (`docs/stage-6-notes.md`)
- A classical multi-colour extension (catching shaded/highlighted apple
  skin) was tried and reverted after it caused a severe false-positive
  regression on real images — documented as a negative result, not
  deleted. (`EXPERIMENTS.md`)
- YOLO's box→circle conversion (radius = average of half-width/
  half-height) is an approximation, not an exact circle fit.
- Precision/recall/F1/mAP and apple-count accuracy (MAE/bias) can
  disagree, sometimes sharply, between methods — which one "wins"
  depends on the downstream use (picking needs individual positions,
  yield estimation needs the total count right).
