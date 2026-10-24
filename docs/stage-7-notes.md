# Stage 7: Pre-labelling loop

## What this stage does

Labelling by hand is slow — the student has produced 27 labelled images
over the project so far. The idea here: let the current best detector
(Stage 6's classical baseline) propose circles on **unlabelled train
images**, and have the student *correct* those proposals in labelme
instead of drawing every circle from scratch. Correcting a roughly-right
guess is faster than starting from a blank image, as long as it's clear
these proposals are not trustworthy until a human has checked them.

## Why only train images

Val and test images are never pre-labelled. If a detector's own blind
spots silently became the "ground truth" used to score that same
detector, the evaluation would be biased in the detector's favour exactly
where it's weakest — the opposite of what val/test are for. Only train
images get this shortcut; val and test still need labels drawn from
scratch.

## What was built and run

1. `src/labels.py` gained `write_labelme_json()` (and the dict-building
   helper under it) — turns a list of (cx, cy, radius) detections into a
   file with the exact structure `load_labelme()` reads back, so a
   pre-label is indistinguishable from a hand-made one once the student
   finishes correcting it. Round-trip tested
   (`test_write_labelme_json_round_trips_through_extract_apples`).
2. `scripts/pre_label.py` loads Stage 6's tuned config
   (`results/baseline_colour_v1_config.json`), runs the classical detector
   on unlabelled **train**-split images (`configs/splits.json`), and
   writes one labelme JSON per image into `data/derived/prelabels/` — a
   *separate* folder, never the raw dataset root (rule 1).
3. Ran it for a first batch of 50 images (of 376 unlabelled train trees).
   **6 to 48 detections per image, median 25 — zero images came back with
   no detections at all**, which is a reasonable first sanity check (if
   many had come back empty, that would suggest the detector is
   completely failing on some entire images rather than just missing some
   apples within them, worth investigating before handing 50 images to
   correct).

## What the student needs to do next (this is where the stage waits)

1. Open each `.png`/`.json` pair in `data/derived/prelabels/` with labelme.
2. Correct the circles: Stage 6 showed this detector has **40% recall**
   (it misses more apples than it finds) and undersizes circles on
   leaf-occluded apples — expect to add a fair number of missed apples and
   resize/reposition a fair number of existing circles, not just spot-fix
   a few.
3. Copy each corrected `.json` into the same folder as its `.png` in the
   **real** RGB folder (not back into `data/derived/prelabels/` — that
   folder is scratch space, never read by the rest of the pipeline).
4. Ask the agent to re-run `scripts/audit_data.py` and
   `scripts/convert_labels.py` (same as happened automatically mid-Stage-5
   when 2 ad hoc labels were added) to pick the new labels up.

## Suggested label-count milestones (from the brief)

~25, ~75, ~150, ~250 total labelled images, each a point on Stage 8's
learning-curve plot. Currently at 27 (7 train / 20 val / 0 test). This
batch of 50, once corrected, would bring train alone to 57 — a reasonable
step toward the ~75 milestone, not a precise hit, since the milestones are
approximate by design. **Still outstanding from Stage 3:** the test split
has zero labelled images, and per rule 3 those must be labelled by hand,
not pre-labelled (pre-labelling test would contaminate the one dataset
that's supposed to stay completely untouched by any model's influence).

## What could go wrong later

- A batch where many images come back with 0 or near-0 detections would be
  a sign to investigate the detector (or that batch of images) before
  asking the student to correct 50 blank proposals from scratch.
- `data/derived/prelabels/` will keep growing if `pre_label.py` is run
  again before the current batch is reviewed — consider clearing it out
  (or using `--batch-size` to target specifically what's still needed)
  once corrected files have been moved into the real dataset.

## Terms used

- **Pre-labelling / human-in-the-loop labelling:** using a model's own
  predictions as a draft for a human to correct, rather than labelling
  from a blank image — faster, but only as good as the human's review, so
  it must never be mistaken for a verified label before that review
  happens.
