# Stage 10: Full inference and deliverables

## What this stage does

Every previous stage worked on a subset of the data (labelled images, a
train/val/test split). This stage runs the chosen, final method
(`model-v1`, confirmed at checkpoint C3) over **every one of the 544
trees** — including the 458 that were never labelled at all — and
produces the project's actual deliverable: the 3D front-centre point of
every detected apple in the whole dataset.

## What was built

1. **`scripts/run_pipeline.py`** — for one tree ID or all 544, runs the
   detector, samples depth at each apple's 2D centre, and back-projects
   the apple's front-centre point via `src/geometry.py`'s
   `apple_front_point_3d()`. Writes `results/apples_3d.csv`: one row per
   detected apple — `tree_id`, `apple_index`, `u`, `v`, `X`, `Y`, `Z`
   (front-centre point, metres), `depth_pixels_used`, `is_valid`,
   `confidence`. A detection with no usable depth still gets a row
   (`X`/`Y`/`Z` empty, `is_valid=False`) — never silently dropped or
   zeroed (rule 4).
2. **`scripts/stage10_summary.py`** — apples-per-tree and distance (Z)
   histograms, with missing-depth counts.
3. **`scripts/plot_apples_3d.py`** — the report figure: for each example
   tree, a 3D graph of its apples' front-centre points in the camera frame
   (axes in metres, X right, Y down, Z depth; equal scaling; camera marked
   at the origin; same viewing angle for every tree; each point labelled
   with its `apple_index`). Also saves the tree's photo with each apple's
   2D centre and index, and a side-by-side version of the two. Output in
   `results/figures/` as 300 dpi PNG and PDF. Each tree has its own camera
   frame, so trees are never combined in one graph.

## Result

**19,294 apples detected across all 544 trees** (`results/apples_3d.csv`).
Apples per tree: min 0, median 36, mean 35.5, max 94 — see
`results/stage10_histograms.png`. 141 detections (0.7%) had no usable
depth sample and are correctly marked `is_valid=False` rather than given
a fabricated position.

**100% of valid Z values fall below the dataset's stated 0.4m minimum
capture distance** (median 0.144m) — the checkpoint-C1 depth-scale
limitation, now visible across the *entire* dataset, not just the small
labelled subset it was first found in. This was expected, not a new
discovery, and is the clearest single piece of evidence in the whole
project for why `docs/limitations.md`'s depth-scale caveat has to travel
with every 3D number this pipeline ever produces.

**5 example trees** in `results/figures/`, deliberately including one edge
case rather than only clean examples:
- `R01U01`, `R08U01` — ordinary, moderate apple counts (13 and 33).
- `R23U14` — the single highest-count tree in the whole dataset (94
  detections), to check the pipeline holds up on a dense, heavily-laden
  tree.
- `R10U14` — **zero detections.** Checked by eye: the tree genuinely has
  no visible ripe red apples in frame — the pipeline correctly produced
  nothing rather than a false detection, and its (empty) figure is kept
  and shown as-is, not swapped for a "nicer" example.
- `R04U13` — a test-split tree, including the one apple from the Stage 9
  final evaluation that had no usable depth sample (see
  `docs/stage-9-notes.md`); its 2 no-depth detections are counted under
  the graph.

## What could go wrong later

- `results/apples_3d.csv` is large (19k+ rows) — anyone consuming it
  programmatically should filter on `is_valid` before using `X/Y/Z`,
  since invalid rows keep those columns empty by design, not zero.
- Every 3D number in this file inherits the depth-scale approximation
  (checkpoint C1) — restated here because this is the file most likely to
  be used directly by someone who hasn't read the earlier stage notes.
- On very dense trees the index labels in the 3D graph can overlap; the
  side-by-side figure with the photo is the easier one to read there.

## Terms used

- **Front-centre point:** the point on an apple's camera-facing surface at
  its 2D centre pixel, placed in 3D using the depth at that spot.
- **Camera frame:** a 3D coordinate system with the camera at the origin,
  X to the right, Y down and Z straight ahead — each tree's photo has its
  own.
