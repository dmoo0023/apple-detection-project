# Stage 10: Full inference and deliverables

## What this stage does

Every previous stage worked on a subset of the data (labelled images, a
train/val/test split). This stage runs the chosen, final method
(`model-v1`, confirmed at checkpoint C3) over **every one of the 544
trees** — including the 458 that were never labelled at all — and
produces the project's actual deliverable: a 3D position for every
detected apple in the whole dataset.

## What was built

1. **`scripts/run_pipeline.py`** — for one tree ID or all 544, runs the
   detector, samples depth, and computes each apple's 3D position via
   `src/geometry.py`'s `surface_point_3d()`. Writes
   `results/apples_3d.csv`: one row per detected apple — `tree_id`,
   `apple_index`, `u`, `v`, `radius_px`, `X`, `Y`, `Z`,
   `depth_pixels_used`, `is_valid`, `confidence`. A detection with no
   usable depth still gets a row (`X`/`Y`/`Z` empty, `is_valid=False`) —
   never silently dropped or zeroed (rule 4). **Column names are `X, Y,
   Z` per the brief's spec, but — per the checkpoint C2 scope decision —
   these are the apple's visible near-surface point, not a corrected
   centre.** Worth restating here since it's easy to forget looking at a
   544-tree CSV in isolation.
2. **`scripts/stage10_summary.py`** — apples-per-tree and distance (Z)
   histograms, with missing-depth counts.
3. **`scripts/stage10_point_clouds.py`** — a PLY point cloud (full scene,
   back-projected from depth, coloured from RGB) with each detected
   apple's surface point marked as a small red sphere, plus a matching 2D
   overlay, for 5 example trees.

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

**5 example trees**, deliberately including one edge case rather than
only clean examples:
- `R01U01`, `R08U01` — ordinary, moderate apple counts, all with usable depth.
- `R23U14` — the single highest-count tree in the whole dataset (94
  detections), to check the pipeline holds up on a dense, heavily-laden
  tree. Visual check: clean, well-separated detections across the full
  canopy.
- `R10U14` — **zero detections.** Checked by eye: the tree genuinely has
  no visible ripe red apples in frame (young/sparse tree, or an off
  angle) — the pipeline correctly produced nothing rather than a false
  detection, and the PLY/overlay for this tree are kept and shown as-is,
  not swapped for a "nicer" example.
- `R04U13` — a test-split tree, including the one apple from the Stage 9
  final evaluation that had no usable depth sample (see
  `docs/stage-9-notes.md`).

PLY files are regenerable from `scripts/stage10_point_clouds.py` and are
not committed to git (bulky binary scene data, same treatment as other
derived imagery in this project) — the 2D overlays (small, informative)
are kept in `results/point_clouds/`.

## What could go wrong later

- `results/apples_3d.csv` is large (19k+ rows) — anyone consuming it
  programmatically should filter on `is_valid` before using `X/Y/Z`,
  since invalid rows keep those columns empty by design, not zero.
- Every 3D number in this file inherits both the depth-scale
  approximation (checkpoint C1) and the surface-point bias (checkpoint
  C2) — restated here because this is the file most likely to be used
  directly by someone who hasn't read the earlier stage notes.

## Terms used

- **Point cloud:** a 3D scene represented as a set of individual points
  (here: one per valid depth pixel, back-projected to 3D and coloured
  from the matching RGB pixel) rather than a continuous surface — the
  standard way to visualise what a depth camera actually measured.
