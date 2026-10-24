# Stage 1: Data audit and path checks

## What this stage does

Before writing any code that depends on the data (labels, 3D geometry,
training), everything about the raw files has to be checked and written
down: are the files really there, are they paired correctly, are they the
format we expect. Mistakes here are the kind that silently corrupt every
later stage, so this audit is systematic rather than a spot-check.

## What was done

`scripts/audit_data.py` reads only from `configs/paths.yaml` (never a
hard-coded path) and never writes into the raw data folder. For every file
in `RGB/` and `D/` it:

1. Confirmed the configured folders exist and that none of the 1,113 raw
   files are OneDrive "cloud-only" placeholders (checked via the Windows
   file-attribute flags that OneDrive sets on a file that looks present but
   isn't really downloaded — none were found).
2. Worked out the filename pattern as a regular expression
   (`R<row>U<unit>_<season>_<RGB|D>-<HH>-<MM>-<SS>.png`, e.g.
   `R01U01_Summer_RGB-12-40-11.png`) and confirmed **all 1,088** image files
   match it exactly, with no stragglers.
3. Extracted the row and unit numbers from each filename as the tree ID
   (e.g. `R03U02`) and confirmed exactly **544 distinct trees**, with no
   duplicate tree IDs within either the RGB or the depth folder.
4. Paired RGB and depth files by (row, unit, season): **544 of 544 pairs
   matched with no orphans**, and — better than expected — the RGB and depth
   timestamps agree exactly for every tree, so there was no ambiguity to
   break by "closest timestamp."
5. Checked for messy filenames (mixed case, double extensions, spaces,
   non-ASCII characters, paths over Windows' length limit): **none found**.
   The project folder path itself has spaces and parentheses
   (`Dataset (Summer Only)`), which is fine — every path in the code goes
   through Python's `pathlib`, which handles that correctly as long as you
   don't build paths by hand with string concatenation.
6. Opened every one of the 544 RGB PNGs and confirmed each is 640×480 with
   3 colour channels, as expected.
7. Opened every one of the 544 depth PNGs — **this is where a real problem
   turned up, see below**.
8. Parsed and validated all 25 labelme JSON label files: each one's
   `imagePath` resolves to a real image, `imageWidth`/`imageHeight` match
   the image, every shape is a `circle` with exactly 2 points, and no
   circle has a radius under 4 pixels or a centre outside the image. **All
   25 passed with no problems.** Together they contain 781 labelled apples.
9. Saved 10 overlays of depth blended over RGB (`data/derived/overlays/`)
   so alignment can be checked by eye. The depth and RGB line up correctly
   pixel-for-pixel (tree and ground shapes match between the two layers) —
   see "What this does NOT tell us" below for what it doesn't prove.
10. Wrote `configs/camera.yaml` with the one value that is known (image size,
    640×480) and explicit `null`s for everything that is not, rather than
    filling in RealSense factory defaults.

## The problem: depth PNGs are not what the brief expected

The brief said depth PNGs are "expected to be 16-bit" with an unverified
scale (RealSense default 1 unit = 1 mm). The audit found something
different: **every one of the 544 depth PNGs is an 8-bit, 3-channel image**
(red, green and blue channels all equal — i.e. a grayscale image saved in
RGB format), with pixel values from 0 to about 155–180 depending on the
image.

Why this matters: a real RealSense depth stream at 0.4–1.2 m, stored as raw
millimetres in a 16-bit image, would contain values roughly in the range
400–1200 — far more than 256 distinct values, and the *same* unit
(millimetres) in every image. What we actually have looks like a
**normalised visualisation**: each image's depth values were probably
rescaled (min depth → 0, max depth → 255, or similar) just for *display*,
which:

- throws away the real distances (an 8-bit normalised image cannot be
  inverted back to metric depth, because the rescaling is different for
  every image — we can't know what raw value 150 in image A corresponds to
  in millimetres, and it means something different in image B);
- would make the 2–3 cm accuracy target impossible to hit or even measure,
  since there would be no ground-truth depth to compare against.

This is exactly the kind of thing rule 2 says to stop and ask about rather
than guess past. It is possible that:
- the true 16-bit depth PNGs exist somewhere else (a different export, a
  different folder, a backup) and weren't copied into this dataset folder, or
- the capture pipeline was only ever saving a visualisation and the raw
  depth was never kept, or
- something else the student knows about that isn't visible from the files
  alone.

**This is the first checkpoint question below.**

## The second problem: no camera intrinsics anywhere

A search of the entire project folder (not just the dataset folder) found
no calibration file, README, or metadata of any kind containing the camera
intrinsics (fx, fy, cx, cy) or a depth scale. The labelme JSON files only
contain image dimensions and label shapes, nothing about the camera.
`configs/camera.yaml` has been created with the image size filled in and
every other field left as `null`, with a comment explaining why.

## What the audit does NOT tell us

The 10 overlays confirm *pixel alignment* — the depth image's content lines
up spatially with the RGB image's content (same tree shape, same ground
line). They do **not** confirm that the pixel *values* are in any particular
unit, nor which direction brighter means (closer or farther) — that's part
of the same open question above.

## Manifest columns

`data/derived/manifest.csv` has one row per tree: `tree_id`, `row`, `unit`,
`rgb_path`, `depth_path`, `rgb_size`, `depth_size`, `depth_dtype`,
`depth_fraction_invalid` (share of zero pixels, the usual "no return"
sentinel for depth sensors), `depth_median_raw`/`depth_min_raw`/
`depth_max_raw` (raw 0–255 pixel values — **not** metres, since the scale
isn't known), and `has_labelme_json`.

## Terms used

- **16-bit vs 8-bit image:** how many distinct brightness levels a pixel can
  hold — 16-bit allows 65,536 levels (enough to store millimetres up to
  65 metres), 8-bit only allows 256.
- **Depth scale:** the number you multiply a raw depth pixel value by to
  get a real-world distance in metres (e.g. RealSense's common default is
  "1 unit = 1 mm", so scale = 0.001).
- **Intrinsics (fx, fy, cx, cy):** the numbers that describe how a camera's
  lens and sensor map a 3D point in front of it to a 2D pixel. fx/fy are
  focal lengths in pixels; cx/cy are the pixel coordinates of the optical
  centre. Needed to convert a labelled pixel + depth into a real 3D point
  (Stage 4).
- **OneDrive placeholder:** a file OneDrive shows in File Explorer but has
  not actually downloaded to disk yet ("Files On-Demand"); reading it
  triggers a download, or fails if offline.

## Result

Structurally, the dataset is in excellent shape: 544/544 trees, perfect
pairing, clean filenames, no renames needed, all 25 labels valid. But the
pipeline cannot proceed to Stage 2 onward in a way that would hit the 2–3 cm
3D accuracy target without resolving the two questions below — this is
checkpoint **C1**.
