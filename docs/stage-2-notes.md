# Stage 2: Label conversion

## What this stage does

The 25 labelled images store each apple as a labelme "circle" — just a
centre point and one point on the edge. Nothing downstream can read that
format directly: the detector (YOLO, Stage 8) needs boxes or polygons in
its own normalised text format, and the geometry module (Stage 4) needs a
plain (centre, radius) pair per apple. This stage converts once, in one
place (`src/labels.py`), so every later stage reads the same converted
labels instead of each re-parsing labelme JSON its own way.

## What was done

1. `extract_apples()` reads a labelme JSON file and returns one `Apple`
   (index, cx, cy, radius — all in pixels) per circle shape, plus a list of
   warnings for: a radius under 4 px, a circle centre outside the image, or
   an image with zero apples. None of the 25 images triggered any warning.
2. `yolo_detection_line()` turns one apple into a YOLO **detection** label:
   a square box, `2 x radius` on each side, centred on the apple, as the
   class-index + normalised `(x_center, y_center, width, height)` line that
   format requires. If the box would extend past the image edge (an apple
   near the border), it's clipped to the image bounds rather than writing
   an invalid negative coordinate.
3. `yolo_segmentation_line()` turns one apple into a YOLO **segmentation**
   label: the circle approximated as a 16-sided polygon (16 points evenly
   spaced around the circle), also normalised and edge-clipped.
4. `scripts/convert_labels.py` runs both converters over all 25 labelled
   images, writing:
   - `data/derived/labels_det/<image_id>.txt` — one file per image
   - `data/derived/labels_seg/<image_id>.txt` — one file per image
   - `data/derived/apples_2d.csv` — one row per apple: `image_id`,
     `apple_index`, `cx`, `cy`, `radius_px`
   - `data/derived/stage2_overlays/` — 10 random images with the labelme
     circle (red), the YOLO box (green) and the centre (blue cross) all
     drawn on top, for a quick visual sanity check
5. `tests/test_labels.py` includes a **round-trip test**: convert an
   apple to a YOLO line, parse that line back into (cx, cy, radius), and
   check it matches the original to within 0.01 px. This catches, for
   example, an off-by-one in how a box's corners are turned back into a
   centre and width. A second test documents the one case where round-trip
   *won't* match exactly on purpose: an apple near the image edge, where
   the box is clipped and the recovered centre/radius legitimately differ
   from the original (this is correct behaviour, not a bug, and the test
   exists so nobody "fixes" it later by accident).

## Result

All 25 labelled images converted with **zero warnings** (781 apples total).
The overlays confirm the green YOLO box and red labelme circle land exactly
on each labelled apple. 16/16 tests pass, including the round-trip checks.

## What could go wrong later

- The 16-point polygon is a coarse approximation of a circle — fine for a
  detector's segmentation mask, but don't use it as a precise geometric
  circle fit (Stage 6's classical pipeline does its own circle fitting from
  pixels, independently).
- Once more images are labelled, some of them
  (being machine-generated and student-corrected) may trigger the warnings
  this stage already checks for (tiny radius, out-of-bounds centre, empty
  image) — `scripts/convert_labels.py` will report them; they should be
  looked at, not silently converted.

## Terms used

- **YOLO detection label:** a text file format where each line is
  `class x_center y_center width height`, all four position numbers scaled
  to 0–1 by dividing by the image width or height (so the same label file
  works regardless of how the image is later resized).
- **YOLO segmentation label:** the same idea but with a polygon
  (`class x1 y1 x2 y2 ...`) instead of a box, for models that predict a
  mask rather than just a bounding box.
- **Round-trip test:** convert data from format A to format B, then back to
  A, and check it matches the original — a strong way to catch conversion
  bugs that a one-directional check would miss.
