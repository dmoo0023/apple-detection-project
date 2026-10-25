# Stage 4: Shared 3D module

## What this stage does

Every method in this project — colour masking (Stage 6), YOLO (Stage 8),
any hybrid (Stage 9) — ends up at the same question: "given a detected
apple (a 2D pixel centre) and a depth image, where is that apple in 3D?"
`src/geometry.py` answers it once, so every later stage calls the same
functions, and results across methods are directly comparable.

## What the 3D output is

The output for each apple is its **front-centre point**: the point on the
apple's front (camera-facing) surface, at the apple's 2D centre pixel,
placed in 3D. Depth is used for exactly one thing — to give that one
visible point its distance from the camera. A depth camera measures the
distance to the surface it can see, so this point is exactly what the
sensor observes.

## What was built

1. **`backproject(u, v, z, intrinsics)`** — the base pinhole-camera
   formula, `X = (u-cx)*Z/fx`, `Y = (v-cy)*Z/fy`. Converts one pixel +
   depth into a 3D point. Tested against a known point and a
   back-project-then-reproject round trip.

2. **`robust_depth(depth_m, cx, cy, radius_px, ...)`** — rather than
   trusting the single pixel at an apple's centre (which could land on a
   gap between leaves, sensor noise, or the image edge), this samples a
   small disc around the centre (half the 2D circle's pixel radius), drops
   invalid (zero) pixels and statistical outliers (more than ~3.5
   median-absolute-deviations from the patch median), and returns the
   median of what's left. If there's nothing valid left, or too few pixels
   to trust, it returns `None` with a `reason` string — **never a silent
   zero**.

3. **`apple_front_point_3d(u, v, radius_px, depth_m, intrinsics)`** — ties
   the two together: robust depth at the 2D centre, then back-project.
   The result is the front-centre point itself. `radius_px` (a 2D pixel
   quantity) is only used to size `robust_depth`'s sampling disc.

4. **Tests** (`tests/test_geometry.py`, 8 tests) include a **synthetic
   depth image test**: a disc of pixels at a known depth (0.6 m), run
   through `apple_front_point_3d`, and the result compared against the
   exact back-projection of that known centre pixel and depth.
   - **No sensor noise: error under 1 mm.**
   - **With 2 mm of simulated depth noise: error under 2 mm** — checking
     that the median-over-a-patch approach actually suppresses noise.

## The worked example

`docs/stage-4-worked-example.md` walks through one real labelled apple by
hand: a robust depth of 0.155 m (median of 162 clustered pixel values) and
the back-projected front-centre point.
`tests/test_geometry.py::test_worked_example_matches_hand_calculation`
asserts the code reproduces those exact numbers.

## Result

8/8 geometry tests pass (49/49 across the whole project). The synthetic
depth image test shows the formula is implemented correctly; the worked
example shows what it actually produces on this project's real
(depth-scale-limited) data.

## What could go wrong later

- Every number coming out of this module inherits the depth-scale and
  intrinsics approximations from `configs/camera.yaml` (checkpoint C1), so
  the absolute position of any reported apple should not be taken at face
  value — restate this in the final report.
- If the 2D centre lands slightly off the apple (a poor detection), the
  sampling disc may pick up leaves or background; the outlier rejection
  and the minimum-pixel check reduce, but cannot fully remove, this risk.

## Terms used

- **Back-projection:** going from a 2D pixel + depth to a 3D point (the
  inverse of how a camera projects a 3D point onto its 2D sensor).
- **Front-centre point:** the point on the apple's camera-facing surface
  at its 2D centre pixel — what the depth sensor sees at that pixel.
- **Median absolute deviation (MAD):** a measure of spread, like standard
  deviation, but based on the median instead of the mean — much less
  thrown off by a few extreme outlier values.
