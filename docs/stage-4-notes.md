# Stage 4: Shared 3D module

## What this stage does

Every method in this project — colour masking (Stage 6), YOLO (Stage 8),
any hybrid (Stage 9) — ends up at the same question: "given a detected
apple (a pixel centre and radius) and a depth image, where is that apple
in 3D?" `src/geometry.py` answers it once, so every later stage calls the
same functions, and results across methods are directly comparable.

## Scope decision at checkpoint C2 (2026-10-24)

A depth camera measures the distance to whatever surface it can see — for
an apple, that's the near side facing the camera, not the middle of the
apple. The first version of this module corrected for that: back-project
the surface point, then push it further from the camera by one apple
radius, along the same viewing ray, to estimate the true centre. It was
implemented, and validated against a synthetic sphere to sub-millimetre
accuracy (see git history for that version if it's ever wanted again).

At checkpoint C2, the student asked to remove that correction: given the
depth scale itself is already only an accepted approximation (checkpoint
C1), adding a geometric correction on top of it felt like more complexity
than the data could actually support. **This is a reasonable call, with a
clear, known consequence, stated here plainly:** the correction removes a
*systematic bias*, not sensor noise — every apple's near-surface point is
offset from its true centre by roughly its own radius (for a real apple,
~3–4.5 cm), always in the same direction (away from the camera), whether
or not the depth scale is calibrated correctly. Dropping the correction
means every reported 3D position in this project carries that same
~3–4.5 cm bias versus the true centre — bigger than the brief's own 2–3 cm
target band. Because the bias is systematic and applied equally to both
predictions and labels (Stage 5's evaluation runs both through the same
`surface_point_3d`), it does **not** distort which *method* looks better
relative to another — but it does mean these 3D numbers cannot be read as
literal centre-accuracy against the brief's target, and the final report
needs to say so explicitly, not just once but wherever a 3D number
appears.

## What was built

1. **`backproject(u, v, z, intrinsics)`** — the base pinhole-camera
   formula, `X = (u-cx)*Z/fx`, `Y = (v-cy)*Z/fy`. Converts one pixel +
   depth into a 3D point. Tested against a known point and a
   back-project-then-reproject round trip.

2. **`robust_depth(depth_m, cx, cy, radius_px, ...)`** — rather than
   trusting the single pixel at an apple's centre (which could land on a
   gap between leaves, sensor noise, or the image edge), this samples a
   disc covering the *inner half* of the apple's radius, drops invalid
   (zero) pixels and statistical outliers (more than ~3.5 median-absolute-
   deviations from the patch median), and returns the median of what's
   left. If there's nothing valid left, or too few pixels to trust, it
   returns `None` with a `reason` string — **never a silent zero**.

3. **`surface_point_3d(u, v, radius_px, depth_m, intrinsics)`** — ties
   the two together: robust depth, then back-project. That's it — no
   centre correction (see scope decision above). `radius_px` is still
   accepted as a parameter because `robust_depth` needs it to size the
   sampling disc, even though no size/offset math happens with it anymore.

4. **Tests** (`tests/test_geometry.py`, 8 tests) include a **synthetic
   sphere test**: render a depth image of a sphere at a known 3D position
   by solving the exact ray-sphere intersection per pixel, run it through
   `surface_point_3d`, and check the recovered point against the sphere's
   true *near-surface* point (not its centre — consistent with the scope
   decision).
   - **No sensor noise: error under 3mm** (not quite sub-millimetre, even
     with a perfect depth render — `robust_depth`'s median is taken over a
     disc covering half the apple's radius, and a curved sphere's depth
     genuinely varies across that disc, so the median legitimately differs
     a little from the single exact centre pixel; this is a real, expected
     effect of the sampling method, not a formula bug).
   - **With 2mm of simulated depth noise: error under 5mm** — checking
     that the median-over-a-patch approach actually suppresses noise.
   - A further assertion confirms the recovered point is roughly one
     radius short of the sphere's true centre, by design.

## The worked example

`docs/stage-4-worked-example.md` walks through one real labelled apple by
hand: a robust depth of 0.155 m (median of 162 clustered pixel values) and
a back-projected 3D position — explicitly labelled as the apple's near
surface, with a text sketch showing where the (uncomputed) true centre
would sit relative to it.
`tests/test_geometry.py::test_worked_example_matches_hand_calculation`
asserts the code reproduces those exact numbers.

## Result

8/8 geometry tests pass (33/33 across the whole project). The synthetic
sphere test is the strongest evidence the surface-point formula is
implemented correctly; the worked example shows what it actually produces
on this project's real (depth-scale-limited) data.

## What could go wrong later

- Every number coming out of this module inherits the depth-scale and
  intrinsics approximations from `configs/camera.yaml` (checkpoint C1)
  *and* the surface-vs-centre bias from this checkpoint's scope decision.
  Two separate, stacked reasons the absolute position of any reported
  apple should not be taken at face value — both need restating in the
  final report, not just the first one found.
- If this project is ever extended with better depth data (e.g. real
  calibration, 16-bit depth), the surface-vs-centre bias becomes the
  dominant remaining error source, and bringing back a centre correction
  (git history has a working, tested version) would then be worth
  revisiting.

## Terms used

- **Back-projection:** going from a 2D pixel + depth to a 3D point (the
  inverse of how a camera projects a 3D point onto its 2D sensor).
- **Systematic bias vs. noise:** noise is random, different each
  measurement, and averages out; a systematic bias is a consistent offset
  in the same direction every time — averaging more measurements does not
  reduce it, only correcting the underlying cause does.
- **Median absolute deviation (MAD):** a measure of spread, like standard
  deviation, but based on the median instead of the mean — much less
  thrown off by a few extreme outlier values.
