# Stage 4: Shared 3D module

## What this stage does

Every method in this project — colour masking (Stage 6), YOLO (Stage 8),
any hybrid (Stage 9) — ends up at the same question: "given a detected
apple (a pixel centre and radius) and a depth image, where is that apple
in 3D?" If each method answered that question with its own slightly
different code, comparisons between methods would be comparing partly
different things. `src/geometry.py` answers it once, so every later stage
calls the same functions.

## The core idea: surface point vs. centre

A depth camera measures the distance to whatever surface it can see —
for an apple, that's the near side facing the camera, not the middle of
the apple. If you just back-project that surface point, every apple's
"position" would actually be a point on its skin closest to the camera,
offset from the true centre by roughly one apple-radius, always in the
direction away from the camera. The whole point of this module is
correcting for that offset.

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
   deviations from the patch median — the kind of robust threshold that
   isn't thrown off by one or two bad pixels the way a plain mean would
   be), and returns the median of what's left. If there's nothing valid
   left, or too few pixels to trust, it returns `None` with a `reason`
   string — **never a silent zero**, since a 3D point at Z=0 would
   silently and wrongly say "the apple is at the camera."

3. **`apple_centre_3d(...)`** — ties it together: get a robust depth,
   back-project to the surface point, convert the detection's pixel
   radius to a metric radius (`radius_px * Z / fx` — the same formula
   that relates a real-world size to its pixel size at a given depth, used
   in reverse), then push the surface point further away from the camera
   by exactly that radius, **along the same viewing ray** (the ray from
   the camera through the detection's pixel centre). A `method="sphere_fit"`
   alternative is also implemented: back-project every valid pixel in the
   full detection radius to 3D, then fit a sphere to that point cloud with
   a closed-form linear least-squares formula. It's included because the
   brief asks for it, but it's not the default — with only one camera
   viewpoint, only the near hemisphere of the apple is ever visible, so a
   sphere fit from that alone is still a biased estimate, not a more
   "correct" measurement.

4. **Tests** (`tests/test_geometry.py`, 9 tests) include a **synthetic
   sphere test**: render a depth image of a sphere at a known 3D position
   by solving the exact ray-sphere intersection per pixel (not an
   approximation), run it through `apple_centre_3d`, and check the
   recovered centre against the true one.
   - **No noise: error under 1mm** — this is checking the geometry formulas
     themselves are implemented correctly, since with a perfect depth
     image there's no reason for the reconstructed centre to be off by any
     more than floating-point rounding.
   - **With 2mm of simulated depth noise (Gaussian, per-pixel): error
     under 5mm** — checking that `robust_depth`'s median-over-a-patch
     actually suppresses noise rather than just passing it through.
   - A looser check (20mm) for the `sphere_fit` method, for the single-
     viewpoint reason above.

## The worked example

`docs/stage-4-worked-example.md` walks through one real labelled apple by
hand: a robust depth of 0.155 m (median of 162 clustered pixel values),
a back-projected surface point, a converted radius of **4.8 mm — not a
realistic apple size** (confirming, concretely, the depth-scale limitation
accepted at checkpoint C1: if the depth were in the dataset's real
0.4–1.2 m range instead, the same pixel radius would convert to roughly
20 mm, still small but far more plausible), and the final corrected
centre. `tests/test_geometry.py::test_worked_example_matches_hand_calculation`
asserts the code reproduces every number in that document exactly.

## Result

9/9 geometry tests pass (29/29 across the whole project). The synthetic
sphere test is the strongest evidence the formulas themselves are correct;
the worked example is the strongest evidence of what they actually produce
on this project's real (depth-scale-limited) data, and ties the abstract
accuracy target (2–3 cm) to a concrete, visible number (4.8 mm apple
radius) that makes the limitation impossible to miss later.

## What could go wrong later

- `apple_centre_3d` assumes the viewing ray through the detection's pixel
  *centre* also passes through the true sphere centre. For a perfectly
  circular, centred detection this is exact (proven by the synthetic
  sphere test); a lopsided or partially-occluded detection (robust_depth's
  job is mainly to handle depth noise, not detection-centre bias) will add
  some error this module can't correct for on its own.
- Every number coming out of this module inherits the depth-scale and
  intrinsics approximations from `configs/camera.yaml` (checkpoint C1).
  The module is numerically correct *given its inputs* — it cannot know
  those inputs are approximate.

## Terms used

- **Back-projection:** going from a 2D pixel + depth to a 3D point (the
  inverse of how a camera projects a 3D point onto its 2D sensor).
- **Median absolute deviation (MAD):** a measure of spread, like standard
  deviation, but based on the median instead of the mean — much less
  thrown off by a few extreme outlier values.
- **Algebraic sphere fit:** fitting a sphere to a set of 3D points by
  solving a linear system, rather than an iterative optimisation — fast
  and simple, at the cost of being more sensitive to one-sided (not
  all-around) point clouds than an iterative fit would be.
