# Stage 4 worked example: one apple, by hand

This walks through `src/geometry.py`'s calculation for one real labelled
apple, with real numbers, so it can be checked by hand (calculator, not
code) and compared against what the code actually outputs.
`tests/test_geometry.py::test_worked_example_matches_hand_calculation`
checks these exact numbers against `apple_centre_3d()`'s output.

## The apple

Image `R01U01_Summer_RGB-12-40-11` / `R01U01_Summer_D-12-40-11`, the first
labelled apple (index 0) in `data/derived/apples_2d.csv`.

**Inputs:**

| Quantity | Value | Units | Source |
|---|---|---|---|
| u (pixel column) | 352.235 | px | labelme circle centre |
| v (pixel row) | 162.607 | px | labelme circle centre |
| radius_px | 14.398 | px | distance from centre to labelme's second point |
| fx | 462.14 | px | `configs/camera.yaml` (approximate, see checkpoint C1) |
| fy | 617.16 | px | `configs/camera.yaml` (approximate) |
| cx | 320.0 | px | `configs/camera.yaml` (assumed image centre) |
| cy | 240.0 | px | `configs/camera.yaml` (assumed image centre) |
| depth_scale_to_metres | 0.001 | m/unit | `configs/camera.yaml` (accepted approximation) |

## Step 1: sample a robust depth at the apple

`robust_depth()` looks at a small disc in the middle of the apple — radius
`0.5 x 14.398 = 7.199 px` around (352.235, 162.607) — and takes the median
of the valid (non-zero) raw pixel values in that disc, after dropping
outliers.

- 162 valid raw pixel values found in the inner disc, all either **154 or
  155** (raw 0–255 units) — a very tight cluster, nothing dropped as an
  outlier.
- **Median raw value = 155.**
- Converted to metres: `Z = 155 x 0.001 = 0.155 m`.

This number being ~0.155 m, far under the dataset's stated 0.4–1.2 m
capture range, is the direct, concrete consequence of the depth-scale
limitation accepted at checkpoint C1 — not a bug in this calculation. It is
exactly the kind of number that flagged the problem in the first place.

## Step 2: back-project the surface point

```
X = (u - cx) * Z / fx = (352.235 - 320.0) * 0.155 / 462.14  =  0.010811 m
Y = (v - cy) * Z / fy = (162.607 - 240.0) * 0.155 / 617.16  = -0.019437 m
Z =                                                            0.155000 m
```

**Surface point = (0.010811, -0.019437, 0.155000) m.** This is where the
depth sensor's view of the apple's near surface sits in 3D — not the
apple's centre yet.

## Step 3: convert the pixel radius to a metric radius

```
radius_m = radius_px * Z / fx = 14.398 * 0.155 / 462.14 = 0.0048291 m
         ≈ 4.83 mm
```

**Is this a realistic apple radius? No.** A real apple is roughly
30–45 mm in radius (60–90 mm diameter). 4.83 mm is a grape, not an apple —
again a direct symptom of the depth value (0.155 m) being about 4-8x
smaller than the real capture distance would have produced. If the depth
scale were correct (e.g. true depth ~0.7 m instead of 0.155 m), the same
pixel radius would convert to `14.398 * 0.7 / 462.14 ≈ 0.0218 m` (21.8 mm)
— still on the small side for an apple but far more plausible, which
illustrates how directly the depth-scale error propagates into every
downstream number, including ones (like physical size) that have nothing
to do with depth accuracy on the surface of it.

## Step 4: push the surface point back to the centre

The viewing ray from the camera through this pixel is the direction of the
surface point itself (the ray passes through the origin and the surface
point by construction). Move along that same direction, away from the
camera, by `radius_m`:

```
ray length  = |surface_point| = sqrt(0.010811^2 + 0.019437^2 + 0.155^2)
            = 0.156588 m
ray unit    = surface_point / ray_length
            = (0.069044, -0.124130, 0.989861)

centre = surface_point + radius_m * ray_unit
       = (0.010811, -0.019437, 0.155000) + 0.0048291 * (0.069044, -0.124130, 0.989861)
       = (0.011145, -0.020037, 0.159780) m
```

**Corrected centre = (0.011145, -0.020037, 0.159780) m.**

## Before / after

| | X (m) | Y (m) | Z (m) |
|---|---|---|---|
| Surface point (before) | 0.010811 | -0.019437 | 0.155000 |
| Apple centre (after) | 0.011145 | -0.020037 | 0.159780 |
| Difference | +0.000334 | -0.000600 | +0.004780 |

The correction moves the point almost entirely in +Z (further from the
camera, deeper into the scene) — exactly what "push along the viewing ray,
away from the camera, by one radius" should do, with only a small X/Y
shift because this particular pixel isn't perfectly centred on the optical
axis.

## Text sketch

```
camera (0,0,0)
   \
    \   ray through pixel (352.2, 162.6)
     \
      \
       \----X---->O         X = surface point (what the sensor sees)
                    \        O = apple centre (what we want)
                     \         (pushed one radius further along the ray)
                      (apple, radius 4.8mm -- unrealistically
                       small, see Step 3)
```

## Check

`src/geometry.py` reproduces every number above exactly (`robust_depth`,
`backproject`, and `apple_centre_3d`'s `project_and_offset` method) — see
`tests/test_geometry.py::test_worked_example_matches_hand_calculation`,
which asserts the code's output against these exact values.
