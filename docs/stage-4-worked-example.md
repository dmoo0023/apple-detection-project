# Stage 4 worked example: one apple, by hand

This walks through `src/geometry.py`'s calculation for one real labelled
apple, with real numbers, so it can be checked by hand (calculator, not
code) and compared against what the code actually outputs.
`tests/test_geometry.py::test_worked_example_matches_hand_calculation`
checks these exact numbers against `surface_point_3d()`'s output.

**Scope note (checkpoint C2, 2026-10-24):** this computes the apple's
**visible near surface**, not a corrected centre. An earlier version of
this module pushed the surface point further away from the camera by one
apple radius, to estimate the true centre; the student's decision at C2
was to drop that correction and report the surface point directly, since
adding a geometric correction on top of an already-approximate depth scale
(checkpoint C1) added complexity without adding trustworthy precision. The
consequence, stated plainly: every reported 3D position in this project is
offset from the apple's true centre by roughly one apple radius, always
toward the camera, always by the same mechanism. Keep that in mind reading
any 3D number in this project, including the one computed below.

## The apple

Image `R01U01_Summer_RGB-12-40-11` / `R01U01_Summer_D-12-40-11`, the first
labelled apple (index 0) in `data/derived/apples_2d.csv`.

**Inputs:**

| Quantity | Value | Units | Source |
|---|---|---|---|
| u (pixel column) | 352.235 | px | labelme circle centre |
| v (pixel row) | 162.607 | px | labelme circle centre |
| radius_px | 14.398 | px | distance from centre to labelme's second point (used only to pick the depth-sampling disc, see Step 1) |
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
limitation accepted at checkpoint C1 — not a bug in this calculation.

## Step 2: back-project the surface point

```
X = (u - cx) * Z / fx = (352.235 - 320.0) * 0.155 / 462.14  =  0.010811 m
Y = (v - cy) * Z / fy = (162.607 - 240.0) * 0.155 / 617.16  = -0.019437 m
Z =                                                            0.155000 m
```

**Surface point = (0.010811, -0.019437, 0.155000) m.** This is this
project's reported 3D position for this apple. Per the scope note above,
it is where the depth sensor's view of the apple's near surface sits in
3D — NOT the apple's centre. If this apple's real radius is, say, 3.5 cm,
its true centre would be roughly a further 3.5 cm away from the camera
along the same line of sight (further +Z, with a small +X/-Y shift from
the ray geometry) — a gap this project's reported numbers do not correct
for, by deliberate choice.

## Text sketch

```
camera (0,0,0)
   \
    \   ray through pixel (352.2, 162.6)
     \
      \
       \----X                    X = reported 3D position
            |<-- ~1 radius -->O     O = true apple centre (not computed
            (not computed here)      by this project, see scope note)
```

## Check

`src/geometry.py` reproduces every number above exactly (`robust_depth`
and `backproject`, combined in `surface_point_3d`) — see
`tests/test_geometry.py::test_worked_example_matches_hand_calculation`,
which asserts the code's output against these exact values.
