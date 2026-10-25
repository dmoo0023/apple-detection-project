# Stage 4 worked example: one apple, by hand

This walks through `src/geometry.py`'s calculation for one real labelled
apple, with real numbers, so it can be checked by hand (calculator, not
code) and compared against what the code actually outputs.
`tests/test_geometry.py::test_worked_example_matches_hand_calculation`
checks these exact numbers against `apple_front_point_3d()`'s output.

The result is the apple's **front-centre point**: the point on its
camera-facing surface at its 2D centre pixel, placed in 3D.

## The apple

Image `R01U01_Summer_RGB-12-40-11` / `R01U01_Summer_D-12-40-11`, the first
labelled apple (index 0) in `data/derived/apples_2d.csv`.

**Inputs:**

| Quantity | Value | Units | Source |
|---|---|---|---|
| u (pixel column) | 352.235 | px | labelme circle centre |
| v (pixel row) | 162.607 | px | labelme circle centre |
| radius_px | 14.398 | px | labelme circle (used only to size the depth-sampling disc, see Step 1) |
| fx | 462.14 | px | `configs/camera.yaml` (approximate, see checkpoint C1) |
| fy | 617.16 | px | `configs/camera.yaml` (approximate) |
| cx | 320.0 | px | `configs/camera.yaml` (assumed image centre) |
| cy | 240.0 | px | `configs/camera.yaml` (assumed image centre) |
| depth_scale_to_metres | 0.001 | m/unit | `configs/camera.yaml` (accepted approximation) |

## Step 1: sample a robust depth at the apple's centre

`robust_depth()` looks at a small disc around the 2D centre — radius
`0.5 x 14.398 = 7.199 px` around (352.235, 162.607) — and takes the median
of the valid (non-zero) raw pixel values in that disc, after dropping
outliers.

- 162 valid raw pixel values found in the disc, all either **154 or
  155** (raw 0–255 units) — a very tight cluster, nothing dropped as an
  outlier.
- **Median raw value = 155.**
- Converted to metres: `Z = 155 x 0.001 = 0.155 m`.

This number being ~0.155 m, far under the dataset's stated 0.4–1.2 m
capture range, is the direct, concrete consequence of the depth-scale
limitation accepted at checkpoint C1 — not a bug in this calculation.

## Step 2: back-project the front-centre point

```
X = (u - cx) * Z / fx = (352.235 - 320.0) * 0.155 / 462.14  =  0.010811 m
Y = (v - cy) * Z / fy = (162.607 - 240.0) * 0.155 / 617.16  = -0.019437 m
Z =                                                            0.155000 m
```

**Front-centre point = (0.010811, -0.019437, 0.155000) m.** This is the
project's reported 3D position for this apple: about 1.1 cm right of the
camera's optical axis, 1.9 cm above it (Y is negative, and Y points
down), and 15.5 cm in front of the camera.

## Text sketch

```
camera (0,0,0)
   \
    \   ray through pixel (352.2, 162.6)
     \
      \
       \----X        X = front-centre point (0.0108, -0.0194, 0.155) m,
            (apple)      on the apple's camera-facing surface
```

## Check

`src/geometry.py` reproduces every number above exactly (`robust_depth`
and `backproject`, combined in `apple_front_point_3d`) — see
`tests/test_geometry.py::test_worked_example_matches_hand_calculation`,
which asserts the code's output against these exact values.
