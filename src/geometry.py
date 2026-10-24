"""The one shared module for turning a detected apple (pixel centre + radius)
into a 3D position. Every method in this project (classical colour masking,
YOLO, any hybrid) must go through these functions, so results are directly
comparable (rule 7: one evaluation code path per concern).

Coordinate convention (camera frame, metres):
  X: right  (increases with pixel column u)
  Y: down   (increases with pixel row v)
  Z: forward, away from the camera, along the optical axis
This matches the usual pinhole-camera / OpenCV / RealSense convention: the
camera sits at the origin looking down +Z, with no coordinate flips.

Pixel convention: u is the column (x, 0 = left edge), v is the row
(y, 0 = top edge), matching image-array indexing [row, col] = [v, u].

Units: pixel coordinates and radii are in pixels unless stated "_m" for
metres. Depth arrays passed into this module must already be in METRES
(converting from whatever raw format a depth file uses, e.g. the 8-bit
raw-as-mm approximation documented in configs/camera.yaml, is the caller's
job -- see src/depth.py once Stage 6 exists, or scripts/*).
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Intrinsics:
    """Pinhole camera intrinsics, in pixels. See configs/camera.yaml for this
    project's values and how approximate they are (checkpoint C1)."""
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_dict(cls, d: dict) -> "Intrinsics":
        return cls(fx=d["fx"], fy=d["fy"], cx=d["cx"], cy=d["cy"])


def backproject(u: float, v: float, z: float, intrinsics: Intrinsics) -> np.ndarray:
    """Convert one pixel + depth into a camera-frame 3D point.

    Inputs:
      u, v: pixel coordinates (column, row), in pixels.
      z: depth at that pixel, in METRES (the distance along the optical
         axis from the camera to the point -- not the straight-line
         distance from the camera to the point).
      intrinsics: fx, fy, cx, cy in pixels.
    Output:
      numpy array [X, Y, Z] in metres, camera frame (see module docstring
      for the axis convention).

    Derivation: a pinhole camera maps a 3D point (X, Y, Z) to pixel
    (u, v) = (fx * X/Z + cx, fy * Y/Z + cy). Solving for X, Y given a known
    Z (from the depth sensor) inverts that:
      X = (u - cx) * Z / fx
      Y = (v - cy) * Z / fy
    """
    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy
    return np.array([x, y, z], dtype=float)


@dataclass
class DepthSample:
    """Result of robust_depth: either a usable depth value, or a reason it's missing."""
    depth_m: float | None
    valid_pixel_count: int
    is_valid: bool
    reason: str | None = None


def robust_depth(
    depth_m: np.ndarray,
    cx: float,
    cy: float,
    radius_px: float,
    inner_frac: float = 0.5,
    min_valid_pixels: int = 5,
    outlier_mad_k: float = 3.5,
) -> DepthSample:
    """Estimate one reliable depth value for an apple from a patch of pixels.

    Rather than trusting the single pixel at the labelled/detected centre
    (easy to land on a gap between leaves, a noisy pixel, or the image
    border), this samples a small disc in the MIDDLE of the apple -- radius
    `inner_frac * radius_px` around (cx, cy) -- where the surface is most
    likely to be the apple itself and not background or occluding leaves,
    then takes the median after dropping zeros (the sensor's "no depth"
    sentinel) and statistical outliers.

    Inputs:
      depth_m: full depth image in METRES, shape (H, W). 0 or NaN = invalid pixel.
      cx, cy: apple centre in pixels.
      radius_px: apple radius in pixels (from the detector or label).
      inner_frac: how much of the radius to sample from (0.5 = inner half).
      min_valid_pixels: below this many valid pixels, treat the estimate as unreliable.
      outlier_mad_k: pixels more than this many median-absolute-deviations
        from the patch median are dropped as outliers (e.g. a stray
        background pixel that leaked into the inner disc).
    Output:
      DepthSample(depth_m, valid_pixel_count, is_valid, reason). depth_m is
      None and reason is set whenever is_valid is False -- callers must
      never silently treat a missing depth as zero (rule 4, Stage 4 item 4).
    """
    h, w = depth_m.shape
    inner_radius = max(1.0, inner_frac * radius_px)

    v0, v1 = max(0, int(cy - inner_radius)), min(h, int(cy + inner_radius) + 1)
    u0, u1 = max(0, int(cx - inner_radius)), min(w, int(cx + inner_radius) + 1)
    if v1 <= v0 or u1 <= u0:
        return DepthSample(None, 0, False, reason="sample window falls entirely outside the image")

    vv, uu = np.mgrid[v0:v1, u0:u1]
    dist = np.hypot(uu - cx, vv - cy)
    disc_mask = dist <= inner_radius
    patch = depth_m[v0:v1, u0:u1]

    valid_mask = disc_mask & np.isfinite(patch) & (patch > 0)
    values = patch[valid_mask]

    if values.size == 0:
        return DepthSample(None, 0, False, reason="no valid (non-zero, finite) depth pixels in the inner disc")

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 0:
        keep = np.abs(values - median) <= outlier_mad_k * mad * 1.4826  # 1.4826 scales MAD to a std-dev estimate
        values = values[keep]

    if values.size == 0:
        return DepthSample(None, 0, False, reason="all pixels rejected as outliers")

    if values.size < min_valid_pixels:
        return DepthSample(float(np.median(values)), int(values.size), False,
                            reason=f"only {values.size} valid pixels, below min_valid_pixels={min_valid_pixels}")

    return DepthSample(float(np.median(values)), int(values.size), True, reason=None)


@dataclass
class AppleCentreResult:
    """Result of apple_centre_3d: either a 3D centre, or a reason it couldn't be computed."""
    centre_m: np.ndarray | None       # [X, Y, Z] of the apple's CENTRE (not surface), metres
    surface_point_m: np.ndarray | None  # [X, Y, Z] of the depth-sampled SURFACE point, metres
    radius_m: float | None
    depth_sample: DepthSample
    is_valid: bool
    reason: str | None = None


def apple_centre_3d(
    u: float,
    v: float,
    radius_px: float,
    depth_m: np.ndarray,
    intrinsics: Intrinsics,
    method: str = "project_and_offset",
    **robust_depth_kwargs,
) -> AppleCentreResult:
    """Compute an apple's 3D CENTRE from its 2D detection (pixel centre + radius).

    A depth sensor only sees the near surface of the apple facing the
    camera, not its centre. Two ways to correct for that, selected by `method`:

    "project_and_offset" (default):
      1. Sample a robust depth Z at the apple's pixel centre (robust_depth).
      2. Back-project (u, v, Z) to the 3D SURFACE point.
      3. Convert the apple's pixel radius to a metric radius at that depth:
         radius_m = radius_px * Z / fx (the same small-angle pinhole
         relationship used to go from 3D size to pixel size, inverted).
      4. Move the surface point further from the camera, along the same
         viewing ray, by radius_m -- i.e. push it from the near surface to
         the sphere's centre, assuming the apple is a sphere of that radius
         centred directly behind the surface point along that ray.

    "sphere_fit":
      Back-projects every valid pixel in the full detection radius (not
      just the inner disc) to 3D, then fits a sphere to that point cloud
      with a linear algebraic least-squares fit. This can do better when
      enough of the apple's curved surface is visible, but with only ONE
      viewpoint (per the brief) only the near hemisphere is ever seen, so
      the fit is still a biased estimate, not a true measurement -- include
      it for comparison, don't treat it as more "correct" by default.

    Returns an AppleCentreResult; centre_m is None (with `reason` set)
    whenever depth is missing or too sparse to trust -- never a silent zero.
    """
    depth_sample = robust_depth(depth_m, u, v, radius_px, **robust_depth_kwargs)
    if not depth_sample.is_valid or depth_sample.depth_m is None:
        return AppleCentreResult(None, None, None, depth_sample, False, reason=depth_sample.reason)

    z = depth_sample.depth_m
    surface_point = backproject(u, v, z, intrinsics)
    radius_m = radius_px * z / intrinsics.fx

    if method == "project_and_offset":
        ray_length = float(np.linalg.norm(surface_point))
        if ray_length == 0:
            return AppleCentreResult(None, surface_point, radius_m, depth_sample, False,
                                      reason="surface point is at the camera origin (zero depth)")
        ray_unit = surface_point / ray_length
        centre = surface_point + radius_m * ray_unit
        return AppleCentreResult(centre, surface_point, radius_m, depth_sample, True, reason=None)

    elif method == "sphere_fit":
        points = _backproject_patch(u, v, radius_px, depth_m, intrinsics)
        if points.shape[0] < 10:
            return AppleCentreResult(None, surface_point, radius_m, depth_sample, False,
                                      reason=f"only {points.shape[0]} valid 3D points for sphere fit, need >= 10")
        centre, fitted_radius = _fit_sphere(points)
        return AppleCentreResult(centre, surface_point, fitted_radius, depth_sample, True, reason=None)

    else:
        raise ValueError(f"unknown method {method!r}, expected 'project_and_offset' or 'sphere_fit'")


def _backproject_patch(u: float, v: float, radius_px: float, depth_m: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
    """Back-project every valid depth pixel within radius_px of (u, v) to 3D. Returns (N, 3) array."""
    h, w = depth_m.shape
    v0, v1 = max(0, int(v - radius_px)), min(h, int(v + radius_px) + 1)
    u0, u1 = max(0, int(u - radius_px)), min(w, int(u + radius_px) + 1)
    vv, uu = np.mgrid[v0:v1, u0:u1]
    dist = np.hypot(uu - u, vv - v)
    patch = depth_m[v0:v1, u0:u1]
    mask = (dist <= radius_px) & np.isfinite(patch) & (patch > 0)
    us, vs, zs = uu[mask], vv[mask], patch[mask]
    xs = (us - intrinsics.cx) * zs / intrinsics.fx
    ys = (vs - intrinsics.cy) * zs / intrinsics.fy
    return np.stack([xs, ys, zs], axis=1)


def _fit_sphere(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Algebraic least-squares sphere fit to an (N, 3) point cloud.

    For a sphere of centre (a, b, c) and radius r, every point (x, y, z) on
    it satisfies x^2+y^2+z^2 = 2ax + 2by + 2cz + (r^2 - a^2 - b^2 - c^2).
    This is linear in [2a, 2b, 2c, d], so it can be solved with ordinary
    least squares (not an exact fit for noisy, one-sided data, but a
    standard closed-form approximation).
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    A = np.stack([2 * x, 2 * y, 2 * z, np.ones_like(x)], axis=1)
    b = x**2 + y**2 + z**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b_, c, d = sol
    centre = np.array([a, b_, c])
    radius_sq = d + a**2 + b_**2 + c**2
    radius = math.sqrt(max(radius_sq, 0.0))
    return centre, radius
