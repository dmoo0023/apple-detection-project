"""The one shared module for turning a detected apple (pixel centre + radius)
into a 3D position. Every method in this project (classical colour masking,
YOLO, any hybrid) must go through these functions, so results are directly
comparable (rule 7: one evaluation code path per concern).

SCOPE NOTE (decided at checkpoint C2, 2026-10-24): this module reports the
3D position of the apple's visible NEAR SURFACE, not a corrected centre. A
depth camera only ever measures the surface facing it, offset from the true
centre by roughly one apple radius (for a real apple, ~3-4.5 cm) along the
viewing ray, always in the same direction. An earlier version of this
module corrected for that (back-project, then push the point away from the
camera by one radius) and validated the correction to <1mm on a synthetic
sphere -- see git history / docs/stage-4-notes.md for that design and why
it was dropped. The student's call: given the depth scale itself is already
only an approximate placeholder (checkpoint C1), adding a geometric
correction on top of it added complexity without adding trustworthy
precision, so the project reports the simpler, more defensible surface
point instead. Concretely, this means every reported 3D position has a
KNOWN, SYSTEMATIC bias of roughly one apple radius versus the true centre,
always in the same direction (away from the camera) -- state this
explicitly anywhere these positions are reported or compared to the
brief's "2-3cm of the true centre" target.

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
job).
"""

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
class SurfacePointResult:
    """Result of surface_point_3d: the 3D position of the apple's visible near
    surface (NOT a centre -- see the module docstring's scope note), or a
    reason it couldn't be computed."""
    point_m: np.ndarray | None  # [X, Y, Z], metres, camera frame
    depth_sample: DepthSample
    is_valid: bool
    reason: str | None = None


def surface_point_3d(
    u: float,
    v: float,
    radius_px: float,
    depth_m: np.ndarray,
    intrinsics: Intrinsics,
    **robust_depth_kwargs,
) -> SurfacePointResult:
    """Compute the 3D position of an apple's visible near surface from its 2D
    detection (pixel centre + radius).

    This is deliberately NOT a centre estimate (see the module's scope
    note): it samples a robust depth at the detection (robust_depth), then
    back-projects (u, v, Z) straight to 3D with no further correction. The
    result is biased toward the camera by roughly one apple radius versus
    the apple's true centre, consistently, for every apple.

    `radius_px` is accepted (not used to compute a size) purely so it can
    be forwarded to robust_depth, which needs it to pick the sampling disc.

    Returns a SurfacePointResult; point_m is None (with `reason` set)
    whenever depth is missing or too sparse to trust -- never a silent zero.
    """
    depth_sample = robust_depth(depth_m, u, v, radius_px, **robust_depth_kwargs)
    if not depth_sample.is_valid or depth_sample.depth_m is None:
        return SurfacePointResult(None, depth_sample, False, reason=depth_sample.reason)

    point = backproject(u, v, depth_sample.depth_m, intrinsics)
    return SurfacePointResult(point, depth_sample, True, reason=None)
