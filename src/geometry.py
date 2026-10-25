"""The one shared module for turning a detected apple (2D pixel centre) into
a 3D position. Every method in this project (classical colour masking,
YOLO, any hybrid) must go through these functions, so results are directly
comparable (rule 7: one evaluation code path per concern).

What this module outputs: the 3D FRONT-CENTRE POINT of each apple -- the
point on the apple's front (camera-facing) surface at its 2D centre pixel.
Depth is used for exactly one thing: to place that one visible point in
3D (robust depth sample at the centre, then back-project).

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
        """Build Intrinsics from a dict with fx, fy, cx, cy keys (e.g. configs/camera.yaml)."""
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
      radius_px: 2D circle radius in pixels (from the detector or label).
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
class FrontPointResult:
    """Result of apple_front_point_3d: the apple's 3D front-centre point, or a
    reason it couldn't be computed."""
    point_m: np.ndarray | None  # [X, Y, Z], metres, camera frame
    depth_sample: DepthSample
    is_valid: bool
    reason: str | None = None


def apple_front_point_3d(
    u: float,
    v: float,
    radius_px: float,
    depth_m: np.ndarray,
    intrinsics: Intrinsics,
    **robust_depth_kwargs,
) -> FrontPointResult:
    """Compute the 3D front-centre point of an apple from its 2D detection.

    Samples a robust depth Z around the 2D centre (robust_depth), then
    back-projects (u, v, Z) to camera-frame XYZ. The output is that front
    surface point itself, with nothing added to it.

    Inputs:
      u, v: 2D apple centre, in pixels.
      radius_px: 2D radius in pixels, used only to size robust_depth's
        sampling disc.
      depth_m: full depth image in METRES, shape (H, W).
      intrinsics: fx, fy, cx, cy in pixels.
    Output:
      FrontPointResult; point_m is [X, Y, Z] in metres (camera frame), or
      None (with `reason` set) whenever depth is missing or too sparse to
      trust -- never a silent zero.
    """
    depth_sample = robust_depth(depth_m, u, v, radius_px, **robust_depth_kwargs)
    if not depth_sample.is_valid or depth_sample.depth_m is None:
        return FrontPointResult(None, depth_sample, False, reason=depth_sample.reason)

    point = backproject(u, v, depth_sample.depth_m, intrinsics)
    return FrontPointResult(point, depth_sample, True, reason=None)
