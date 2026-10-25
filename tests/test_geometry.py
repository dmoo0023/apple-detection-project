"""Stage 4 tests: backprojection, robust depth sampling, and the 3D
front-centre point computed from a detection, including a synthetic depth
image accuracy test (with and without depth noise).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.geometry import (  # noqa: E402
    Intrinsics,
    backproject,
    robust_depth,
    apple_front_point_3d,
)
from src.config import load_paths, load_camera  # noqa: E402

INTR = Intrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0)


def test_backproject_known_point() -> None:
    """A pixel at the principal point back-projects onto the optical axis (X = Y = 0)."""
    # a point straight ahead of the camera at depth 1m should land at X=Y=0
    xyz = backproject(INTR.cx, INTR.cy, 1.0, INTR)
    assert xyz == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)


def test_backproject_reprojects_to_the_same_pixel() -> None:
    """Back-projecting then projecting again returns the original pixel."""
    u, v, z = 450.0, 100.0, 0.8
    xyz = backproject(u, v, z, INTR)
    u_rt = INTR.fx * xyz[0] / xyz[2] + INTR.cx
    v_rt = INTR.fy * xyz[1] / xyz[2] + INTR.cy
    assert u_rt == pytest.approx(u, abs=1e-9)
    assert v_rt == pytest.approx(v, abs=1e-9)


def test_robust_depth_missing_returns_none_with_reason() -> None:
    """An all-zero depth patch gives None plus a reason, never a zero depth."""
    depth = np.zeros((480, 640), dtype=float)  # all invalid (zero) pixels
    result = robust_depth(depth, cx=320, cy=240, radius_px=20)
    assert result.depth_m is None
    assert result.is_valid is False
    assert result.reason is not None


def test_robust_depth_median_ignores_outliers() -> None:
    """One wild outlier pixel does not move the median depth."""
    depth = np.full((480, 640), 0.80, dtype=float)
    depth[235:245, 315:325] = 0.80  # consistent patch
    depth[238, 318] = 5.0  # one wild outlier pixel inside the sampled disc
    result = robust_depth(depth, cx=320, cy=240, radius_px=20, inner_frac=0.5)
    assert result.is_valid
    assert result.depth_m == pytest.approx(0.80, abs=1e-6)


def test_front_point_missing_depth_is_none_not_zero() -> None:
    """Missing depth gives no 3D point plus a reason, never (0, 0, 0)."""
    depth = np.zeros((480, 640), dtype=float)
    result = apple_front_point_3d(320, 240, 20, depth, INTR)
    assert result.point_m is None
    assert result.is_valid is False
    assert result.reason is not None


# --- synthetic depth image: accuracy of the recovered front-centre point ------

def make_disc_depth_image(
    u_centre: float, v_centre: float, radius_px: float, front_z_m: float,
    img_w: int = 640, img_h: int = 480, noise_std_m: float = 0.0, seed: int = 0,
) -> np.ndarray:
    """Synthetic depth image: one apple-sized disc of pixels at a known depth
    `front_z_m` (metres), background left as 0 (= no depth).

    The known front-centre point is then backproject(u_centre, v_centre, front_z_m).
    """
    vv, uu = np.mgrid[0:img_h, 0:img_w]
    mask = np.hypot(uu - u_centre, vv - v_centre) <= radius_px
    depth = np.zeros((img_h, img_w), dtype=float)
    depth[mask] = front_z_m

    if noise_std_m > 0:
        rng = np.random.default_rng(seed)
        depth[mask] += rng.normal(0, noise_std_m, size=mask.sum())
        depth[mask] = np.clip(depth[mask], 1e-6, None)
    return depth


def test_synthetic_front_point_no_noise_within_1mm() -> None:
    """On a noise-free synthetic depth image the front-centre point is within 1 mm."""
    u, v, r_px, z = 340.0, 230.0, 35.0, 0.60  # off-centre apple, front at 0.6m
    depth = make_disc_depth_image(u, v, r_px, z)
    result = apple_front_point_3d(u, v, r_px, depth, INTR)

    assert result.is_valid, result.reason
    expected = backproject(u, v, z, INTR)
    error = np.linalg.norm(result.point_m - expected)
    assert error < 0.001, f"error {error * 1000:.3f} mm exceeds 1mm (no sensor noise)"


def test_synthetic_front_point_with_noise_within_2mm() -> None:
    """With 2 mm depth noise the front-centre point is still within 2 mm."""
    u, v, r_px, z = 340.0, 230.0, 35.0, 0.60
    depth = make_disc_depth_image(u, v, r_px, z, noise_std_m=0.002, seed=1)  # 2mm depth noise, typical of a real sensor
    result = apple_front_point_3d(u, v, r_px, depth, INTR)

    assert result.is_valid, result.reason
    expected = backproject(u, v, z, INTR)
    error = np.linalg.norm(result.point_m - expected)
    # the median over ~960 noisy pixels averages most of the 2mm noise away
    assert error < 0.002, f"error {error * 1000:.3f} mm exceeds 2mm (2mm depth noise)"


# --- docs/stage-4-worked-example.md: code must reproduce the hand calc -----

def test_worked_example_matches_hand_calculation() -> None:
    """The code reproduces docs/stage-4-worked-example.md's hand calculation exactly."""
    paths = load_paths()
    cam = load_camera()
    intr = Intrinsics.from_dict(cam)
    scale = cam["depth_scale_to_metres"]

    depth_path = paths["depth_dir"] / "R01U01_Summer_D-12-40-11.png"
    depth_raw = np.array(Image.open(depth_path).convert("L")).astype(float)
    depth_m = depth_raw * scale

    u, v, r_px = 352.234957, 162.607450, 14.398103

    ds = robust_depth(depth_m, u, v, r_px)
    assert ds.is_valid
    assert ds.valid_pixel_count == 162
    assert ds.depth_m == pytest.approx(0.155, abs=1e-9)

    result = apple_front_point_3d(u, v, r_px, depth_m, intr)
    assert result.is_valid
    assert result.point_m == pytest.approx([0.010811, -0.019437, 0.155000], abs=1e-6)
