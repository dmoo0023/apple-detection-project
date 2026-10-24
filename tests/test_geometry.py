"""Stage 4 tests: backprojection, robust depth sampling, and apple centre
reconstruction, including a synthetic-sphere accuracy test (with and
without depth noise) as required by the workflow's Stage 4 exit check.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.geometry import (  # noqa: E402
    Intrinsics,
    apple_centre_3d,
    backproject,
    robust_depth,
)
from src.config import load_paths, load_camera  # noqa: E402

INTR = Intrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0)


def test_backproject_known_point() -> None:
    # a point straight ahead of the camera at depth 1m should land at X=Y=0
    xyz = backproject(INTR.cx, INTR.cy, 1.0, INTR)
    assert xyz == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)


def test_backproject_reprojects_to_the_same_pixel() -> None:
    u, v, z = 450.0, 100.0, 0.8
    xyz = backproject(u, v, z, INTR)
    u_rt = INTR.fx * xyz[0] / xyz[2] + INTR.cx
    v_rt = INTR.fy * xyz[1] / xyz[2] + INTR.cy
    assert u_rt == pytest.approx(u, abs=1e-9)
    assert v_rt == pytest.approx(v, abs=1e-9)


def test_robust_depth_missing_returns_none_with_reason() -> None:
    depth = np.zeros((480, 640), dtype=float)  # all invalid (zero) pixels
    result = robust_depth(depth, cx=320, cy=240, radius_px=20)
    assert result.depth_m is None
    assert result.is_valid is False
    assert result.reason is not None


def test_robust_depth_median_ignores_outliers() -> None:
    depth = np.full((480, 640), 0.80, dtype=float)
    depth[235:245, 315:325] = 0.80  # consistent patch
    depth[238, 318] = 5.0  # one wild outlier pixel inside the sampled disc
    result = robust_depth(depth, cx=320, cy=240, radius_px=20, inner_frac=0.5)
    assert result.is_valid
    assert result.depth_m == pytest.approx(0.80, abs=1e-6)


def test_apple_centre_missing_depth_is_none_not_zero() -> None:
    depth = np.zeros((480, 640), dtype=float)
    result = apple_centre_3d(320, 240, 20, depth, INTR)
    assert result.centre_m is None
    assert result.is_valid is False
    assert result.reason is not None


# --- synthetic sphere: exact accuracy test (Stage 4 exit check) ------------

def make_sphere_depth_image(
    centre_m: np.ndarray, radius_m: float, intrinsics: Intrinsics,
    img_w: int = 640, img_h: int = 480, noise_std_m: float = 0.0, seed: int = 0,
) -> tuple[np.ndarray, float, float, float]:
    """Render a synthetic depth image of one sphere (near-surface Z only).

    Returns (depth_m, u_centre, v_centre, radius_px) -- the ground-truth 2D
    detection a perfect detector would report for this sphere.
    """
    cx3, cy3, cz3 = centre_m
    u_centre = intrinsics.cx + intrinsics.fx * cx3 / cz3
    v_centre = intrinsics.cy + intrinsics.fy * cy3 / cz3
    radius_px = radius_m * intrinsics.fx / cz3

    depth = np.zeros((img_h, img_w), dtype=float)
    pad = int(radius_px) + 2
    v0, v1 = max(0, int(v_centre) - pad), min(img_h, int(v_centre) + pad)
    u0, u1 = max(0, int(u_centre) - pad), min(img_w, int(u_centre) + pad)

    for v in range(v0, v1):
        for u in range(u0, u1):
            kx = (u - intrinsics.cx) / intrinsics.fx
            ky = (v - intrinsics.cy) / intrinsics.fy
            # ray: P(Z) = Z * (kx, ky, 1); solve |P(Z) - centre|^2 = radius_m^2
            a = kx * kx + ky * ky + 1.0
            b = -2.0 * (kx * cx3 + ky * cy3 + cz3)
            c = cx3**2 + cy3**2 + cz3**2 - radius_m**2
            disc = b * b - 4 * a * c
            if disc < 0:
                continue  # ray misses the sphere
            z_near = (-b - np.sqrt(disc)) / (2 * a)
            if z_near > 0:
                depth[v, u] = z_near

    if noise_std_m > 0:
        rng = np.random.default_rng(seed)
        mask = depth > 0
        depth[mask] += rng.normal(0, noise_std_m, size=mask.sum())
        depth[mask] = np.clip(depth[mask], 1e-6, None)

    return depth, u_centre, v_centre, radius_px


def test_synthetic_sphere_no_noise_within_1mm() -> None:
    true_centre = np.array([0.02, -0.01, 0.60])  # 2cm right, 1cm up, 0.6m deep
    radius_m = 0.035  # 3.5cm, a realistic apple radius

    depth, u, v, r_px = make_sphere_depth_image(true_centre, radius_m, INTR, noise_std_m=0.0)
    result = apple_centre_3d(u, v, r_px, depth, INTR, method="project_and_offset")

    assert result.is_valid, result.reason
    error = np.linalg.norm(result.centre_m - true_centre)
    assert error < 0.001, f"error {error * 1000:.3f} mm exceeds 1mm (no noise)"


def test_synthetic_sphere_with_noise_within_5mm() -> None:
    true_centre = np.array([0.02, -0.01, 0.60])
    radius_m = 0.035

    depth, u, v, r_px = make_sphere_depth_image(
        true_centre, radius_m, INTR, noise_std_m=0.002, seed=1  # 2mm depth noise, typical of a real sensor
    )
    result = apple_centre_3d(u, v, r_px, depth, INTR, method="project_and_offset")

    assert result.is_valid, result.reason
    error = np.linalg.norm(result.centre_m - true_centre)
    assert error < 0.005, f"error {error * 1000:.3f} mm exceeds 5mm (2mm depth noise)"


def test_synthetic_sphere_sphere_fit_method() -> None:
    true_centre = np.array([0.0, 0.0, 0.70])
    radius_m = 0.04

    depth, u, v, r_px = make_sphere_depth_image(true_centre, radius_m, INTR, noise_std_m=0.0)
    result = apple_centre_3d(u, v, r_px, depth, INTR, method="sphere_fit")

    assert result.is_valid, result.reason
    error = np.linalg.norm(result.centre_m - true_centre)
    # sphere_fit only sees the near hemisphere from one viewpoint, so it is
    # a biased estimate even with perfect data -- looser tolerance than
    # project_and_offset is expected and documented in src/geometry.py.
    assert error < 0.02, f"error {error * 1000:.3f} mm exceeds the 20mm single-view sphere-fit tolerance"


# --- docs/stage-4-worked-example.md: code must reproduce the hand calc -----

def test_worked_example_matches_hand_calculation() -> None:
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

    surf = backproject(u, v, ds.depth_m, intr)
    assert surf == pytest.approx([0.010811, -0.019437, 0.155000], abs=1e-6)

    radius_m = r_px * ds.depth_m / intr.fx
    assert radius_m == pytest.approx(0.0048291, abs=1e-6)

    result = apple_centre_3d(u, v, r_px, depth_m, intr, method="project_and_offset")
    assert result.is_valid
    assert result.centre_m == pytest.approx([0.011145, -0.020037, 0.159780], abs=1e-5)
