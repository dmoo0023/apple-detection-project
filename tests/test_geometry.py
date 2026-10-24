"""Stage 4 tests: backprojection, robust depth sampling, and the 3D surface
point computed from a detection, including a synthetic-sphere accuracy
test (with and without depth noise).

SCOPE NOTE: as of checkpoint C2 (2026-10-24), this module reports the
apple's visible near surface, not a corrected centre -- see
src/geometry.py's module docstring. The synthetic sphere test below checks
the recovered point against the sphere's true NEAR-SURFACE intersection
point (what a perfect depth sensor would actually measure), not the
sphere's centre.
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
    surface_point_3d,
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


def test_surface_point_missing_depth_is_none_not_zero() -> None:
    depth = np.zeros((480, 640), dtype=float)
    result = surface_point_3d(320, 240, 20, depth, INTR)
    assert result.point_m is None
    assert result.is_valid is False
    assert result.reason is not None


# --- synthetic sphere: accuracy of the recovered SURFACE point -------------

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


def true_near_surface_point(centre_m: np.ndarray, radius_m: float, intrinsics: Intrinsics) -> np.ndarray:
    """The exact point on the sphere closest to the camera, along the ray through its
    projected centre -- what a noise-free depth sensor would measure for this apple."""
    cz3 = centre_m[2]
    u = intrinsics.cx + intrinsics.fx * centre_m[0] / cz3
    v = intrinsics.cy + intrinsics.fy * centre_m[1] / cz3
    direction = np.array([(u - intrinsics.cx) / intrinsics.fx, (v - intrinsics.cy) / intrinsics.fy, 1.0])
    near_z = cz3 * (np.linalg.norm(centre_m) - radius_m) / np.linalg.norm(centre_m)
    return direction * near_z


def test_synthetic_sphere_surface_point_no_noise_within_3mm() -> None:
    true_centre = np.array([0.02, -0.01, 0.60])  # 2cm right, 1cm up, 0.6m deep
    radius_m = 0.035  # 3.5cm, a realistic apple radius

    depth, u, v, r_px = make_sphere_depth_image(true_centre, radius_m, INTR, noise_std_m=0.0)
    result = surface_point_3d(u, v, r_px, depth, INTR)

    assert result.is_valid, result.reason
    expected_surface = true_near_surface_point(true_centre, radius_m, INTR)
    error = np.linalg.norm(result.point_m - expected_surface)
    # Not sub-millimetre even with zero sensor noise: robust_depth's median is taken
    # over a disc covering half the apple's radius, and a curved sphere's depth varies
    # across that disc, so the median legitimately differs a little from the exact
    # single centre pixel. This tolerance reflects that real, expected effect.
    assert error < 0.003, f"error {error * 1000:.3f} mm exceeds 3mm (no sensor noise)"

    # and it should NOT match the sphere's true centre -- that's the whole point of the
    # checkpoint-C2 scope decision: this is a surface point, roughly one radius short of
    # the true centre (a little less than the full radius, since the disc-median again
    # pulls the estimate slightly past the exact apex, toward the centre).
    centre_error = np.linalg.norm(result.point_m - true_centre)
    assert centre_error == pytest.approx(radius_m, abs=0.003)


def test_synthetic_sphere_surface_point_with_noise_within_5mm() -> None:
    true_centre = np.array([0.02, -0.01, 0.60])
    radius_m = 0.035

    depth, u, v, r_px = make_sphere_depth_image(
        true_centre, radius_m, INTR, noise_std_m=0.002, seed=1  # 2mm depth noise, typical of a real sensor
    )
    result = surface_point_3d(u, v, r_px, depth, INTR)

    assert result.is_valid, result.reason
    expected_surface = true_near_surface_point(true_centre, radius_m, INTR)
    error = np.linalg.norm(result.point_m - expected_surface)
    assert error < 0.005, f"error {error * 1000:.3f} mm exceeds 5mm (2mm depth noise)"


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

    result = surface_point_3d(u, v, r_px, depth_m, intr)
    assert result.is_valid
    assert result.point_m == pytest.approx([0.010811, -0.019437, 0.155000], abs=1e-6)
