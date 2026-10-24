"""Stage 6 tests: colour segmentation, circle fitting, and watershed
splitting of touching apples, on small synthetic images."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.classical import (  # noqa: E402
    ClassicalConfig,
    clean_mask,
    detect_apples,
    fit_circle_to_contour,
    segment_red_mask,
)

APPLE_RGB = (200, 30, 30)   # a saturated red, Lab 'a' channel and HSV hue both clearly "red"
LEAF_RGB = (40, 130, 40)    # green, should fail both red tests
BG_RGB = (90, 60, 30)       # dull brown background/soil


def blank_image(w=200, h=200, rgb=BG_RGB) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = rgb
    return img


def test_fit_circle_to_contour_recovers_known_circle() -> None:
    cx, cy, r = 50.3, 40.7, 15.2
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    points = np.stack([cx + r * np.cos(angles), cy + r * np.sin(angles)], axis=1)
    fit = fit_circle_to_contour(points)
    assert fit is not None
    fcx, fcy, fr = fit
    assert fcx == pytest.approx(cx, abs=0.05)
    assert fcy == pytest.approx(cy, abs=0.05)
    assert fr == pytest.approx(r, abs=0.05)


def test_fit_circle_to_contour_too_few_points_returns_none() -> None:
    assert fit_circle_to_contour(np.array([[0, 0], [1, 1]])) is None


def test_segment_red_mask_detects_red_rejects_green_and_background() -> None:
    img = blank_image()
    cv2.circle(img, (100, 100), 20, APPLE_RGB, -1)
    cv2.circle(img, (40, 40), 15, LEAF_RGB, -1)

    mask = segment_red_mask(img, ClassicalConfig())
    assert mask[100, 100] == 255  # centre of the red circle: detected
    assert mask[40, 40] == 0      # centre of the green circle: rejected
    assert mask[10, 10] == 0      # plain background: rejected


def test_clean_mask_removes_small_speckle() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[50:70, 50:70] = 255  # a real 20x20 blob
    mask[10, 10] = 255        # a 1px speckle, should be removed by opening
    cleaned = clean_mask(mask, ClassicalConfig(morph_kernel_px=3))
    assert cleaned[60, 60] == 255  # the real blob survives
    assert cleaned[10, 10] == 0    # the speckle is gone


def test_detect_apples_single_isolated_apple() -> None:
    img = blank_image()
    cv2.circle(img, (100, 90), 18, APPLE_RGB, -1)

    detections = detect_apples(img, ClassicalConfig())
    assert len(detections) == 1
    d = detections[0]
    assert d.cx == pytest.approx(100, abs=2.0)
    assert d.cy == pytest.approx(90, abs=2.0)
    assert d.radius_px == pytest.approx(18, abs=2.5)


def test_detect_apples_separates_two_touching_apples() -> None:
    img = blank_image(w=260, h=160)
    # two apples touching (distance between centres < sum of radii)
    cv2.circle(img, (100, 80), 25, APPLE_RGB, -1)
    cv2.circle(img, (150, 80), 25, APPLE_RGB, -1)

    detections = detect_apples(img, ClassicalConfig())
    assert len(detections) == 2, f"expected 2 separated apples, got {len(detections)}"
    centres = sorted((d.cx, d.cy) for d in detections)
    assert centres[0][0] == pytest.approx(100, abs=6.0)
    assert centres[1][0] == pytest.approx(150, abs=6.0)


SHADOW_APPLE_RGB = (60, 45, 42)      # dark, desaturated red -- an apple in shade
HIGHLIGHT_APPLE_RGB = (230, 120, 60)  # bright, orange-shifted -- an apple in direct sun glare


def test_primary_variant_alone_misses_shadow_and_highlight_apples() -> None:
    img = blank_image()
    cv2.circle(img, (100, 100), 15, SHADOW_APPLE_RGB, -1)
    cv2.circle(img, (60, 60), 15, HIGHLIGHT_APPLE_RGB, -1)
    primary_only = ClassicalConfig(use_shadow_variant=False, use_highlight_variant=False)
    mask = segment_red_mask(img, primary_only)
    assert mask[100, 100] == 0
    assert mask[60, 60] == 0


def test_shadow_variant_catches_dark_desaturated_red() -> None:
    img = blank_image()
    cv2.circle(img, (100, 100), 15, SHADOW_APPLE_RGB, -1)
    mask = segment_red_mask(img, ClassicalConfig(use_shadow_variant=True, use_highlight_variant=False))
    assert mask[100, 100] == 255


def test_highlight_variant_catches_bright_orange_shifted_red() -> None:
    img = blank_image()
    cv2.circle(img, (100, 100), 15, HIGHLIGHT_APPLE_RGB, -1)
    mask = segment_red_mask(img, ClassicalConfig(use_shadow_variant=False, use_highlight_variant=True))
    assert mask[100, 100] == 255


def test_shadow_and_highlight_variants_still_reject_background() -> None:
    img = blank_image()
    cv2.circle(img, (100, 100), 15, LEAF_RGB, -1)
    config = ClassicalConfig(use_shadow_variant=True, use_highlight_variant=True)
    mask = segment_red_mask(img, config)
    assert mask[100, 100] == 0
    assert mask[10, 10] == 0  # plain brown/soil background


def test_detect_apples_finds_shaded_apple_end_to_end() -> None:
    img = blank_image()
    cv2.circle(img, (100, 90), 18, SHADOW_APPLE_RGB, -1)
    detections = detect_apples(img, ClassicalConfig(use_shadow_variant=True))
    assert len(detections) == 1
    assert detections[0].cx == pytest.approx(100, abs=2.0)
    assert detections[0].cy == pytest.approx(90, abs=2.0)


def test_detect_apples_ignores_pure_background() -> None:
    img = blank_image()
    cv2.circle(img, (60, 60), 20, LEAF_RGB, -1)
    detections = detect_apples(img, ClassicalConfig())
    assert detections == []
