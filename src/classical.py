"""Classical (non-learned) apple detector: colour thresholding + watershed
splitting + circle fitting. The Stage 6 baseline every learned method
(Stage 8+) gets compared against.

Pipeline, per image:
  1. segment_red_mask: HSV + Lab thresholding -> binary mask of "red" pixels.
  2. clean_mask: morphological open (remove speckle) + close (fill small holes).
  3. split_touching_apples: distance transform + watershed, so two touching
     apples become two separate regions instead of one big blob.
  4. fit_circle_to_contour: least-squares circle fit per region, to turn an
     irregular blob outline into a clean (cx, cy, radius).
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ClassicalConfig:
    # --- Primary variant: well-lit red apple skin (grid-searched, Stage 6) ---
    # HSV red thresholds. Hue wraps at 0/180 in OpenCV's 0-179 scale, so red
    # needs TWO ranges: one just above 0, one just below 180.
    hue_low_max: int = 10        # red range 1: hue in [0, hue_low_max]
    hue_high_min: int = 170      # red range 2: hue in [hue_high_min, 179]
    sat_min: int = 80
    val_min: int = 40
    # Lab 'a' channel (green-red axis; higher = more red), 0-255, 128 = neutral.
    lab_a_min: int = 150

    # --- Extra colour variants, OR'd with the primary when enabled. OFF BY
    # DEFAULT: tried on real data and made every metric much worse (F1
    # 0.560->0.366, false positives 332->1134, count MAE 5.00->33.68) --
    # see docs/stage-6-notes.md. The intent was sound (real apple skin
    # ranges from a shaded, darker red to a washed-out, orange highlight
    # where direct sun overexposes it), but these fixed, hand-picked
    # thresholds turned out to match real background clutter (soil, bark,
    # a background wall) far more than synthetic test colours suggested.
    # Kept in the code, disabled by default, for anyone who wants to
    # retighten the thresholds against real data rather than guesses. ---
    use_shadow_variant: bool = False
    shadow_sat_min: int = 35     # shaded red loses saturation as well as brightness
    shadow_val_min: int = 15     # dark, but still above near-black background/soil
    shadow_lab_a_min: int = 120  # a weaker, but still clearly reddish, Lab 'a' signal

    use_highlight_variant: bool = False
    highlight_hue_max: int = 22  # extends toward orange (OpenCV hue 0-179 scale)
    highlight_sat_min: int = 60
    highlight_val_min: int = 130  # bright only -- this is what distinguishes a sunlit
                                   # highlight from dull orange-brown soil/branches

    # --- Shared post-processing ---
    morph_kernel_px: int = 5
    # Minimum region area to keep (rejects speckle/noise before circle fitting)
    min_area_px: int = 30
    # Watershed: foreground seeds are distance-transform pixels above this
    # fraction of that connected component's own max distance.
    watershed_dist_frac: float = 0.5

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _colour_mask(h: np.ndarray, s: np.ndarray, v: np.ndarray, a_channel: np.ndarray,
                  hue_low_max: int, hue_high_min: int, sat_min: int, val_min: int, lab_a_min: int) -> np.ndarray:
    """One HSV+Lab colour-variant test, combined with AND (both cues must agree)."""
    hue_mask = (h <= hue_low_max) | (h >= hue_high_min)
    sv_mask = (s >= sat_min) & (v >= val_min)
    lab_mask = a_channel >= lab_a_min
    return hue_mask & sv_mask & lab_mask


def segment_red_mask(rgb: np.ndarray, config: ClassicalConfig) -> np.ndarray:
    """Binary mask (uint8, 0 or 255) of pixels that look like red apple skin.

    Real apple skin spans a range of appearances in one image, not one
    colour: a well-lit, saturated red; a darker, less saturated red where
    the fruit falls into its own or a leaf's shadow; and a washed-out,
    more orange/yellow patch where direct sun overexposes the skin. Each
    is tested as its own HSV+Lab colour variant (hue/sat/val AND Lab 'a'
    channel must all agree within that variant -- the usual single-colour
    cross-check), and a pixel counts as apple-coloured if it matches ANY
    enabled variant (OR'd together). This trades a bit more false-positive
    risk (more colours, more chances for a coincidental match) for
    recovering apples the single, well-lit-only threshold would miss
    entirely under shade or glare -- see docs/stage-6-notes.md's "shade
    kills recall" failure mode, which this is meant to fix.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    a_channel = lab[..., 1]

    combined_mask = _colour_mask(
        h, s, v, a_channel,
        config.hue_low_max, config.hue_high_min, config.sat_min, config.val_min, config.lab_a_min,
    )

    if config.use_shadow_variant:
        combined_mask |= _colour_mask(
            h, s, v, a_channel,
            config.hue_low_max, config.hue_high_min,
            config.shadow_sat_min, config.shadow_val_min, config.shadow_lab_a_min,
        )

    if config.use_highlight_variant:
        combined_mask |= _colour_mask(
            h, s, v, a_channel,
            config.highlight_hue_max, config.hue_high_min,
            config.highlight_sat_min, config.highlight_val_min, config.lab_a_min,
        )

    return (combined_mask.astype(np.uint8)) * 255


def clean_mask(mask: np.ndarray, config: ClassicalConfig) -> np.ndarray:
    """Morphological open (erode then dilate -- removes small speckle noise)
    followed by close (dilate then erode -- fills small holes inside a
    blob, e.g. a stem highlight)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.morph_kernel_px, config.morph_kernel_px))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def split_touching_apples(mask: np.ndarray, rgb: np.ndarray, config: ClassicalConfig) -> np.ndarray:
    """Distance-transform + watershed split of a cleaned binary mask.

    Two touching apples usually form one connected blob in `mask`. The
    distance transform is highest at each apple's own centre and dips
    where two apples touch, so thresholding it (per connected component,
    not globally -- a small apple's whole distance transform could be
    below a global threshold) gives one separate "seed" per apple, and
    watershed grows each seed out to the blob's true boundary.

    Returns an int32 label image, same (H, W) as mask: 0 = background,
    -1 = watershed boundary, N = the N-th detected region (1-indexed).
    """
    n_components, components = cv2.connectedComponents(mask)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    sure_fg = np.zeros_like(mask)
    for comp_id in range(1, n_components):
        comp_mask = components == comp_id
        comp_max = dist[comp_mask].max() if comp_mask.any() else 0
        if comp_max <= 0:
            continue
        sure_fg[comp_mask & (dist >= config.watershed_dist_frac * comp_max)] = 255

    sure_bg = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.watershed(bgr, markers)
    return markers


def fit_circle_to_contour(points: np.ndarray) -> tuple[float, float, float] | None:
    """Least-squares (Kasa method) circle fit to a set of (x, y) contour points.

    For a circle of centre (cx, cy) and radius r, every point (x, y) on it
    satisfies x^2+y^2 = 2*cx*x + 2*cy*y + (r^2 - cx^2 - cy^2), which is
    linear in [2cx, 2cy, d], so it solves with ordinary least squares.

    Returns (cx, cy, radius) in pixels, or None if there are too few points.
    """
    if points.shape[0] < 5:
        return None
    x, y = points[:, 0].astype(float), points[:, 1].astype(float)
    A = np.stack([2 * x, 2 * y, np.ones_like(x)], axis=1)
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, d = sol
    radius_sq = d + cx**2 + cy**2
    if radius_sq <= 0:
        return None
    return float(cx), float(cy), float(np.sqrt(radius_sq))


@dataclass
class ClassicalDetection:
    cx: float
    cy: float
    radius_px: float
    area_px: int  # used as a simple "confidence" score (bigger, cleaner blob = more confident)


def detect_apples(rgb: np.ndarray, config: ClassicalConfig) -> list[ClassicalDetection]:
    """Full classical detection pipeline for one RGB image. Returns one
    ClassicalDetection per detected apple."""
    mask = segment_red_mask(rgb, config)
    mask = clean_mask(mask, config)
    markers = split_touching_apples(mask, rgb, config)

    detections = []
    for label in range(2, markers.max() + 1):  # 1 = background after +1 shift
        region_mask = (markers == label).astype(np.uint8) * 255
        area = int(cv2.countNonZero(region_mask))
        if area < config.min_area_px:
            continue
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
        fit = fit_circle_to_contour(contour)
        if fit is None:
            continue
        cx, cy, r = fit
        if r < 2:  # degenerate fit
            continue
        detections.append(ClassicalDetection(cx=cx, cy=cy, radius_px=r, area_px=area))
    return detections
