"""Stage 5 tests (the workflow's own exit check): a trivial predictor (the
labels themselves) must score perfectly, and a deliberately shifted copy
must give the expected error -- not some other number.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluate import (  # noqa: E402
    Apple2D,
    average_precision_at_iou,
    match_all,
    per_match_errors,
    precision_recall_f1,
    summarise,
)
from src.evaluate import DISTANCE_BANDS  # noqa: E402
from src.geometry import Intrinsics  # noqa: E402

INTR = Intrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0)
CONST_DEPTH_M = 0.70


def make_gt(image_id: str = "img1") -> list[Apple2D]:
    return [
        Apple2D(image_id, cx=200.0, cy=150.0, radius_px=20.0),
        Apple2D(image_id, cx=400.0, cy=300.0, radius_px=15.0),
        Apple2D(image_id, cx=100.0, cy=350.0, radius_px=25.0),
    ]


def make_depth_lookup(image_ids: list[str]) -> dict[str, np.ndarray]:
    return {iid: np.full((480, 640), CONST_DEPTH_M, dtype=float) for iid in image_ids}


# --- trivial predictor: predictions == labels -------------------------------

def test_trivial_predictor_scores_perfectly() -> None:
    gt = make_gt()
    pred = [Apple2D(a.image_id, a.cx, a.cy, a.radius_px, score=1.0) for a in gt]
    depth_lookup = make_depth_lookup(["img1"])

    matches = match_all(gt, pred)
    det = precision_recall_f1(matches)
    assert det["tp"] == 3
    assert det["fp"] == 0
    assert det["fn"] == 0
    assert det["precision"] == pytest.approx(1.0)
    assert det["recall"] == pytest.approx(1.0)
    assert det["f1"] == pytest.approx(1.0)

    ap = average_precision_at_iou(gt, pred, iou_threshold=0.5)
    assert ap == pytest.approx(1.0)

    match_df = per_match_errors(gt, pred, matches, depth_lookup, INTR)
    assert (match_df["error_2d_px"] < 1e-9).all()
    assert (match_df["error_2d_cm"] < 1e-6).all()
    assert (match_df["error_3d_proxy_cm"] < 1e-6).all()

    summary = summarise(gt, pred, matches, match_df)
    assert summary["overall"]["median_error_2d_px"] == pytest.approx(0.0, abs=1e-9)
    assert summary["overall"]["median_error_3d_proxy_cm"] == pytest.approx(0.0, abs=1e-6)
    assert summary["overall"]["within_2cm_frac"] == pytest.approx(1.0)
    assert summary["overall"]["within_3cm_frac"] == pytest.approx(1.0)


# --- deliberately shifted copy: expected, predictable error -----------------

def test_shifted_copy_gives_expected_2d_error() -> None:
    gt = make_gt()
    dx, dy = 3.0, 4.0  # 3-4-5 shift (5px), comfortably under every gt radius's 0.5*r threshold (10, 7.5, 12.5)
    pred = [Apple2D(a.image_id, a.cx + dx, a.cy + dy, a.radius_px, score=1.0) for a in gt]
    depth_lookup = make_depth_lookup(["img1"])

    matches = match_all(gt, pred)
    det = precision_recall_f1(matches)
    assert det["tp"] == 3  # shift (5px) stays under every gt radius's 0.5*r threshold (10, 7.5, 12.5)
    assert det["fp"] == 0 and det["fn"] == 0

    expected_px_error = np.hypot(dx, dy)  # = 5.0
    match_df = per_match_errors(gt, pred, matches, depth_lookup, INTR)
    assert np.allclose(match_df["error_2d_px"].to_numpy(), expected_px_error, atol=1e-9)

    expected_cm_error = 100 * np.hypot(dx * CONST_DEPTH_M / INTR.fx, dy * CONST_DEPTH_M / INTR.fy)
    assert np.allclose(match_df["error_2d_cm"].to_numpy(), expected_cm_error, atol=1e-6)

    # 3D proxy error should be close to the 2D cm error (same depth, same radius -> the
    # surface-to-centre offset is nearly parallel for two nearby rays at this depth/shift).
    assert np.all(np.abs(match_df["error_3d_proxy_cm"].to_numpy() - expected_cm_error) < 0.05)


def test_shift_beyond_threshold_breaks_the_match() -> None:
    gt = [Apple2D("img1", cx=200.0, cy=150.0, radius_px=20.0)]  # threshold = 0.5*20 = 10px
    pred = [Apple2D("img1", cx=235.0, cy=150.0, radius_px=20.0)]  # 35px shift, well past the threshold

    matches = match_all(gt, pred)
    det = precision_recall_f1(matches)
    assert det["tp"] == 0
    assert det["fp"] == 1
    assert det["fn"] == 1


def test_missing_prediction_counts_as_false_negative() -> None:
    gt = make_gt()
    pred = [Apple2D(a.image_id, a.cx, a.cy, a.radius_px) for a in gt[:2]]  # drop the 3rd apple
    matches = match_all(gt, pred)
    det = precision_recall_f1(matches)
    assert det["tp"] == 2
    assert det["fn"] == 1
    assert det["fp"] == 0
    assert det["recall"] == pytest.approx(2 / 3)
    assert det["precision"] == pytest.approx(1.0)


def test_summarise_reports_out_of_range_band_not_silently() -> None:
    # every DISTANCE_BANDS entry is >= 0.4m; a depth below that must still show up
    # in the report as "out_of_range", not just vanish from the per-band table.
    assert all(lo >= 0.4 for lo, hi in DISTANCE_BANDS)
    gt = [Apple2D("img1", cx=200.0, cy=150.0, radius_px=20.0)]
    pred = [Apple2D("img1", cx=200.0, cy=150.0, radius_px=20.0)]
    depth_lookup = {"img1": np.full((480, 640), 0.15, dtype=float)}  # below every band

    matches = match_all(gt, pred)
    match_df = per_match_errors(gt, pred, matches, depth_lookup, INTR)
    summary = summarise(gt, pred, matches, match_df)

    assert "out_of_range" in summary["by_distance_band"]
    assert summary["by_distance_band"]["out_of_range"]["n"] == 1
    assert all(band not in summary["by_distance_band"] for band in [f"{lo:.1f}-{hi:.1f}m" for lo, hi in DISTANCE_BANDS])


def test_extra_prediction_counts_as_false_positive() -> None:
    gt = make_gt()
    pred = [Apple2D(a.image_id, a.cx, a.cy, a.radius_px) for a in gt]
    pred.append(Apple2D("img1", cx=500.0, cy=50.0, radius_px=10.0))  # spurious extra detection
    matches = match_all(gt, pred)
    det = precision_recall_f1(matches)
    assert det["tp"] == 3
    assert det["fp"] == 1
    assert det["fn"] == 0
    assert det["precision"] == pytest.approx(3 / 4)
    assert det["recall"] == pytest.approx(1.0)
