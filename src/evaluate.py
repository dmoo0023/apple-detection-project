"""The one shared evaluation module (rule 7). Every method -- colour masking
(Stage 6), YOLO (Stage 8), any hybrid (Stage 9) -- is scored by the same
code here, on the same metrics, so results across methods are directly
comparable instead of each method grading its own homework slightly
differently.

Two kinds of apples flow through this module:
  - "ground truth" (gt): from labelme circles (via src/labels.py)
  - "predictions" (pred): from a detector or classical pipeline

Both are represented as Apple2D (image_id, cx, cy, radius_px, score).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.geometry import Intrinsics, apple_centre_3d

DISTANCE_BANDS = [(0.4, 0.7), (0.7, 1.0), (1.0, 1.2)]


@dataclass
class Apple2D:
    image_id: str
    cx: float
    cy: float
    radius_px: float
    score: float = 1.0  # detection confidence; 1.0 for ground truth / scoreless methods


def distance_band(z_m: float | None) -> str:
    """Label which capture-distance band a depth falls in, for the breakdown in item 5."""
    if z_m is None or not np.isfinite(z_m):
        return "unknown"
    for lo, hi in DISTANCE_BANDS:
        if lo <= z_m < hi:
            return f"{lo:.1f}-{hi:.1f}m"
    return "out_of_range"


# --- 1. Matching -------------------------------------------------------------

@dataclass
class ImageMatchResult:
    image_id: str
    matched: list[tuple[int, int, float]] = field(default_factory=list)  # (gt_idx, pred_idx, distance_px)
    unmatched_gt: list[int] = field(default_factory=list)
    unmatched_pred: list[int] = field(default_factory=list)


def match_image(gt: list[Apple2D], pred: list[Apple2D], match_frac: float = 0.5) -> ImageMatchResult:
    """One-to-one Hungarian matching on centre distance, for apples in ONE image.

    A match counts only if the distance is under `match_frac` * the
    GROUND-TRUTH apple's radius (the brief's rule: 0.5 * labelled radius).
    Hungarian assignment (scipy linear_sum_assignment) finds the
    globally-optimal one-to-one pairing minimising total distance, rather
    than each prediction greedily grabbing the nearest gt apple (which can
    produce avoidable bad pairings when two apples are close together).
    """
    image_id = gt[0].image_id if gt else (pred[0].image_id if pred else "")
    result = ImageMatchResult(image_id=image_id)
    if not gt or not pred:
        result.unmatched_gt = list(range(len(gt)))
        result.unmatched_pred = list(range(len(pred)))
        return result

    cost = np.zeros((len(gt), len(pred)))
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            cost[i, j] = float(np.hypot(g.cx - p.cx, g.cy - p.cy))

    gt_idx, pred_idx = linear_sum_assignment(cost)

    matched_gt, matched_pred = set(), set()
    for i, j in zip(gt_idx, pred_idx):
        dist = cost[i, j]
        threshold = match_frac * gt[i].radius_px
        if dist < threshold:
            result.matched.append((int(i), int(j), float(dist)))
            matched_gt.add(i)
            matched_pred.add(j)

    result.unmatched_gt = [i for i in range(len(gt)) if i not in matched_gt]
    result.unmatched_pred = [j for j in range(len(pred)) if j not in matched_pred]
    return result


def match_all(gt: list[Apple2D], pred: list[Apple2D], match_frac: float = 0.5) -> dict[str, ImageMatchResult]:
    """match_image, grouped by image_id, over a whole dataset."""
    image_ids = sorted(set(a.image_id for a in gt) | set(a.image_id for a in pred))
    results = {}
    for image_id in image_ids:
        g = [a for a in gt if a.image_id == image_id]
        p = [a for a in pred if a.image_id == image_id]
        results[image_id] = match_image(g, p, match_frac)
    return results


# --- 2. Detection metrics ----------------------------------------------------

def precision_recall_f1(match_results: dict[str, ImageMatchResult]) -> dict:
    tp = sum(len(r.matched) for r in match_results.values())
    fp = sum(len(r.unmatched_pred) for r in match_results.values())
    fn = sum(len(r.unmatched_gt) for r in match_results.values())
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _iou_square(a: Apple2D, b: Apple2D) -> float:
    """IoU of the two apples' square detection boxes (2*radius side, centred on cx,cy) -- same
    box convention as src/labels.py's YOLO detection export."""
    ax0, ax1 = a.cx - a.radius_px, a.cx + a.radius_px
    ay0, ay1 = a.cy - a.radius_px, a.cy + a.radius_px
    bx0, bx1 = b.cx - b.radius_px, b.cx + b.radius_px
    by0, by1 = b.cy - b.radius_px, b.cy + b.radius_px

    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def average_precision_at_iou(gt: list[Apple2D], pred: list[Apple2D], iou_threshold: float = 0.5) -> float:
    """Single-class (apple) Average Precision at a fixed IoU threshold, computed the standard
    way: rank all predictions (across all images) by score, greedily assign each to its best
    unmatched same-image GT box if IoU >= threshold, then take the area under the resulting
    precision-recall curve (all-points / envelope interpolation).
    """
    if not gt:
        return float("nan")
    gt_by_image: dict[str, list[int]] = {}
    for i, g in enumerate(gt):
        gt_by_image.setdefault(g.image_id, []).append(i)
    gt_taken = [False] * len(gt)

    preds_sorted = sorted(enumerate(pred), key=lambda t: -t[1].score)
    tp = np.zeros(len(preds_sorted))
    fp = np.zeros(len(preds_sorted))

    for rank, (_, p) in enumerate(preds_sorted):
        best_iou, best_gt_i = 0.0, -1
        for gi in gt_by_image.get(p.image_id, []):
            if gt_taken[gi]:
                continue
            iou = _iou_square(gt[gi], p)
            if iou > best_iou:
                best_iou, best_gt_i = iou, gi
        if best_iou >= iou_threshold:
            tp[rank] = 1
            gt_taken[best_gt_i] = True
        else:
            fp[rank] = 1

    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    recall = tp_cum / len(gt)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    # envelope (precision is monotonically non-increasing as recall increases)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[precision[0] if len(precision) else 0.0], precision, [0.0]])
    ap = float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))
    return ap


# --- 3/4. Per-match 2D and 3D (proxy) error ----------------------------------

def per_match_errors(
    gt: list[Apple2D],
    pred: list[Apple2D],
    match_results: dict[str, ImageMatchResult],
    depth_lookup: dict[str, np.ndarray],
    intrinsics: Intrinsics,
    geometry_method: str = "project_and_offset",
) -> pd.DataFrame:
    """One row per matched apple pair: 2D pixel/cm error and the 3D proxy error.

    3D error is a PROXY (item 4): there is no physical ground truth, so
    "truth" here means the 3D centre computed from the LABELLED circle
    through the exact same geometry module used for the prediction. This
    measures how much the prediction's 2D circle disagrees with the label's
    2D circle, expressed in 3D -- not distance to a real, physically
    measured apple centre. State this wherever the numbers are reported.
    """
    rows = []
    gt_by_image: dict[str, list[Apple2D]] = {}
    pred_by_image: dict[str, list[Apple2D]] = {}
    for a in gt:
        gt_by_image.setdefault(a.image_id, []).append(a)
    for a in pred:
        pred_by_image.setdefault(a.image_id, []).append(a)

    for image_id, result in match_results.items():
        depth_m = depth_lookup.get(image_id)
        g_list = gt_by_image.get(image_id, [])
        p_list = pred_by_image.get(image_id, [])

        for gt_idx, pred_idx, dist_px in result.matched:
            g, p = g_list[gt_idx], p_list[pred_idx]

            row = {
                "image_id": image_id,
                "gt_index": gt_idx,
                "pred_index": pred_idx,
                "error_2d_px": dist_px,
                "error_2d_cm": np.nan,
                "error_3d_proxy_cm": np.nan,
                "z_m": np.nan,
                "distance_band": "unknown",
            }

            if depth_m is not None:
                gt_3d = apple_centre_3d(g.cx, g.cy, g.radius_px, depth_m, intrinsics, method=geometry_method)
                pred_3d = apple_centre_3d(p.cx, p.cy, p.radius_px, depth_m, intrinsics, method=geometry_method)

                if gt_3d.is_valid:
                    row["z_m"] = gt_3d.depth_sample.depth_m
                    row["distance_band"] = distance_band(gt_3d.depth_sample.depth_m)

                    du, dv = g.cx - p.cx, g.cy - p.cy
                    z = gt_3d.depth_sample.depth_m
                    dx, dy = du * z / intrinsics.fx, dv * z / intrinsics.fy
                    row["error_2d_cm"] = 100 * float(np.hypot(dx, dy))

                if gt_3d.is_valid and pred_3d.is_valid:
                    row["error_3d_proxy_cm"] = 100 * float(np.linalg.norm(gt_3d.centre_m - pred_3d.centre_m))

            rows.append(row)

    return pd.DataFrame(rows)


# --- 5/6. Summary -------------------------------------------------------------

def summarise(
    gt: list[Apple2D],
    pred: list[Apple2D],
    match_results: dict[str, ImageMatchResult],
    match_df: pd.DataFrame,
) -> dict:
    """Overall + per-distance-band summary, ready to print as a markdown table."""
    detection = precision_recall_f1(match_results)
    detection["mAP@0.5"] = average_precision_at_iou(gt, pred, iou_threshold=0.5)

    def _block(df: pd.DataFrame) -> dict:
        if df.empty:
            return {"n": 0}
        valid_3d = df["error_3d_proxy_cm"].dropna()
        return {
            "n": len(df),
            "median_error_2d_px": float(df["error_2d_px"].median()),
            "median_error_2d_cm": float(df["error_2d_cm"].median(skipna=True)),
            "median_error_3d_proxy_cm": float(valid_3d.median()) if len(valid_3d) else float("nan"),
            "within_2cm_frac": float((valid_3d <= 2.0).mean()) if len(valid_3d) else float("nan"),
            "within_3cm_frac": float((valid_3d <= 3.0).mean()) if len(valid_3d) else float("nan"),
        }

    summary = {"detection": detection, "overall": _block(match_df), "by_distance_band": {}}
    for band_label in [f"{lo:.1f}-{hi:.1f}m" for lo, hi in DISTANCE_BANDS] + ["unknown"]:
        sub = match_df[match_df["distance_band"] == band_label] if not match_df.empty else match_df
        if len(sub):
            summary["by_distance_band"][band_label] = _block(sub)
    return summary


def summary_to_markdown(summary: dict, method_name: str, split_name: str) -> str:
    d = summary["detection"]
    o = summary["overall"]
    lines = [
        f"### {method_name} on {split_name}",
        "",
        "**3D error is a proxy** (no physical ground truth): the distance between the 3D "
        "centre computed from the predicted circle and from the labelled circle, through "
        "the same geometry module -- not distance to a physically measured apple.",
        "",
        f"- Detection: precision={d['precision']:.3f}, recall={d['recall']:.3f}, "
        f"F1={d['f1']:.3f}, mAP@0.5={d['mAP@0.5']:.3f} (TP={d['tp']}, FP={d['fp']}, FN={d['fn']})",
    ]
    if o.get("n", 0):
        lines.append(
            f"- Overall ({o['n']} matched apples): median 2D error = {o['median_error_2d_px']:.2f}px "
            f"({o['median_error_2d_cm']:.2f}cm); median 3D proxy error = "
            f"{o['median_error_3d_proxy_cm']:.2f}cm; within 2cm = {100*o['within_2cm_frac']:.1f}%; "
            f"within 3cm = {100*o['within_3cm_frac']:.1f}%"
        )
    else:
        lines.append("- Overall: no matched apples")

    lines.append("\n| Distance band | n | median 2D err (cm) | median 3D proxy err (cm) | within 2cm | within 3cm |")
    lines.append("|---|---|---|---|---|---|")
    for band, b in summary["by_distance_band"].items():
        if b.get("n", 0):
            lines.append(
                f"| {band} | {b['n']} | {b['median_error_2d_cm']:.2f} | "
                f"{b['median_error_3d_proxy_cm']:.2f} | {100*b['within_2cm_frac']:.1f}% | "
                f"{100*b['within_3cm_frac']:.1f}% |"
            )
    return "\n".join(lines)
