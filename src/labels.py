"""Read labelme circle annotations and convert them to YOLO label formats.

A labelme "circle" shape stores two points: the centre, and one point on the
circumference. Stage 1 uses this module to validate that format; Stage 2
uses it to convert circles into YOLO detection (square box) and
segmentation (polygon) labels, plus a flat per-apple CSV.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path


class LabelmeError(ValueError):
    """Raised when a labelme JSON file is structurally invalid."""


def load_labelme(json_path: Path) -> dict:
    """Load a labelme JSON file, tolerating a UTF-8 BOM (labelme writes one on Windows)."""
    with open(json_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def circle_centre_radius(shape: dict) -> tuple[float, float, float]:
    """Convert a labelme circle shape to (cx, cy, radius), all in pixels.

    Inputs: shape["points"] = [[cx, cy], [px, py]] where (px, py) is any
    point on the circle's circumference.
    Output: pixel coordinates in the image's (u, v) convention — u is the
    column (x, rightwards), v is the row (y, downwards), origin top-left.
    """
    if shape.get("shape_type") != "circle":
        raise LabelmeError(f"expected shape_type 'circle', got {shape.get('shape_type')!r}")
    points = shape.get("points")
    if not points or len(points) != 2:
        raise LabelmeError(f"circle shape must have exactly 2 points, got {points!r}")
    (cx, cy), (px, py) = points
    radius = math.hypot(px - cx, py - cy)
    return float(cx), float(cy), float(radius)


def validate_labelme_file(json_path: Path, rgb_dir: Path) -> list[str]:
    """Check one labelme JSON file against the checks in Stage 1 of the workflow.

    Returns a list of human-readable problem strings; empty list means the
    file passed every check.
    """
    problems: list[str] = []
    try:
        data = load_labelme(json_path)
    except Exception as exc:  # malformed JSON, bad encoding, etc.
        return [f"could not parse JSON: {exc}"]

    image_path_field = data.get("imagePath")
    if not image_path_field:
        problems.append("missing imagePath")
    else:
        resolved = (json_path.parent / image_path_field).resolve()
        if not resolved.exists():
            problems.append(f"imagePath does not resolve to a real file: {image_path_field}")
        expected_stem = json_path.stem
        if Path(image_path_field).stem != expected_stem:
            problems.append(
                f"imagePath stem {Path(image_path_field).stem!r} does not match JSON stem {expected_stem!r}"
            )

    width = data.get("imageWidth")
    height = data.get("imageHeight")
    if width != 640 or height != 480:
        problems.append(f"imageWidth/imageHeight = {width}x{height}, expected 640x480")

    shapes = data.get("shapes", [])
    if not shapes:
        problems.append("no shapes (image has no labelled apples)")

    labels_seen = set()
    for i, shape in enumerate(shapes):
        labels_seen.add(shape.get("label"))
        if shape.get("shape_type") != "circle":
            problems.append(f"shape {i}: shape_type is {shape.get('shape_type')!r}, expected 'circle'")
            continue
        points = shape.get("points")
        if not points or len(points) != 2:
            problems.append(f"shape {i}: expected 2 points, got {len(points) if points else 0}")
            continue
        cx, cy, radius = circle_centre_radius(shape)
        if radius < 4:
            problems.append(f"shape {i}: radius {radius:.1f}px is under 4px")
        if not (0 <= cx < width and 0 <= cy < height):
            problems.append(f"shape {i}: centre ({cx:.1f}, {cy:.1f}) is outside the image")

    if len(labels_seen) > 1:
        problems.append(f"inconsistent label names in one file: {labels_seen}")

    return problems


# --- Stage 2: conversion to YOLO formats -----------------------------------

MIN_RADIUS_PX = 4.0


@dataclass
class Apple:
    """One labelled apple, in pixel coordinates (u rightwards, v downwards, origin top-left)."""
    index: int
    cx: float
    cy: float
    radius: float


def extract_apples(json_path: Path) -> tuple[list[Apple], list[str]]:
    """Read every circle shape in a labelme file as a list of Apple records.

    Returns (apples, warnings). Warnings cover: circles whose centre falls
    outside the image, circles with radius under MIN_RADIUS_PX, and an image
    with zero apples. Does not raise on a malformed file -- callers should
    already have validated the file with validate_labelme_file if they need
    a hard failure.
    """
    data = load_labelme(json_path)
    width, height = data.get("imageWidth"), data.get("imageHeight")
    apples: list[Apple] = []
    warnings: list[str] = []

    for i, shape in enumerate(data.get("shapes", [])):
        if shape.get("shape_type") != "circle":
            continue
        cx, cy, radius = circle_centre_radius(shape)
        if radius < MIN_RADIUS_PX:
            warnings.append(f"{json_path.name} shape {i}: radius {radius:.1f}px is under {MIN_RADIUS_PX}px")
        if width is not None and height is not None and not (0 <= cx < width and 0 <= cy < height):
            warnings.append(f"{json_path.name} shape {i}: centre ({cx:.1f}, {cy:.1f}) is outside the image")
        apples.append(Apple(index=len(apples), cx=cx, cy=cy, radius=radius))

    if not apples:
        warnings.append(f"{json_path.name}: image has no apples")

    return apples, warnings


def circle_to_polygon(cx: float, cy: float, radius: float, n_points: int = 16) -> list[tuple[float, float]]:
    """Approximate a circle as an n_points-sided polygon, in pixel coordinates."""
    return [
        (cx + radius * math.cos(2 * math.pi * k / n_points), cy + radius * math.sin(2 * math.pi * k / n_points))
        for k in range(n_points)
    ]


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def yolo_detection_line(apple: Apple, img_w: int, img_h: int, class_id: int = 0) -> str:
    """A square YOLO detection label line: 'class x_center y_center width height', all normalised 0-1.

    The box is 2*radius square, clipped to the image bounds (so a labelled
    apple whose circle extends past the edge still gets a valid in-image box).
    """
    x0 = _clip(apple.cx - apple.radius, 0, img_w)
    y0 = _clip(apple.cy - apple.radius, 0, img_h)
    x1 = _clip(apple.cx + apple.radius, 0, img_w)
    y1 = _clip(apple.cy + apple.radius, 0, img_h)
    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = x1 - x0, y1 - y0
    return f"{class_id} {xc / img_w:.6f} {yc / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}"


def yolo_segmentation_line(apple: Apple, img_w: int, img_h: int, class_id: int = 0, n_points: int = 16) -> str:
    """A YOLO segmentation label line: 'class x1 y1 x2 y2 ... xn yn', all normalised 0-1, points clipped to the image."""
    points = circle_to_polygon(apple.cx, apple.cy, apple.radius, n_points)
    clipped = [(_clip(x, 0, img_w) / img_w, _clip(y, 0, img_h) / img_h) for x, y in points]
    flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in clipped)
    return f"{class_id} {flat}"


def parse_yolo_detection_line(line: str, img_w: int, img_h: int) -> tuple[float, float, float]:
    """Inverse of yolo_detection_line: returns (cx, cy, radius) recovered from a box, in pixels.

    Radius is recovered as half the mean of the box's width and height
    (they're equal unless the box was clipped at an image edge).
    """
    _, xc, yc, w, h = line.split()
    xc, yc, w, h = float(xc) * img_w, float(yc) * img_h, float(w) * img_w, float(h) * img_h
    return xc, yc, (w + h) / 4.0


def parse_yolo_segmentation_line(line: str, img_w: int, img_h: int) -> tuple[float, float, float]:
    """Inverse of yolo_segmentation_line: returns (cx, cy, radius) recovered from a polygon, in pixels.

    Centre is the mean of the polygon points; radius is the mean distance
    from the centre to each point.
    """
    parts = line.split()
    coords = [float(v) for v in parts[1:]]
    xs = [coords[i] * img_w for i in range(0, len(coords), 2)]
    ys = [coords[i] * img_h for i in range(1, len(coords), 2)]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    radius = sum(math.hypot(x - cx, y - cy) for x, y in zip(xs, ys)) / len(xs)
    return cx, cy, radius
