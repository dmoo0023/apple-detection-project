"""Read labelme circle annotations.

A labelme "circle" shape stores two points: the centre, and one point on the
circumference. This module only reads/validates that format for the Stage 1
audit; conversion to YOLO labels happens in Stage 2.
"""

import json
import math
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
