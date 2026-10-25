"""Stage 2 tests: labelme parsing, YOLO conversion, and round-trip accuracy."""

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.labels import (  # noqa: E402
    Apple,
    circle_centre_radius,
    circle_to_polygon,
    extract_apples,
    parse_yolo_detection_line,
    parse_yolo_segmentation_line,
    validate_labelme_file,
    write_labelme_json,
    yolo_detection_line,
    yolo_segmentation_line,
)

IMG_W, IMG_H = 640, 480


def test_circle_centre_radius() -> None:
    """A labelme circle gives its centre point and the distance to its second point as radius."""
    shape = {"shape_type": "circle", "points": [[100.0, 100.0], [103.0, 104.0]]}
    cx, cy, r = circle_centre_radius(shape)
    assert (cx, cy) == (100.0, 100.0)
    assert r == pytest.approx(5.0)  # 3-4-5 triangle


def test_circle_centre_radius_rejects_non_circle() -> None:
    """Non-circle shapes are rejected."""
    with pytest.raises(Exception):
        circle_centre_radius({"shape_type": "polygon", "points": [[0, 0], [1, 1]]})


def test_circle_to_polygon_points_are_on_the_circle() -> None:
    """Every polygon vertex lies exactly on the circle."""
    pts = circle_to_polygon(50.0, 60.0, 10.0, n_points=16)
    assert len(pts) == 16
    for x, y in pts:
        assert math.hypot(x - 50.0, y - 60.0) == pytest.approx(10.0, abs=1e-9)


# --- round-trip: an apple well inside the image, no clipping ----------------

@pytest.fixture
def interior_apple() -> Apple:
    """An apple well inside the image (no edge clipping)."""
    return Apple(index=0, cx=300.0, cy=200.0, radius=25.0)


def test_detection_round_trip(interior_apple: Apple) -> None:
    """Circle -> YOLO detection line -> circle returns the original centre and radius."""
    line = yolo_detection_line(interior_apple, IMG_W, IMG_H)
    cx, cy, r = parse_yolo_detection_line(line, IMG_W, IMG_H)
    assert cx == pytest.approx(interior_apple.cx, abs=0.01)
    assert cy == pytest.approx(interior_apple.cy, abs=0.01)
    assert r == pytest.approx(interior_apple.radius, abs=0.01)


def test_segmentation_round_trip(interior_apple: Apple) -> None:
    """Circle -> YOLO segmentation line -> circle returns the original centre and radius."""
    line = yolo_segmentation_line(interior_apple, IMG_W, IMG_H, n_points=16)
    cx, cy, r = parse_yolo_segmentation_line(line, IMG_W, IMG_H)
    assert cx == pytest.approx(interior_apple.cx, abs=0.01)
    assert cy == pytest.approx(interior_apple.cy, abs=0.01)
    assert r == pytest.approx(interior_apple.radius, abs=0.01)


def test_detection_box_clips_at_image_edge() -> None:
    """A box that would extend past the image edge is clipped to the image."""
    # centre 5px from the left edge with a 20px radius: the box would start at x=-15,
    # which must be clipped to 0, so the recovered box is no longer centred on the
    # true apple centre or the true radius -- this is expected, documented behaviour.
    near_edge = Apple(index=0, cx=5.0, cy=200.0, radius=20.0)
    line = yolo_detection_line(near_edge, IMG_W, IMG_H)
    cx, cy, r = parse_yolo_detection_line(line, IMG_W, IMG_H)
    assert cx != pytest.approx(near_edge.cx, abs=0.01)
    assert r < near_edge.radius


# --- extract_apples warnings -------------------------------------------------

def _write_labelme(tmp_path: Path, shapes: list[dict], width=IMG_W, height=IMG_H) -> Path:
    """Write a minimal labelme JSON with the given shapes and return its path."""
    data = {
        "version": "5.0.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": "fake.png",
        "imageData": None,
        "imageWidth": width,
        "imageHeight": height,
    }
    p = tmp_path / "fake.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_extract_apples_warns_on_tiny_radius(tmp_path: Path) -> None:
    """A circle with a radius under 4 px produces a warning."""
    shapes = [{"label": "Apple", "shape_type": "circle", "points": [[100, 100], [102, 100]]}]  # radius 2px
    p = _write_labelme(tmp_path, shapes)
    apples, warnings = extract_apples(p)
    assert len(apples) == 1
    assert any("under" in w for w in warnings)


def test_extract_apples_warns_on_out_of_bounds_centre(tmp_path: Path) -> None:
    """A circle centred outside the image produces a warning."""
    shapes = [{"label": "Apple", "shape_type": "circle", "points": [[700, 100], [710, 100]]}]  # cx > 640
    p = _write_labelme(tmp_path, shapes)
    apples, warnings = extract_apples(p)
    assert any("outside the image" in w for w in warnings)


def test_extract_apples_warns_on_empty_image(tmp_path: Path) -> None:
    """A label file with no apples produces a warning."""
    p = _write_labelme(tmp_path, [])
    apples, warnings = extract_apples(p)
    assert apples == []
    assert any("no apples" in w for w in warnings)


def test_extract_apples_no_warnings_on_clean_shape(tmp_path: Path) -> None:
    """A normal circle produces no warnings."""
    shapes = [{"label": "Apple", "shape_type": "circle", "points": [[300, 200], [320, 200]]}]  # radius 20px
    p = _write_labelme(tmp_path, shapes)
    apples, warnings = extract_apples(p)
    assert len(apples) == 1
    assert warnings == []


# --- Stage 7: writing pre-labels round-trips through the real labelme reader ---

def test_write_labelme_json_round_trips_through_extract_apples(tmp_path: Path) -> None:
    """A pre-label JSON written by write_labelme_json is valid and reads back the same circles."""
    (tmp_path / "fake.png").write_bytes(b"\x89PNG\r\n\x1a\n")  # just needs to exist for the path check
    detections = [(100.0, 80.0, 18.5), (300.0, 200.0, 12.0)]
    json_path = tmp_path / "fake.json"
    write_labelme_json(json_path, "fake.png", detections, img_w=640, img_h=480)

    problems = validate_labelme_file(json_path, tmp_path)
    assert not problems

    apples, warnings = extract_apples(json_path)
    assert warnings == []
    assert len(apples) == 2
    for a, (cx, cy, r) in zip(apples, detections):
        assert a.cx == pytest.approx(cx, abs=1e-6)
        assert a.cy == pytest.approx(cy, abs=1e-6)
        assert a.radius == pytest.approx(r, abs=1e-6)


# --- real dataset: every labelled image converts without exceptions --------
#
# The labelled count grows over time (the student labels more as the project
# progresses, formally via Stage 7's pre-labelling loop, but also ad hoc).
# This test does not hard-code that count -- it checks every labelled image
# that exists RIGHT NOW converts cleanly, whatever the count is.

def test_all_labelled_images_convert() -> None:
    """Every labelled image in the raw data converts without errors."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import load_paths

    rgb_dir = load_paths()["rgb_dir"]
    json_files = sorted(rgb_dir.glob("*.json"))
    assert len(json_files) >= 25, f"expected at least the 25 images found in Stage 1, found {len(json_files)}"

    total_apples = 0
    for jp in json_files:
        problems = validate_labelme_file(jp, rgb_dir)
        assert not problems, f"{jp.name}: {problems}"
        apples, warnings = extract_apples(jp)
        assert not warnings, f"{jp.name}: unexpected warnings {warnings}"
        for a in apples:
            det = yolo_detection_line(a, IMG_W, IMG_H)
            seg = yolo_segmentation_line(a, IMG_W, IMG_H)
            assert det.startswith("0 ")
            assert seg.startswith("0 ")
        total_apples += len(apples)
    assert total_apples > 0
