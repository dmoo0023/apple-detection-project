"""Stage 2 exit check: convert labelme circle labels to YOLO formats.

Produces, for every labelled image:
  data/derived/labels_det/<image_id>.txt   - YOLO detection labels (square box)
  data/derived/labels_seg/<image_id>.txt   - YOLO segmentation labels (polygon)
  data/derived/apples_2d.csv               - one row per apple (image, index, cx, cy, radius_px)
  data/derived/stage2_overlays/            - 10 images with labels drawn on, for visual checking
  data/derived/stage2_report.md            - warnings (circles outside image, tiny radius, no apples)

Run with: .venv\\Scripts\\python.exe scripts\\convert_labels.py
"""

import random
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_paths  # noqa: E402
from src.labels import (  # noqa: E402
    extract_apples,
    yolo_detection_line,
    yolo_segmentation_line,
)

random.seed(0)


def draw_overlay(rgb_path: Path, apples: list, out_path: Path) -> None:
    """Draw labelled circles (red), YOLO boxes (green) and centres (blue) on an image and save it."""
    img = cv2.imread(str(rgb_path))
    for apple in apples:
        cx, cy, r = int(round(apple.cx)), int(round(apple.cy)), int(round(apple.radius))
        cv2.circle(img, (cx, cy), r, (0, 0, 255), 2)  # labelled circle, red
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), (0, 255, 0), 1)  # YOLO square box, green
        cv2.drawMarker(img, (cx, cy), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)
    cv2.imwrite(str(out_path), img)


def main() -> int:
    """Run Stage 2: convert every labelme JSON to YOLO detection/segmentation labels and
    apples_2d.csv, save sample overlays and a report. Returns 0 on success.
    """
    report_lines: list[str] = ["# Stage 2 label conversion report\n"]

    def log(line: str = "") -> None:
        """Print a line and also keep it for the markdown report."""
        print(line)
        report_lines.append(line)

    paths = load_paths()
    rgb_dir = paths["rgb_dir"]
    derived_dir = paths["derived_dir"]
    det_dir = derived_dir / "labels_det"
    seg_dir = derived_dir / "labels_seg"
    overlay_dir = derived_dir / "stage2_overlays"
    for d in (det_dir, seg_dir, overlay_dir):
        d.mkdir(parents=True, exist_ok=True)

    json_files = sorted(rgb_dir.glob("*.json"))
    log(f"Found {len(json_files)} labelme JSON files.\n")

    csv_rows = []
    all_warnings: list[str] = []
    converted = []  # (image_id, rgb_path, apples) for overlay sampling

    for jp in json_files:
        image_id = jp.stem
        rgb_path = jp.with_suffix(".png")
        apples, warnings = extract_apples(jp)
        all_warnings.extend(warnings)

        img_w, img_h = 640, 480  # confirmed in Stage 1 for every image

        det_lines = [yolo_detection_line(a, img_w, img_h) for a in apples]
        seg_lines = [yolo_segmentation_line(a, img_w, img_h) for a in apples]
        (det_dir / f"{image_id}.txt").write_text("\n".join(det_lines) + ("\n" if det_lines else ""), encoding="utf-8")
        (seg_dir / f"{image_id}.txt").write_text("\n".join(seg_lines) + ("\n" if seg_lines else ""), encoding="utf-8")

        for a in apples:
            csv_rows.append({"image_id": image_id, "apple_index": a.index, "cx": a.cx, "cy": a.cy, "radius_px": a.radius})

        converted.append((image_id, rgb_path, apples))

    apples_df = pd.DataFrame(csv_rows)
    apples_csv_path = derived_dir / "apples_2d.csv"
    apples_df.to_csv(apples_csv_path, index=False)
    log(f"Converted {len(json_files)} images, {len(apples_df)} apples total.")
    log(f"Wrote {apples_csv_path.relative_to(Path.cwd())}")
    log(f"Wrote YOLO detection labels to {det_dir.relative_to(Path.cwd())}/")
    log(f"Wrote YOLO segmentation labels to {seg_dir.relative_to(Path.cwd())}/")

    log(f"\n## Warnings ({len(all_warnings)})")
    for w in all_warnings:
        log(f"- {w}")
    if not all_warnings:
        log("- none: no circles outside the image, none under the radius threshold, no empty images.")

    sample = random.sample(converted, min(10, len(converted)))
    for image_id, rgb_path, apples in sample:
        draw_overlay(rgb_path, apples, overlay_dir / f"{image_id}_labels.png")
    log(f"\n## Overlays\nSaved {len(sample)} labelled overlays to {overlay_dir.relative_to(Path.cwd())}/ "
        f"(red = labelme circle, green = YOLO detection box, blue cross = centre).")

    (derived_dir / "stage2_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    exit_ok = len(json_files) >= 25 and len(apples_df) > 0
    log(f"\n## Exit check: {'PASS' if exit_ok else 'ISSUES FOUND'} "
        f"({len(json_files)} labelled images converted, >= the 25 from Stage 1)")
    return 0 if exit_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
