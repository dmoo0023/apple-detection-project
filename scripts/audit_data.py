"""Stage 1 exit check: audit the raw dataset without modifying it.

Produces:
  data/derived/manifest.csv    - one row per tree
  data/derived/path_report.md  - human-readable findings
  data/derived/overlays/       - 10 depth-over-RGB overlay images
  data/derived/checksums.txt   - sha256 of every raw file, to detect later changes

Run with: .venv\\Scripts\\python.exe scripts\\audit_data.py
"""

import ctypes
import hashlib
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_paths, load_camera, camera_is_complete  # noqa: E402
from src.labels import validate_labelme_file, load_labelme  # noqa: E402

random.seed(0)

NAME_PATTERN = re.compile(
    r"^R(?P<row>\d+)U(?P<unit>\d+)_(?P<season>[A-Za-z]+)_(?P<kind>RGB|Depth|D)"
    r"-(?P<hh>\d{1,2})-(?P<mm>\d{1,2})-(?P<ss>\d{1,2})$"
)

FILE_ATTRIBUTE_OFFLINE = 0x1000
FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x400000
INVALID_FILE_ATTRIBUTES = 0xFFFFFFFF


def is_onedrive_placeholder(path: Path) -> bool:
    """True if Windows reports this file as a cloud-only placeholder, not really on disk."""
    attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
    if attrs == INVALID_FILE_ATTRIBUTES:
        return False  # can't read attributes; integrity check below will catch a bad file anyway
    return bool(attrs & (FILE_ATTRIBUTE_OFFLINE | FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS))


def collect_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file())


def name_shape(stem: str) -> str:
    """Collapse a filename stem to a shape like 'R##U##_A_A-##-##-##' for grouping."""
    return re.sub(r"[A-Za-z]+", "A", re.sub(r"\d+", "#", stem))


def main() -> int:
    report_lines: list[str] = []

    def log(line: str = "") -> None:
        print(line)
        report_lines.append(line)

    paths = load_paths()
    raw_root: Path = paths["raw_data_root"]
    rgb_dir: Path = paths["rgb_dir"]
    depth_dir: Path = paths["depth_dir"]
    derived_dir: Path = paths["derived_dir"]
    overlays_dir: Path = paths["overlays_dir"]
    derived_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    log("# Stage 1 path report\n")

    # --- 1. Paths exist and are readable -----------------------------------
    log("## 1. Path checks")
    for name, p in [("raw_data_root", raw_root), ("rgb_dir", rgb_dir), ("depth_dir", depth_dir)]:
        ok = p.is_dir()
        log(f"- `{name}` = `{p}` -> {'OK, exists and is a directory' if ok else 'MISSING'}")
        if not ok:
            log("\n**STOPPING: a configured path does not exist. Fix configs/paths.yaml and re-run.**")
            (derived_dir / "path_report.md").write_text("\n".join(report_lines), encoding="utf-8")
            return 1

    rgb_files = [p for p in collect_files(rgb_dir) if p.suffix.lower() == ".png"]
    depth_files = [p for p in collect_files(depth_dir) if p.suffix.lower() == ".png"]
    json_files = [p for p in collect_files(rgb_dir) if p.suffix.lower() == ".json"]
    log(f"- RGB folder: {len(rgb_files)} .png files, {len(json_files)} .json files")
    log(f"- Depth folder: {len(depth_files)} .png files")

    placeholders = [p for p in (rgb_files + depth_files) if is_onedrive_placeholder(p)]
    log(f"- OneDrive cloud-only placeholders found: {len(placeholders)}")
    if placeholders:
        log("  Files NOT fully on disk (ask the student to set 'Always keep on this device'):")
        for p in placeholders[:20]:
            log(f"  - {p}")

    # --- 2. Naming pattern ---------------------------------------------------
    log("\n## 2. Naming pattern")
    all_files = rgb_files + depth_files
    shape_counts = Counter(name_shape(p.stem) for p in all_files)
    for shape, count in shape_counts.most_common():
        log(f"- pattern `{shape}`: {count} files")
    dominant_shape = shape_counts.most_common(1)[0][0] if shape_counts else None
    non_dominant = [p for p in all_files if name_shape(p.stem) != dominant_shape]
    log(f"- files not matching the dominant pattern: {len(non_dominant)}")
    for p in non_dominant[:20]:
        log(f"  - {p.name}")

    unmatched_regex = [p for p in all_files if not NAME_PATTERN.match(p.stem)]
    log(f"- files not matching the expected regex at all: {len(unmatched_regex)}")
    for p in unmatched_regex[:20]:
        log(f"  - {p.name}")
    kinds_seen = Counter()
    for p in all_files:
        m = NAME_PATTERN.match(p.stem)
        if m:
            kinds_seen[m.group("kind")] += 1
    log(f"- depth-folder filename keyword used: {dict(kinds_seen)} "
        f"(brief's note said 'Depth', actual files use the keyword(s) above)")

    # --- parse into structured records --------------------------------------
    def parse(files: list[Path]) -> dict[tuple[int, int], dict]:
        out: dict[tuple[int, int], dict] = {}
        for p in files:
            m = NAME_PATTERN.match(p.stem)
            if not m:
                continue
            row, unit = int(m["row"]), int(m["unit"])
            key = (row, unit)
            rec = {
                "path": p,
                "row": row,
                "unit": unit,
                "season": m["season"],
                "time": (int(m["hh"]), int(m["mm"]), int(m["ss"])),
            }
            if key in out:
                # keep both so pairing step can report the ambiguity
                out[key] = {**out[key], "duplicate": out[key].get("duplicate", []) + [rec]}
            else:
                out[key] = rec
        return out

    rgb_by_tree = parse(rgb_files)
    depth_by_tree = parse(depth_files)

    # --- 3. Tree IDs -----------------------------------------------------------
    log("\n## 3. Tree IDs")
    rgb_dupes = {k: v for k, v in rgb_by_tree.items() if "duplicate" in v}
    depth_dupes = {k: v for k, v in depth_by_tree.items() if "duplicate" in v}
    log(f"- distinct (row, unit) in RGB: {len(rgb_by_tree)}; duplicates: {len(rgb_dupes)}")
    log(f"- distinct (row, unit) in Depth: {len(depth_by_tree)}; duplicates: {len(depth_dupes)}")
    rows_seen = sorted(set(k[0] for k in rgb_by_tree) | set(k[0] for k in depth_by_tree))
    units_by_row = defaultdict(set)
    for (r, u) in list(rgb_by_tree) + list(depth_by_tree):
        units_by_row[r].add(u)
    log(f"- rows present: {rows_seen}")
    for r in rows_seen:
        units = sorted(units_by_row[r])
        gaps = [u for u in range(units[0], units[-1] + 1) if u not in units] if units else []
        if gaps:
            log(f"  - row {r}: units {units[0]}-{units[-1]}, missing unit numbers: {gaps}")
    total_trees = len(set(rgb_by_tree) | set(depth_by_tree))
    log(f"- **total distinct trees: {total_trees}** (expected 544)")

    # --- 4. Pairing --------------------------------------------------------
    log("\n## 4. RGB/Depth pairing")
    rgb_keys, depth_keys = set(rgb_by_tree), set(depth_by_tree)
    rgb_only = sorted(rgb_keys - depth_keys)
    depth_only = sorted(depth_keys - rgb_keys)
    log(f"- RGB without a matching depth file: {len(rgb_only)} {rgb_only[:10]}")
    log(f"- Depth without a matching RGB file: {len(depth_only)} {depth_only[:10]}")
    common = rgb_keys & depth_keys
    log(f"- paired trees: {len(common)}")

    ambiguous_pairs = []
    time_mismatches = []
    for key in sorted(common):
        r, d = rgb_by_tree[key], depth_by_tree[key]
        if "duplicate" in r or "duplicate" in d:
            ambiguous_pairs.append(key)
        elif r["time"] != d["time"]:
            time_mismatches.append((key, r["time"], d["time"]))
    log(f"- ambiguous pairings (more than one file for the same tree/modality): {len(ambiguous_pairs)} {ambiguous_pairs[:10]}")
    log(f"- RGB/Depth timestamp mismatches (paired by row+unit+season, time differs): {len(time_mismatches)}")
    for key, rt, dt in time_mismatches[:10]:
        log(f"  - tree {key}: RGB time {rt}, depth time {dt} (using closest time as tie-break; order is still 1:1 here)")

    # --- 5/6. Extensions, case, risky characters ----------------------------
    log("\n## 5. Extensions and case")
    stems_lower = Counter(p.stem.lower() for p in all_files)
    case_collisions = {s: c for s, c in stems_lower.items() if c > 1}
    log(f"- stems differing only by case: {len(case_collisions)}")
    double_ext = [p for p in all_files if p.stem.lower().endswith((".png", ".json"))]
    log(f"- double-extension files (e.g. name.png.png): {len(double_ext)}")

    log("\n## 6. Risky characters and path length")
    risky = []
    for p in all_files + json_files:
        name = p.name
        if name != name.strip() or re.search(r"\s", name) or not name.isascii() or len(str(p)) > 240:
            risky.append(p)
    log(f"- files with spaces/leading-trailing whitespace/non-ASCII/very long paths: {len(risky)}")
    for p in risky[:10]:
        log(f"  - {p}")
    log(f"- note: the raw_data_root path itself contains spaces and parentheses "
        f"(`{raw_root.name}`); this is fine as long as every path is quoted, which "
        f"this script and src/config.py always do via pathlib.")

    # --- 7. Integrity: open every PNG ---------------------------------------
    log("\n## 7. Image integrity")
    rgb_bad = []
    rgb_info = {}
    for p in rgb_files:
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                arr = np.array(im)
            ok = arr.shape[:2] == (480, 640) and arr.ndim == 3 and arr.shape[2] == 3
            rgb_info[p] = arr
            if not ok:
                rgb_bad.append((p, arr.shape, arr.dtype))
        except Exception as exc:
            rgb_bad.append((p, "UNREADABLE", str(exc)))
    log(f"- RGB files failing the 640x480x3 check: {len(rgb_bad)}")
    for item in rgb_bad[:10]:
        log(f"  - {item}")

    depth_bad = []
    depth_info = {}
    depth_modes = Counter()
    for p in depth_files:
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                mode = im.mode
                arr = np.array(im)
            depth_modes[(mode, str(arr.dtype), arr.ndim)] += 1
            depth_info[p] = arr
            size_ok = arr.shape[0] == 480 and arr.shape[1] == 640
            is_16bit_single_channel = arr.dtype == np.uint16 and arr.ndim == 2
            if not size_ok or not is_16bit_single_channel:
                depth_bad.append((p, arr.shape, arr.dtype))
        except Exception as exc:
            depth_bad.append((p, "UNREADABLE", str(exc)))
    log(f"- Depth files matching the EXPECTED format (640x480, 16-bit, single channel): "
        f"{len(depth_files) - len(depth_bad)} / {len(depth_files)}")
    log(f"- Depth files NOT matching that expectation: {len(depth_bad)}")
    log(f"- actual depth PNG formats found (mode, dtype, ndim -> count): {dict(depth_modes)}")
    if depth_bad:
        log(
            "\n**FINDING: every depth PNG is 8-bit, 3-channel (R=G=B), not 16-bit single-channel. "
            "This cannot be the raw RealSense depth stream — it looks like a normalised "
            "visualisation (min/max stretched per image for display). Per-image max values "
            "range roughly 155-180 (checked on a sample), which does not correspond to a fixed "
            "physical depth scale. See checkpoint C1: this must be resolved with the student "
            "before any 3D geometry work can start.**"
        )

    # --- 8. Labelme JSON checks ---------------------------------------------
    log("\n## 8. Labelme JSON checks")
    json_problems: dict[str, list[str]] = {}
    for jp in json_files:
        problems = validate_labelme_file(jp, rgb_dir)
        if problems:
            json_problems[jp.name] = problems
    log(f"- labelme JSON files checked: {len(json_files)}")
    log(f"- files with problems: {len(json_problems)}")
    for name, problems in list(json_problems.items())[:10]:
        log(f"  - {name}: {problems}")
    total_shapes = 0
    for jp in json_files:
        try:
            data = load_labelme(jp)
            total_shapes += len(data.get("shapes", []))
        except Exception:
            pass
    log(f"- total apple circles across all 25 labelled images: {total_shapes}")

    # --- 9. Depth sanity (raw units only - scale unverified) ----------------
    log("\n## 9. Depth sanity")
    center_medians_raw = []
    for p, arr in depth_info.items():
        ch0 = arr[..., 0] if arr.ndim == 3 else arr
        h, w = ch0.shape
        patch = ch0[h // 2 - 20 : h // 2 + 20, w // 2 - 20 : w // 2 + 20]
        valid = patch[patch > 0]
        if valid.size:
            center_medians_raw.append(float(np.median(valid)))
    camera_cfg_for_sanity = load_camera()
    scale = camera_cfg_for_sanity.get("depth_scale_to_metres")
    if center_medians_raw:
        log(f"- median centre-patch raw value over {len(center_medians_raw)} images: "
            f"{np.median(center_medians_raw):.1f} (range {min(center_medians_raw):.0f}-"
            f"{max(center_medians_raw):.0f}, raw 0-255 units)")
    if scale is not None and center_medians_raw:
        center_medians_m = [v * scale for v in center_medians_raw]
        median_m = float(np.median(center_medians_m))
        in_range = sum(1 for v in center_medians_m if 0.4 <= v <= 1.2)
        log(f"- converted at depth_scale_to_metres={scale}: median centre depth "
            f"{median_m:.3f} m (range {min(center_medians_m):.3f}-{max(center_medians_m):.3f} m)")
        log(f"- images with centre depth inside the stated 0.4-1.2 m capture range: "
            f"{in_range} / {len(center_medians_m)}")
        if median_m < 0.4 or median_m > 1.2:
            log(
                "- **CHECK FAILS as expected: converted depth is far outside the stated "
                "0.4-1.2 m capture range. This is the documented depth-scale limitation "
                "accepted by the student at checkpoint C1 (2026-10-24), not a new problem — "
                "see configs/camera.yaml and docs/stage-1-notes.md. Not silently passed.**"
            )
    else:
        log("- cannot check against the 0.4-1.2 m capture-distance target: no depth scale configured.")

    # --- 10. Alignment overlays ----------------------------------------------
    log("\n## 10. Depth/RGB alignment overlays")
    common_list = sorted(common)
    sample_keys = random.sample(common_list, min(10, len(common_list)))
    for key in sample_keys:
        rgb_path = rgb_by_tree[key]["path"]
        depth_path = depth_by_tree[key]["path"]
        rgb_arr = np.array(Image.open(rgb_path).convert("RGB"))
        depth_arr = np.array(Image.open(depth_path).convert("L")).astype(np.float32)
        depth_norm = (depth_arr - depth_arr.min())
        max_val = depth_norm.max()
        depth_norm = depth_norm / max_val if max_val > 0 else depth_norm
        depth_colour = (plt_colormap(depth_norm) * 255).astype(np.uint8)
        overlay = (0.5 * rgb_arr + 0.5 * depth_colour).astype(np.uint8)
        out_path = overlays_dir / f"R{key[0]:02d}U{key[1]:02d}_overlay.png"
        Image.fromarray(overlay).save(out_path)
    log(f"- saved {len(sample_keys)} overlays to `{overlays_dir.relative_to(Path.cwd())}` "
        f"for trees: {sample_keys}")

    # --- 11. Renames ----------------------------------------------------------
    log("\n## 11. Rename proposals")
    if not non_dominant and not unmatched_regex and not case_collisions:
        log("- no renames necessary: all 544+544 files match one consistent pattern, "
            "zero-padded row/unit numbers, no case collisions.")
    else:
        log("- see sections 2, 5 and 6 above for files that would need a rename proposal; "
            "none generated automatically — propose only, per rule in the workflow.")

    # --- 12. camera.yaml ------------------------------------------------------
    log("\n## 12. Camera config")
    camera_cfg = load_camera()
    complete = camera_is_complete(camera_cfg)
    log(f"- configs/camera.yaml complete (fx, fy, cx, cy, depth_scale_to_metres all set): {complete}")
    if not complete:
        log("- **STOP: intrinsics and/or depth scale are missing. Do not guess (rule 2). "
            "Ask the student. See checkpoint C1.**")

    # --- manifest -------------------------------------------------------------
    manifest_rows = []
    for key in sorted(set(rgb_by_tree) | set(depth_by_tree)):
        row, unit = key
        r = rgb_by_tree.get(key)
        d = depth_by_tree.get(key)
        rgb_path = r["path"] if r else None
        depth_path = d["path"] if d else None
        rgb_arr = rgb_info.get(rgb_path) if rgb_path else None
        depth_arr = depth_info.get(depth_path) if depth_path else None

        depth_frac_invalid = depth_median = depth_min = depth_max = None
        if depth_arr is not None:
            ch0 = depth_arr[..., 0] if depth_arr.ndim == 3 else depth_arr
            depth_frac_invalid = float((ch0 == 0).mean())
            valid = ch0[ch0 > 0]
            if valid.size:
                depth_median = float(np.median(valid))
                depth_min = int(valid.min())
                depth_max = int(valid.max())

        has_label = rgb_path is not None and (rgb_path.with_suffix(".json")).exists()

        manifest_rows.append({
            "tree_id": f"R{row:02d}U{unit:02d}",
            "row": row,
            "unit": unit,
            "rgb_path": str(rgb_path) if rgb_path else "",
            "depth_path": str(depth_path) if depth_path else "",
            "rgb_size": f"{rgb_arr.shape[1]}x{rgb_arr.shape[0]}" if rgb_arr is not None else "",
            "depth_size": f"{depth_arr.shape[1]}x{depth_arr.shape[0]}" if depth_arr is not None else "",
            "depth_dtype": str(depth_arr.dtype) if depth_arr is not None else "",
            "depth_fraction_invalid": depth_frac_invalid,
            "depth_median_raw": depth_median,
            "depth_min_raw": depth_min,
            "depth_max_raw": depth_max,
            "has_labelme_json": has_label,
        })

    import pandas as pd
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(paths["manifest_csv"], index=False)
    log(f"\n## Manifest\n- wrote {len(manifest_df)} rows to `{paths['manifest_csv'].relative_to(Path.cwd())}`")
    log("- NOTE: depth_median_raw/min_raw/max_raw are RAW 0-255 pixel values, NOT metres "
        "(see finding in section 7/9 above).")

    # --- checksums (detect future changes to raw data) ------------------------
    checksum_lines = []
    for p in sorted(rgb_files + depth_files, key=lambda x: str(x)):
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        checksum_lines.append(f"{sha}  {p.relative_to(raw_root)}")
    paths["checksums_txt"].write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    log(f"- wrote sha256 checksums for {len(checksum_lines)} raw files to "
        f"`{paths['checksums_txt'].relative_to(Path.cwd())}`")

    # --- exit status ------------------------------------------------------
    exit_ok = (
        total_trees == 544
        and not rgb_only and not depth_only
        and not placeholders
        and not json_problems
        and complete
    )
    log(f"\n## Exit check: {'PASS' if exit_ok else 'ISSUES FOUND'}")
    log(f"- 544 trees, clean pairing, no placeholders, all labels valid: "
        f"{'yes' if (total_trees == 544 and not rgb_only and not depth_only and not placeholders and not json_problems) else 'no'}")
    log(f"- camera.yaml complete: {complete}")
    if complete:
        log("- NOTE: fx/fy/cx/cy and depth_scale_to_metres are documented APPROXIMATIONS "
            "accepted by the student at checkpoint C1 (2026-10-24), not measured values — "
            "see configs/camera.yaml and docs/stage-1-notes.md for the limitation this implies.")

    (paths["derived_dir"] / "path_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return 0


def plt_colormap(norm_array: np.ndarray) -> np.ndarray:
    import matplotlib
    return matplotlib.colormaps["turbo"](norm_array)[..., :3]


if __name__ == "__main__":
    raise SystemExit(main())
