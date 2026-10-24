# Stage 1 path report

## 1. Path checks
- `raw_data_root` = `C:\Users\melbo\OneDrive - Monash University\Documents\01_Monash\MMA3001\Apple Detection Project\Dataset (Summer Only)` -> OK, exists and is a directory
- `rgb_dir` = `C:\Users\melbo\OneDrive - Monash University\Documents\01_Monash\MMA3001\Apple Detection Project\Dataset (Summer Only)\RGB` -> OK, exists and is a directory
- `depth_dir` = `C:\Users\melbo\OneDrive - Monash University\Documents\01_Monash\MMA3001\Apple Detection Project\Dataset (Summer Only)\D` -> OK, exists and is a directory
- RGB folder: 544 .png files, 25 .json files
- Depth folder: 544 .png files
- OneDrive cloud-only placeholders found: 0

## 2. Naming pattern
- pattern `A#A#_A_A-#-#-#`: 1088 files
- files not matching the dominant pattern: 0
- files not matching the expected regex at all: 0
- depth-folder filename keyword used: {'RGB': 544, 'D': 544} (brief's note said 'Depth', actual files use the keyword(s) above)

## 3. Tree IDs
- distinct (row, unit) in RGB: 544; duplicates: 0
- distinct (row, unit) in Depth: 544; duplicates: 0
- rows present: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
- **total distinct trees: 544** (expected 544)

## 4. RGB/Depth pairing
- RGB without a matching depth file: 0 []
- Depth without a matching RGB file: 0 []
- paired trees: 544
- ambiguous pairings (more than one file for the same tree/modality): 0 []
- RGB/Depth timestamp mismatches (paired by row+unit+season, time differs): 0

## 5. Extensions and case
- stems differing only by case: 0
- double-extension files (e.g. name.png.png): 0

## 6. Risky characters and path length
- files with spaces/leading-trailing whitespace/non-ASCII/very long paths: 0
- note: the raw_data_root path itself contains spaces and parentheses (`Dataset (Summer Only)`); this is fine as long as every path is quoted, which this script and src/config.py always do via pathlib.

## 7. Image integrity
- RGB files failing the 640x480x3 check: 0
- Depth files matching the EXPECTED format (640x480, 16-bit, single channel): 0 / 544
- Depth files NOT matching that expectation: 544
- actual depth PNG formats found (mode, dtype, ndim -> count): {('RGB', 'uint8', 3): 544}

**FINDING: every depth PNG is 8-bit, 3-channel (R=G=B), not 16-bit single-channel. This cannot be the raw RealSense depth stream — it looks like a normalised visualisation (min/max stretched per image for display). Per-image max values range roughly 155-180 (checked on a sample), which does not correspond to a fixed physical depth scale. See checkpoint C1: this must be resolved with the student before any 3D geometry work can start.**

## 8. Labelme JSON checks
- labelme JSON files checked: 25
- files with problems: 0
- total apple circles across all 25 labelled images: 781

## 9. Depth sanity
- median centre-patch raw value over 542 images: 144.0 (range 116-173, raw 0-255 units)
- converted at depth_scale_to_metres=0.001: median centre depth 0.144 m (range 0.116-0.173 m)
- images with centre depth inside the stated 0.4-1.2 m capture range: 0 / 542
- **CHECK FAILS as expected: converted depth is far outside the stated 0.4-1.2 m capture range. This is the documented depth-scale limitation accepted by the student at checkpoint C1 (2026-10-24), not a new problem — see configs/camera.yaml and docs/stage-1-notes.md. Not silently passed.**

## 10. Depth/RGB alignment overlays
- saved 10 overlays to `data\derived\overlays` for trees: [(20, 26), (21, 24), (3, 2), (15, 7), (23, 27), (23, 1), (21, 8), (17, 13), (22, 36), (19, 28)]

## 11. Rename proposals
- no renames necessary: all 544+544 files match one consistent pattern, zero-padded row/unit numbers, no case collisions.

## 12. Camera config
- configs/camera.yaml complete (fx, fy, cx, cy, depth_scale_to_metres all set): True

## Manifest
- wrote 544 rows to `data\derived\manifest.csv`
- NOTE: depth_median_raw/min_raw/max_raw are RAW 0-255 pixel values, NOT metres (see finding in section 7/9 above).
- wrote sha256 checksums for 1088 raw files to `data\derived\checksums.txt`

## Exit check: PASS
- 544 trees, clean pairing, no placeholders, all labels valid: yes
- camera.yaml complete: True
- NOTE: fx/fy/cx/cy and depth_scale_to_metres are documented APPROXIMATIONS accepted by the student at checkpoint C1 (2026-10-24), not measured values — see configs/camera.yaml and docs/stage-1-notes.md for the limitation this implies.
