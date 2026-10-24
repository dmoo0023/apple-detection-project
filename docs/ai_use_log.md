# AI use log

One entry per stage: date, what the agent was asked to do, what it produced.
The "Student review notes" line under each entry is for the student to fill
in themselves; the agent never writes on that line.

## Stage 0: Environment — 2026-10-24

**Asked:** Set up a Python environment outside OneDrive with GPU-enabled
PyTorch and the project's other dependencies, write a GPU check script, and
initialise git with a `.gitignore` and first commit.

**Produced:** `.venv` at `C:\Users\melbo\dev\apple-detection-project\.venv`,
`requirements.txt`, `scripts/check_env.py`, `.gitignore`, git repo
initialised with `main` branch, `docs/stage-0-notes.md`.

**Student review notes:**

## Stage 1: Data audit and path checks — 2026-10-24

**Asked:** Audit the raw dataset (paths, naming, pairing, integrity,
labelme labels, depth sanity), without modifying any raw files, and verify
camera intrinsics and depth scale rather than assuming RealSense defaults.

**Produced:** `configs/paths.yaml`, `scripts/audit_data.py`,
`data/derived/manifest.csv`, `data/derived/path_report.md`,
`tests/test_paths.py`, depth/RGB overlays, `docs/stage-1-notes.md`. Camera
intrinsics and depth scale could **not** be verified from the data — see
checkpoint C1 below.

**Student review notes:**

## Checkpoint C1 resolution — 2026-10-24

**Asked:** Student reviewed the overlays (confirmed alignment correct) and
decided, after being shown the concrete numeric consequences, how to handle
the missing depth scale and camera intrinsics (no camera/capture setup
available to measure either).

**Produced:** Updated `configs/camera.yaml` with documented approximations
(depth scale = RealSense mm convention applied directly to the 8-bit
values; intrinsics derived from Intel's published D435i FOV spec, cx/cy at
image centre), both flagged with long comments and a stated error range.
Updated `scripts/audit_data.py` to run the 0.4-1.2 m depth sanity check
honestly against the confirmed scale (it fails, as expected, and the
failure is printed rather than hidden). Updated `docs/stage-1-notes.md`
with the full reasoning and the practical implication: absolute 3D
accuracy cannot be rigorously verified with this setup. Stage 1 exit check
now passes; tagging `stage-1-done` and merging to `main`.

**Student review notes:**

## Stage 2: Label conversion — 2026-10-24

**Asked:** Convert the 25 labelme circle labels into YOLO detection and
segmentation formats plus a flat per-apple CSV, in one shared module, with
warnings for bad circles and a round-trip test.

**Produced:** Conversion functions added to `src/labels.py`
(`extract_apples`, `yolo_detection_line`, `yolo_segmentation_line`, and
their inverses), `scripts/convert_labels.py`, `data/derived/apples_2d.csv`,
`data/derived/labels_det/`, `data/derived/labels_seg/`,
`data/derived/stage2_overlays/`, `tests/test_labels.py` (11 new tests, all
passing), `docs/stage-2-notes.md`. All 25 images converted with zero
warnings (781 apples).

**Student review notes:**
