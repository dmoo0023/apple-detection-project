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
