# Stage 0: Environment

## What this stage does

Before any data work, the project needs a working Python environment that can
use the GPU, and a git repository to track every change. "Environment" means
the specific versions of Python, PyTorch and other libraries installed on
this machine — getting this wrong (e.g. a PyTorch build that doesn't support
this GPU) causes confusing failures much later, so it is checked first and in
isolation.

## What was done

1. Created a **virtual environment** (an isolated copy of Python with its own
   packages, so this project's library versions don't clash with anything
   else on the machine) at `C:\Users\melbo\dev\apple-detection-project\.venv`.
   This is **outside OneDrive** on purpose: OneDrive continuously syncs files
   in the background, and a virtual environment contains tens of thousands of
   small files that change during installs. Running a venv (or git) inside a
   synced folder causes file-lock errors and sync conflicts. The raw data
   stays in OneDrive; the code, environment and git history live outside it.
2. Installed **PyTorch 2.11.0 with CUDA 12.8** (`+cu128` build), plus
   `torchvision`, `ultralytics` (YOLO), `opencv-python`, `numpy`, `pandas`,
   `scipy`, `scikit-image`, `matplotlib`, `pyyaml`, `pytest`, and `open3d`.
   The CUDA version matters: this machine has an RTX 5070, a newer GPU
   architecture ("Blackwell") that only recent PyTorch builds know how to
   generate instructions for. An older PyTorch build (e.g. CUDA 11.8) would
   either refuse to use the GPU or silently run on the much slower CPU.
3. Wrote `scripts/check_env.py`. It doesn't just call
   `torch.cuda.is_available()` — that can return `True` even when the
   installed build cannot actually run on this specific GPU. The script also
   runs a real matrix multiplication on the GPU and waits for it to finish
   (`torch.cuda.synchronize()`), which is a much stronger proof the GPU path
   works end to end.
4. Ran `pip freeze > requirements.txt` so the exact set of installed package
   versions is recorded. A comment at the top of the file explains that
   `torch`/`torchvision` must be installed first from PyTorch's CUDA 12.8
   package index, because `pip freeze` does not record where a package came
   from — only its name and version.
5. Initialised git (`git init`), renamed the default branch to `main`, and
   added a `.gitignore` that excludes the virtual environment, raw/derived
   images, and model weight files (these are either regenerable, too large
   for git, or — in the case of raw data — never supposed to be copied into
   the repository at all, per rule 1 in the workflow).

## Result

`scripts/check_env.py` printed `CUDA available: True`, the device name
`NVIDIA GeForce RTX 5070`, and `GPU CHECK PASSED` after a real GPU
computation. The repository has an initial commit with the scaffold,
`.gitignore`, `requirements.txt` and this note.

## What could go wrong later

- If the venv or repo is accidentally created *inside* the OneDrive folder in
  a future session, move it out before continuing — do not just add it to
  `.gitignore` and leave it there, since OneDrive will still try to sync it.
- If a future `pip install -r requirements.txt` on a different machine fails
  to find `torch==2.11.0+cu128`, the CUDA index was not used — see the
  comment at the top of `requirements.txt`.
- Package versions will drift over time (e.g. a newer `ultralytics` release).
  If results stop reproducing, check `requirements.txt` against what is
  actually installed.

## Terms used

- **Virtual environment (venv):** an isolated set of installed Python
  packages, separate from any other Python installation on the machine.
- **CUDA:** NVIDIA's platform that lets libraries like PyTorch run
  computations on the GPU instead of the CPU.
