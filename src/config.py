"""Load configs/paths.yaml and configs/camera.yaml.

Every script and module gets its paths from here instead of hard-coding
them (rule 8 in agent_workflow.md).
"""

from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_yaml(relative_path: str) -> dict:
    full_path = REPO_ROOT / relative_path
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_paths() -> dict:
    """Return the parsed contents of configs/paths.yaml.

    Relative paths in the file (derived_dir, manifest_csv, etc.) are
    resolved against the repository root; raw_data_root is used as given
    since it is an absolute Windows path.
    """
    cfg = _load_yaml("configs/paths.yaml")
    resolved = dict(cfg)
    resolved["raw_data_root"] = Path(cfg["raw_data_root"])
    resolved["rgb_dir"] = resolved["raw_data_root"] / cfg["rgb_subdir"]
    resolved["depth_dir"] = resolved["raw_data_root"] / cfg["depth_subdir"]
    for key in ("derived_dir", "manifest_csv", "path_report_md", "overlays_dir", "checksums_txt"):
        resolved[key] = REPO_ROOT / cfg[key]
    return resolved


def load_camera() -> dict:
    """Return the parsed contents of configs/camera.yaml (may contain nulls)."""
    return _load_yaml("configs/camera.yaml")


def camera_is_complete(camera_cfg: dict) -> bool:
    required = ("fx", "fy", "cx", "cy", "depth_scale_to_metres")
    return all(camera_cfg.get(k) is not None for k in required)
