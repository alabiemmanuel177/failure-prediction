"""Load and validate candidate fault parameters from the Research 2 repository."""

from __future__ import annotations

from pathlib import Path

import yaml


FAMILIES = {
    "none", "camera_occlusion", "lidar_dropout", "wheel_slip",
    "localisation_perturbation", "dynamic_blockage", "planner_oscillation",
    "semantic_corruption",
}
SEVERITIES = {"none", "low", "medium", "high"}


def load_fault_parameters(root: Path, family: str, severity: str) -> dict:
    if family not in FAMILIES:
        raise ValueError(f"unknown fault family: {family}")
    if family == "none":
        if severity != "none":
            raise ValueError("family none requires severity none")
        return {}
    if severity not in SEVERITIES - {"none"}:
        raise ValueError(f"fault family {family} requires low, medium, or high severity")
    path = root / "configs" / "faults" / f"{family}.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if document.get("family") != family:
        raise ValueError(f"fault file {path} declares a different family")
    if document.get("status") not in {"candidate", "frozen"}:
        raise ValueError(f"fault file {path} has an invalid status")
    values = (document.get("severities") or {}).get(severity)
    if not isinstance(values, dict):
        raise ValueError(f"fault file {path} has no severity {severity}")
    if any(value is None for value in values.values()):
        raise ValueError(f"fault file {path} has unresolved parameters")
    return dict(values)

