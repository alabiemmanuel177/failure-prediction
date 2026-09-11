"""Shared fail-closed boundary for artifacts derived from protected episodes."""

from __future__ import annotations


def enforce_protected_boundary(
    protected_test_used: object, *, explicitly_allowed: bool,
    confirmatory_gate_passed: bool,
) -> None:
    if protected_test_used is False:
        return
    if protected_test_used is not True:
        raise ValueError("source must explicitly declare protected_test_used")
    if not explicitly_allowed:
        raise ValueError("refusing protected data access without explicit approval")
    if not confirmatory_gate_passed:
        raise ValueError("refusing protected data access before confirmatory freeze")


HELD_OUT_SPLIT = "held_out_map_test"
HELD_OUT_ASSIGNED_STATUS = "assigned_after_model_freeze"
FREEZE_DECLARATION_FIELDS = (
    "model_selection_complete", "calibration_selection_complete",
    "threshold_selection_complete",
)


def model_freeze_findings(freeze: object) -> list[str]:
    """Mirror the model-freeze checks of ``check_readiness --stage confirmatory``."""
    findings: list[str] = []
    if not isinstance(freeze, dict):
        return ["model freeze document must be a mapping"]
    if freeze.get("frozen") is not True:
        findings.append("model_freeze:frozen is not true")
    if freeze.get("protected_outcomes_consulted") is not False:
        findings.append("model freeze must precede protected-outcome inspection")
    declaration = freeze.get("declaration") or {}
    for field in FREEZE_DECLARATION_FIELDS:
        if declaration.get(field) is not True:
            findings.append(f"model_freeze:declaration.{field} is not true")
    if declaration.get("protected_maps_or_outcomes_inspected") is not False:
        findings.append("model_freeze: protected maps or outcomes must be uninspected")

    def walk(value: object, path: str = ""):
        if isinstance(value, dict):
            for key, child in value.items():
                yield from walk(child, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from walk(child, f"{path}[{index}]")
        else:
            yield path, value

    for field, value in walk(freeze):
        if isinstance(value, str) and value.startswith("TODO"):
            findings.append(f"model_freeze:{field} is {value}")
    return findings


def held_out_assignment_findings(splits: object, map_ids=()) -> list[str]:
    """Require an assigned post-freeze held-out split that covers ``map_ids``."""
    if not isinstance(splits, dict):
        return ["split manifest must be a mapping"]
    split = splits.get(HELD_OUT_SPLIT) or {}
    findings: list[str] = []
    if split.get("status") != HELD_OUT_ASSIGNED_STATUS:
        findings.append(
            f"splits:{HELD_OUT_SPLIT}.status is {split.get('status')!r}, "
            f"expected {HELD_OUT_ASSIGNED_STATUS!r}"
        )
    if not split.get("maps"):
        findings.append(f"splits:{HELD_OUT_SPLIT}.maps is empty")
    if not split.get("routes"):
        findings.append(f"splits:{HELD_OUT_SPLIT}.routes is empty")
    for map_id in map_ids:
        if map_id not in (split.get("maps") or []):
            findings.append(f"map {map_id} is not an assigned held-out map")
    return findings


def held_out_campaign_gate(root, map_ids=(), *, run_readiness: bool = True) -> list[str]:
    """Fail closed unless the model freeze and protected split assignment exist.

    ``root`` is the repository root. When ``run_readiness`` is true the full
    ``scripts/check_readiness.py --stage confirmatory`` gate is executed as well.
    """
    from pathlib import Path
    import subprocess
    import sys

    import yaml

    root = Path(root)
    findings: list[str] = []
    freeze_path = root / "configs" / "model_freeze.yaml"
    if not freeze_path.exists():
        findings.append("model freeze missing: configs/model_freeze.yaml")
    else:
        findings.extend(
            model_freeze_findings(yaml.safe_load(freeze_path.read_text(encoding="utf-8")))
        )
    splits_path = root / "data" / "manifests" / "splits.template.yaml"
    if not splits_path.exists():
        findings.append("split manifest missing: data/manifests/splits.template.yaml")
    else:
        findings.extend(held_out_assignment_findings(
            yaml.safe_load(splits_path.read_text(encoding="utf-8")), map_ids
        ))
    if run_readiness and not findings:
        gate = subprocess.run(
            [sys.executable, str(root / "scripts" / "check_readiness.py"),
             "--stage", "confirmatory"],
            check=False, capture_output=True, text=True,
        )
        if gate.returncode:
            findings.append(
                "check_readiness --stage confirmatory failed:\n" + gate.stdout.strip()
            )
    return findings
