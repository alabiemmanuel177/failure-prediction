"""Synthetic prediction tables and fake repositories for confirmatory-tooling tests."""

from __future__ import annotations

import csv
from pathlib import Path
import random
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]
UNASSIGNED_HELD_OUT_BLOCK = (
    "held_out_map_test:\n  maps: []\n  routes: []\n  inspect_only_after_policy_freeze: true\n"
)
CONTRACT_COLUMNS = (
    "run_id", "decision_index", "decision_time", "split", "map_id", "route_id",
    "fault_family", "severity", "seed", "protected_test_used", "eligibility", "label",
    "primary_event_class", "primary_event_time", "model_id", "raw_score", "risk_score",
)
FAMILIES = (
    "camera_occlusion", "lidar_dropout", "wheel_slip", "localisation_perturbation",
    "dynamic_blockage", "planner_oscillation", "semantic_corruption",
)
EVENT_TIME = 30.0
WARNING_HORIZON = 10.0
TOO_LATE_GUARD = 1.0
NEGATIVE_GUARD = 5.0


def held_out_episodes(*, maps=("test_00", "test_01", "test_02"), routes_per_map=2, seeds=2,
                      split="held_out_map_test", event_classes=("collision",)) -> list[dict]:
    """A small paired design: clean + seven families per map/route/seed."""
    episodes = []
    rng = random.Random(1)
    for map_id in maps:
        for route_index in range(routes_per_map):
            route_id = f"{map_id}_r{route_index}"
            for seed in range(seeds):
                for family in ("none", *FAMILIES):
                    run_id = f"{map_id}-{route_id}-{family}-{seed}"
                    has_event = family != "none" and rng.random() < 0.8
                    event_class = rng.choice(event_classes) if has_event else None
                    episodes.append({
                        "run_id": run_id, "split": split, "map_id": map_id, "route_id": route_id,
                        "fault_family": family, "severity": "none" if family == "none" else "medium",
                        "seed": 10_000_000 + seed, "primary_event_class": event_class,
                        "primary_event_time": EVENT_TIME if has_event else None,
                    })
    return episodes


def table_rows(
    episodes: list[dict], *, model_id: str, protected: bool, detect_probability: float,
    false_alert_probability: float, lead_seconds: float = 4.0, seed: int = 0,
    calibrate=lambda value: value,
) -> list[dict]:
    """Contract-column rows plus applied alarm decisions on a 0.5 s grid."""
    rng = random.Random(seed)
    rows = []
    for episode in episodes:
        event_time = episode["primary_event_time"]
        detect = event_time is not None and rng.random() < detect_probability
        false_alert = rng.random() < false_alert_probability
        alarm_time = event_time - lead_seconds if detect else None
        false_time = 12.0 if false_alert else None
        for index in range(60):
            decision_time = 10.0 + 0.5 * index
            if event_time is None:
                eligibility = "eligible_negative"
                label = 0
            elif decision_time > event_time:
                eligibility, label = "excluded_post_event", 0
            elif event_time - WARNING_HORIZON <= decision_time <= event_time - TOO_LATE_GUARD:
                eligibility, label = "eligible_positive", 1
            elif decision_time > event_time - TOO_LATE_GUARD:
                eligibility, label = "excluded_too_late", 0
            elif decision_time >= event_time - WARNING_HORIZON - NEGATIVE_GUARD:
                eligibility, label = "excluded_near_event_or_injection", 0
            else:
                eligibility, label = "eligible_negative", 0
            raw = 0.15 + 0.1 * rng.random() + (0.5 if label else 0.0)
            alarm = (alarm_time is not None and abs(decision_time - alarm_time) < 1e-9) or (
                false_time is not None and abs(decision_time - false_time) < 1e-9
            )
            rows.append({
                "run_id": episode["run_id"], "decision_index": index, "decision_time": decision_time,
                "split": episode["split"], "map_id": episode["map_id"], "route_id": episode["route_id"],
                "fault_family": episode["fault_family"], "severity": episode["severity"],
                "seed": episode["seed"], "protected_test_used": str(protected).lower(),
                "eligibility": eligibility, "label": label,
                "primary_event_class": episode["primary_event_class"] or "",
                "primary_event_time": "" if event_time is None else event_time,
                "model_id": model_id, "raw_score": round(min(raw, 1.0), 4),
                "risk_score": round(min(calibrate(min(raw, 1.0)), 1.0), 4),
                "alarm": str(alarm).lower(), "persistent": str(alarm).lower(),
            })
    return rows


def write_table(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [*CONTRACT_COLUMNS, "alarm", "persistent"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def frozen_model_freeze() -> dict:
    document = yaml.safe_load(
        (ROOT / "configs/model_freeze.template.yaml").read_text(encoding="utf-8")
    )

    def fill(value):
        if isinstance(value, dict):
            return {key: fill(child) for key, child in value.items()}
        if isinstance(value, str) and value.startswith("TODO"):
            return "0" * 64 if "SHA" in value else "filled"
        return value

    document = fill(document)
    document["frozen"] = True
    document["declaration"] = {
        "model_selection_complete": True, "calibration_selection_complete": True,
        "threshold_selection_complete": True, "protected_maps_or_outcomes_inspected": False,
    }
    return document


def unassigned_splits_text() -> str:
    """The real split manifest with the held-out block reset to its pre-freeze form.

    The held-out split is assigned strictly after the model freeze, so the real
    manifest may already carry the assignment; tests that need the pre-assignment
    state work on this copy instead of the workspace file.
    """
    from scripts.assign_protected_split import replace_held_out_block

    text = (ROOT / "data/manifests/splits.template.yaml").read_text(encoding="utf-8")
    return replace_held_out_block(text, UNASSIGNED_HELD_OUT_BLOCK)


def unassigned_splits() -> dict:
    return yaml.safe_load(unassigned_splits_text())


def write_unfrozen_alarm_policy(path: Path) -> Path:
    """Copy configs/alarm_policy.yaml with its single threshold line reset to null."""
    text = (ROOT / "configs/alarm_policy.yaml").read_text(encoding="utf-8")
    pattern = re.compile(r"^threshold: .*$", re.MULTILINE)
    assert len(pattern.findall(text)) == 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pattern.sub("threshold: null", text, count=1), encoding="utf-8")
    return path


def assigned_splits(*, maps=("test_00", "test_01", "test_02"), routes_per_map=8) -> dict:
    splits = unassigned_splits()
    splits["held_out_map_test"] = {
        "maps": list(maps),
        "routes": [f"{map_id}_r{index}" for map_id in maps for index in range(routes_per_map)],
        "inspect_only_after_policy_freeze": True,
        "status": "assigned_after_model_freeze",
        "assigned_utc": "2026-09-03T00:00:00Z",
        "model_freeze_sha256": "0" * 64,
    }
    return splits


def fake_repository(root: Path, *, frozen: bool = True, assigned: bool = True) -> Path:
    """A minimal repository tree for src.protected_data.held_out_campaign_gate."""
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "data/manifests").mkdir(parents=True, exist_ok=True)
    if frozen:
        (root / "configs/model_freeze.yaml").write_text(
            yaml.safe_dump(frozen_model_freeze()), encoding="utf-8"
        )
    splits = assigned_splits() if assigned else unassigned_splits()
    (root / "data/manifests/splits.template.yaml").write_text(yaml.safe_dump(splits), encoding="utf-8")
    return root
