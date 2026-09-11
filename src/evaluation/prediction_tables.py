"""Shared I/O and boundary checks for the immutable prediction-table contract."""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

import yaml

from src.dataset_inventory import publish_new_bytes, sha256_file
from src.protected_data import enforce_protected_boundary


ROOT = Path(__file__).resolve().parents[2]

PREDICTION_COLUMNS = (
    "run_id", "decision_index", "decision_time", "split", "map_id", "route_id",
    "fault_family", "severity", "seed", "protected_test_used", "eligibility", "label",
    "primary_event_class", "primary_event_time", "model_id", "raw_score", "risk_score",
)
ALARM_COLUMNS = ("alarm", "persistent")
PROTECTED_SPLIT = "held_out_map_test"
ELIGIBLE = {"eligible_positive", "eligible_negative"}


def truth(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def read_prediction_table(
    path: Path, *, require_alarm: bool = False
) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        rows = list(reader)
    missing = [column for column in PREDICTION_COLUMNS if column not in fieldnames]
    if missing:
        raise ValueError(f"{path}: prediction table lacks columns {missing}")
    if require_alarm:
        absent = [column for column in ALARM_COLUMNS if column not in fieldnames]
        if absent:
            raise ValueError(f"{path}: prediction table lacks alarm columns {absent}")
    if not rows:
        raise ValueError(f"{path}: prediction table is empty")
    return rows


def table_fieldnames(rows: Sequence[Mapping[str, object]]) -> list[str]:
    names = list(PREDICTION_COLUMNS)
    for column in ALARM_COLUMNS:
        if column in rows[0]:
            names.append(column)
    return names


def prediction_table_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    fieldnames = table_fieldnames(rows)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: _cell(row.get(name)) for name in fieldnames})
    return buffer.getvalue().encode("utf-8")


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def write_prediction_table(path: Path, rows: Sequence[Mapping[str, object]]) -> str:
    """Publish an immutable prediction table and return its SHA-256."""
    payload = prediction_table_bytes(rows)
    publish_new_bytes(path, payload)
    return hashlib.sha256(payload).hexdigest()


def write_yaml_report(path: Path, payload: Mapping[str, object]) -> str:
    text = yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True)
    publish_new_bytes(path, text.encode("utf-8"))
    return sha256_file(path)


def group_episodes(rows: Sequence[Mapping[str, object]]) -> dict[str, list[dict[str, object]]]:
    episodes: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        episodes.setdefault(str(row["run_id"]), []).append(dict(row))
    for run_id, episode_rows in episodes.items():
        episode_rows.sort(key=lambda row: float(row["decision_time"]))
        times = [float(row["decision_time"]) for row in episode_rows]
        if any(later <= earlier for earlier, later in zip(times, times[1:])):
            raise ValueError(f"{run_id}: decision_time is not strictly increasing")
    return episodes


def clean_run_ids(rows: Sequence[Mapping[str, object]]) -> set[str]:
    return {str(row["run_id"]) for row in rows if row.get("fault_family") == "none"}


def model_id_of(rows: Sequence[Mapping[str, object]]) -> str:
    identifiers = {str(row["model_id"]) for row in rows}
    if len(identifiers) != 1:
        raise ValueError(f"prediction table mixes model ids: {sorted(identifiers)}")
    return identifiers.pop()


def splits_of(rows: Sequence[Mapping[str, object]]) -> set[str]:
    return {str(row["split"]) for row in rows}


def require_validation_only(rows: Sequence[Mapping[str, object]], purpose: str) -> None:
    splits = splits_of(rows)
    if splits != {"validation"}:
        raise ValueError(f"{purpose} accepts validation rows only, found splits {sorted(splits)}")
    if any(truth(row.get("protected_test_used", "false")) for row in rows):
        raise ValueError(f"protected outcomes cannot be used for {purpose}")


def touches_protected(rows: Sequence[Mapping[str, object]]) -> bool:
    return any(
        truth(row.get("protected_test_used", "false")) or row.get("split") == PROTECTED_SPLIT
        for row in rows
    )


def confirmatory_gate_passed(root: Path = ROOT) -> bool:
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "check_readiness.py"), "--stage", "confirmatory"],
        capture_output=True, text=True, check=False,
    )
    return result.returncode == 0


def guard_protected_rows(
    rows: Sequence[Mapping[str, object]], *, explicitly_allowed: bool, root: Path = ROOT
) -> bool:
    """Fail closed on protected rows; return whether protected data is present."""
    protected = touches_protected(rows)
    if not protected:
        return False
    gate = confirmatory_gate_passed(root) if explicitly_allowed else False
    enforce_protected_boundary(
        True, explicitly_allowed=explicitly_allowed, confirmatory_gate_passed=gate
    )
    return True


def policy_settings(alarm: Mapping[str, object]) -> dict[str, object]:
    persistence = alarm["persistence"]
    return {
        "required_above": int(persistence["required_above_threshold"]),
        "decisions_considered": int(persistence["decisions_considered"]),
        "cooldown_seconds": float(alarm["cooldown_seconds"]),
        "false_alert_budget": float(alarm["false_alert_budget_per_clean_mission"]),
    }


def with_risk_scores(
    rows: Sequence[Mapping[str, object]], scores: Sequence[float]
) -> list[dict[str, object]]:
    if len(rows) != len(scores):
        raise ValueError("score vector length must match row count")
    output = []
    for row, score in zip(rows, scores):
        value = float(score)
        if not 0.0 <= value <= 1.0:
            raise ValueError("risk_score must lie in [0, 1]")
        output.append({**row, "risk_score": value})
    return output
