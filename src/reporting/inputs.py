"""Resolve immutable artifacts and derive figure/table inputs without fabrication.

Prediction tables follow the frozen contract in docs/model-development-contracts.md
(``run_id, decision_index, decision_time, split, map_id, route_id, fault_family,
severity, seed, protected_test_used, eligibility, label, primary_event_class,
primary_event_time, model_id, raw_score, risk_score`` plus ``alarm, persistent`` once
the alarm policy has been applied). Every function here only reads what exists; a
missing artifact is reported as pending by the callers, never synthesised.
"""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from src.evaluation import (
    AlarmPolicy, apply_alarm_policy, evaluate_event_warnings, reliability_curve,
)


ARTIFACT_DEFAULTS: dict[str, str] = {
    "failure_events": "configs/failure_events.yaml",
    "alarm_policy": "configs/alarm_policy.yaml",
    "recovery_guards": "configs/recovery_guards.yaml",
    "recovery_costs": "configs/recovery_costs.yaml",
    "model_freeze": "configs/model_freeze.yaml",
    # Prediction inputs may be one alarm-applied table or a directory of them. When the
    # default path is absent, held-out and validation tables are discovered from the
    # input checksums recorded by reports/confirmatory/held_out_map.yaml.
    "predictions_validation": "reports/predictions/validation",
    "predictions_held_out": "reports/predictions/held_out",
    # scripts/run_unseen_family_folds.py: <work_root>/<family>/held_out_p{3,1}_alarmed.csv
    "predictions_unseen_family": "reports/unseen_family",
    # scripts/run_ablations.py: <predictions_root>/<ablation>.calibrated.csv (alarm applied here)
    "predictions_ablation": "reports/predictions/ablations",
    "held_out_report": "reports/confirmatory/held_out_map.yaml",
    "unseen_family_report": "reports/confirmatory/unseen_family.yaml",
    "paired_recovery": "reports/recovery/paired_recovery.yaml",
    "guard_verification": "reports/recovery/guard_verification.yaml",
    # scripts/measure_inference_latency.py: reports/latency/<name>.json (one model each)
    "latency_report": "reports/latency",
    "independent_rerun": "reports/reproduction/independent_rerun.yaml",
}
FAMILIES = (
    "camera_occlusion", "lidar_dropout", "wheel_slip", "localisation_perturbation",
    "dynamic_blockage", "planner_oscillation", "semantic_corruption",
)
PREDICTION_COLUMNS = (
    "run_id", "decision_index", "decision_time", "split", "map_id", "route_id",
    "fault_family", "severity", "seed", "protected_test_used", "eligibility", "label",
    "primary_event_class", "primary_event_time", "model_id", "raw_score", "risk_score",
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ArtifactRegistry:
    """Named artifact paths with overrides; records the sha256 of everything read."""

    root: Path
    overrides: dict[str, Path] = field(default_factory=dict)
    used: dict[str, dict[str, str]] = field(default_factory=dict)

    def path(self, name: str) -> Path:
        if name in self.overrides:
            return self.overrides[name]
        if name not in ARTIFACT_DEFAULTS:
            raise KeyError(f"unknown artifact name: {name}")
        default = self.root / ARTIFACT_DEFAULTS[name]
        # Released inputs: the final confirmatory report (held-out ablations evaluated,
        # written after the first report) and the frozen model's latency directory.
        if name == "held_out_report":
            final = default.with_name("held_out_map.final.yaml")
            if final.is_file():
                return final
        if name == "latency_report" and default.is_dir() and not any(default.glob("*.json")):
            freeze = self.root / "configs/model_freeze.yaml"
            if freeze.is_file():
                document = yaml.safe_load(freeze.read_text(encoding="utf-8")) or {}
                predictor = document.get("predictor") if isinstance(document.get("predictor"), dict) else {}
                checkpoint = str(predictor.get("checkpoint") or "")
                model_dir = str(predictor.get("model_dir") or (Path(checkpoint).parent if checkpoint else "") or "")
                tag = Path(model_dir).parent.name if model_dir else ""
                if tag and (default / tag).is_dir():
                    return default / tag
        return default

    def available(self, name: str) -> bool:
        path = self.path(name)
        if path.is_file():
            return True
        if path.is_dir() and self._table_files(name, path):
            return True
        # Held-out/validation tables may live under campaign sub-directories; the
        # confirmatory report names them, so discovery decides availability.
        try:
            return bool(self._discovered_tables(name))
        except (FileNotFoundError, OSError, yaml.YAMLError):
            return False

    def record(self, name: str) -> Path:
        path = self.path(name)
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
        self.used[name] = {"path": str(path), "sha256": sha256_path(path)}
        return path

    def _record_file(self, key: str, path: Path) -> None:
        self.used[key] = {"path": str(path), "sha256": sha256_path(path)}

    @staticmethod
    def _table_files(name: str, directory: Path) -> list[Path]:
        if name == "predictions_unseen_family":
            return sorted(p for p in directory.glob("*/held_out_*_alarmed.csv") if p.is_file())
        if name == "predictions_ablation":
            return sorted(p for p in directory.glob("*.calibrated.csv") if p.is_file())
        if name == "latency_report":
            return sorted(p for p in directory.glob("*.json") if p.is_file())
        return sorted(p for p in directory.glob("*.csv") if p.is_file() and ".raw." not in p.name)

    def _discovered_tables(self, name: str) -> list[Path]:
        """Held-out/validation tables named in the confirmatory report's input checksums."""
        if name not in {"predictions_held_out", "predictions_validation"}:
            return []
        report_path = self.path("held_out_report")
        if not report_path.is_file():
            return []
        document = yaml.safe_load(report_path.read_text(encoding="utf-8")) or {}
        wanted = "held_out_map_test" if name == "predictions_held_out" else "validation"
        found = []
        for key in (document.get("inputs_sha256") or {}):
            candidate = Path(key)
            if not candidate.is_absolute():
                candidate = self.root / candidate
            if candidate.suffix == ".csv" and candidate.is_file():
                with candidate.open(newline="", encoding="utf-8") as stream:
                    reader = csv.DictReader(stream)
                    first = next(reader, None)
                if first and first.get("split") == wanted and "alarm" in first:
                    found.append(candidate)
        return sorted(found)

    def prediction_rows(self, name: str) -> list[dict[str, str]]:
        """Load one table, every table of a directory, or the discovered tables."""
        path = self.path(name)
        if path.is_file():
            files = [path]
        elif path.is_dir():
            # A directory that holds only campaign sub-directories (reports/predictions/held_out/<campaign>/)
            # falls back to the tables the confirmatory report names.
            files = self._table_files(name, path) or self._discovered_tables(name)
        else:
            files = self._discovered_tables(name)
        if not files:
            raise FileNotFoundError(f"{name}: {path}")
        rows: list[dict[str, str]] = []
        for file in files:
            require_alarm = name != "predictions_ablation"
            table = load_prediction_table(file, require_alarm=require_alarm)
            if name == "predictions_unseen_family":
                fold = file.parent.name
                for row in table:
                    row.setdefault("fold_family", fold)
            if name == "predictions_ablation":
                ablation = file.name.split(".")[0]
                if "alarm" not in table[0]:
                    table = self._apply_frozen_alarm(table)
                for row in table:
                    row.setdefault("ablation", ablation)
            self._record_file(f"{name}:{file.name}" if len(files) > 1 or path.is_dir() else name, file)
            rows.extend(table)
        return rows

    def _apply_frozen_alarm(self, table: list[dict[str, str]]) -> list[dict[str, str]]:
        policy_document = self.yaml("alarm_policy")
        threshold = policy_document.get("threshold")
        if not isinstance(threshold, (int, float)):
            raise FileNotFoundError("alarm_policy: frozen threshold is not set; cannot alarm ablation tables")
        policy = alarm_policy_from_config(policy_document, float(threshold))
        output: list[dict[str, str]] = []
        for _run_id, rows in group_episodes(table).items():
            output.extend({**row, "alarm": str(row_out["alarm"]).lower(), "persistent": str(row_out["persistent"]).lower()}
                          for row, row_out in zip(rows, apply_alarm_policy(rows, policy)))
        return output

    def latency_records(self) -> list[dict[str, Any]]:
        path = self.path("latency_report")
        files = [path] if path.is_file() else (self._table_files("latency_report", path) if path.is_dir() else [])
        records = []
        for file in files:
            self._record_file(f"latency_report:{file.name}", file)
            document = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
            if "end_to_end" in document:
                end = document["end_to_end"]
                records.append({"model_id": document.get("model_id"), "median_ms": end.get("median_ms"),
                                "p95_ms": end.get("p95_ms"), "max_ms": end.get("max_ms")})
            else:
                for item in document.get("models") or document.get("results") or []:
                    records.append({key: item.get(key) for key in ("model_id", "median_ms", "p95_ms", "max_ms")})
        return records

    def yaml(self, name: str) -> dict[str, Any]:
        document = yaml.safe_load(self.record(name).read_text(encoding="utf-8"))
        return document if isinstance(document, dict) else {}

    def sources(self, names: Iterable[str]) -> dict[str, dict[str, str]]:
        return {name: dict(self.used[name]) for name in names if name in self.used}

    @staticmethod
    def parse_overrides(items: Sequence[str]) -> dict[str, Path]:
        overrides = {}
        for item in items:
            name, _sep, value = item.partition("=")
            if not _sep or not name or not value:
                raise ValueError(f"override must look like name=path: {item}")
            overrides[name] = Path(value)
        return overrides


def load_prediction_table(path: Path, *, require_alarm: bool = True) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames or []
        missing = [name for name in PREDICTION_COLUMNS if name not in names]
        if require_alarm and "alarm" not in names:
            missing.append("alarm")
        if missing:
            raise ValueError(f"{path}: prediction table lacks columns {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: prediction table is empty")
    return rows


def split_models(rows: Sequence[Mapping[str, str]]) -> dict[str, list[Mapping[str, str]]]:
    by_model: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        by_model[str(row["model_id"])].append(row)
    return dict(sorted(by_model.items()))


def group_episodes(rows: Iterable[Mapping[str, str]]) -> dict[str, list[Mapping[str, str]]]:
    episodes: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        episodes[str(row["run_id"])].append(row)
    for run_id, items in episodes.items():
        items.sort(key=lambda row: float(row["decision_time"]))
    return dict(episodes)


def _is_clean(rows: Sequence[Mapping[str, str]]) -> bool:
    return all(str(row.get("fault_family", "none")) in {"none", ""} for row in rows)


TIMEOUT_EVENT_CLASS = "mission_timeout"


def split_timeout_episodes(episodes: Mapping[str, Sequence[Mapping[str, str]]]) -> tuple[dict, dict]:
    """Mission timeouts are analysed separately from the primary event set, exactly as
    scripts/evaluate_confirmatory.py does (protocol: timeouts analysed separately)."""
    primary, timeouts = {}, {}
    for run_id, items in episodes.items():
        if any(str(row.get("primary_event_class") or "") == TIMEOUT_EVENT_CLASS for row in items):
            timeouts[run_id] = items
        else:
            primary[run_id] = items
    return primary, timeouts


def predictor_metrics(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Event-level metrics for one model's alarmed rows plus per-family recall.

    Mission-timeout episodes are excluded from the primary denominators and reported
    under ``timeouts`` (count and recall), matching the confirmatory evaluator."""
    episodes, timeout_episodes = split_timeout_episodes(group_episodes(rows))
    timeout_metrics = evaluate_event_warnings(timeout_episodes) if timeout_episodes else None
    metrics = evaluate_event_warnings(episodes)
    clean_ids = {run_id for run_id, items in episodes.items() if _is_clean(items)}
    clean_false = sum(
        item["false_alerts"] for item in metrics["per_episode"] if item["run_id"] in clean_ids
    )
    family_of = {run_id: str(items[0].get("fault_family", "none")) for run_id, items in episodes.items()}
    by_family: dict[str, dict[str, Any]] = {}
    for item in metrics["per_episode"]:
        family = family_of[item["run_id"]]
        entry = by_family.setdefault(family, {"episodes": 0, "events": 0, "detected": 0,
                                              "false_alerts": 0, "lead_seconds": []})
        entry["episodes"] += 1
        entry["false_alerts"] += item["false_alerts"]
        if item["has_event"]:
            entry["events"] += 1
            entry["detected"] += int(item["detected"])
            if item["first_useful_lead_seconds"] is not None:
                entry["lead_seconds"].append(item["first_useful_lead_seconds"])
    for entry in by_family.values():
        entry["event_recall"] = entry["detected"] / entry["events"] if entry["events"] else None
        entry["false_alerts_per_mission"] = entry["false_alerts"] / entry["episodes"]
    return {
        **{key: value for key, value in metrics.items() if key != "per_episode"},
        "per_episode": metrics["per_episode"],
        "clean_mission_count": len(clean_ids),
        "false_alerts_per_clean_mission": clean_false / len(clean_ids) if clean_ids else None,
        "by_family": dict(sorted(by_family.items())),
        "timeouts": {
            "analysed_separately": True,
            "timeout_episode_count": len(timeout_episodes),
            "timeout_recall": timeout_metrics["event_recall"] if timeout_metrics else None,
        },
    }


def alarm_policy_from_config(document: Mapping[str, Any], threshold: float) -> AlarmPolicy:
    persistence = document.get("persistence", {})
    return AlarmPolicy(
        threshold=threshold,
        required_above=int(persistence.get("required_above_threshold", 2)),
        decisions_considered=int(persistence.get("decisions_considered", 3)),
        cooldown_seconds=float(document.get("cooldown_seconds", 10.0)),
    )


def recall_false_alert_curve(
    rows: Sequence[Mapping[str, str]], policy_config: Mapping[str, Any], thresholds: Sequence[float],
) -> list[dict[str, float | None]]:
    """Recall and clean-mission false-alert burden when the frozen policy sweeps tau."""
    episodes, _timeouts = split_timeout_episodes(group_episodes(rows))
    clean_ids = {run_id for run_id, items in episodes.items() if _is_clean(items)}
    curve = []
    for threshold in thresholds:
        policy = alarm_policy_from_config(policy_config, float(threshold))
        predicted = {run_id: apply_alarm_policy(items, policy) for run_id, items in episodes.items()}
        metrics = evaluate_event_warnings(predicted)
        clean_false = sum(
            item["false_alerts"] for item in metrics["per_episode"] if item["run_id"] in clean_ids
        )
        curve.append({
            "threshold": float(threshold),
            "event_recall": metrics["event_recall"],
            "false_alerts_per_mission": metrics["false_alerts_per_mission"],
            "false_alerts_per_clean_mission": clean_false / len(clean_ids) if clean_ids else None,
        })
    return curve


def lead_time_curve_by_family(
    rows: Sequence[Mapping[str, str]], grid: Sequence[float]
) -> dict[str, dict[str, Any]]:
    """Proportion of events warned at least ``t`` seconds before failure, per family.

    Undetected events stay in the denominator at every ``t``."""
    metrics = predictor_metrics(rows)
    episodes = group_episodes(rows)
    family_of = {run_id: str(items[0].get("fault_family", "none")) for run_id, items in episodes.items()}
    leads: dict[str, list[float | None]] = defaultdict(list)
    for item in metrics["per_episode"]:
        if item["has_event"]:
            leads[family_of[item["run_id"]]].append(item["first_useful_lead_seconds"])
    curves = {}
    for family, values in sorted(leads.items()):
        curves[family] = {
            "event_count": len(values),
            "undetected": sum(1 for value in values if value is None),
            "grid_seconds": [float(t) for t in grid],
            "warned_fraction": [
                sum(1 for value in values if value is not None and value >= t) / len(values)
                for t in grid
            ],
        }
    return curves


def reliability_by_score(rows: Sequence[Mapping[str, str]], score_field: str, bins: int = 10) -> dict[str, Any]:
    substituted = [{**row, "risk_score": row[score_field]} for row in rows]
    return reliability_curve(substituted, bins=bins)


def write_sidecar(path: Path, payload: Mapping[str, Any]) -> None:
    sidecar = path.with_suffix(path.suffix + ".json")
    if sidecar.exists():
        raise FileExistsError(f"refusing to overwrite sidecar: {sidecar}")
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def publish_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite generated artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
