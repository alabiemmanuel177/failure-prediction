#!/usr/bin/env python3
"""Fill the model and dataset cards from immutable artifacts and flip ``Status: final``.

The cards are never edited unless ``scripts/audit_project_completion.py`` passes when
only the four package-finalisation findings (cards, manuscript, release record) are
tolerated. ``--dry-run`` prints the finalised text without writing.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.release import blocking_findings, git_head, run_completion_audit, sha256_file  # noqa: E402


TOLERATED = ("model_card", "dataset_card", "manuscript", "release")


def _yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def collect(root: Path) -> dict[str, Any]:
    freeze = _yaml(root / "configs/model_freeze.yaml")
    predictor = freeze.get("predictor", {})
    checkpoint = root / str(predictor.get("checkpoint", ""))
    training = {}
    if checkpoint.suffix and checkpoint.parent.is_dir():
        record = checkpoint.parent / "training_record.json"
        if record.exists():
            training = json.loads(record.read_text(encoding="utf-8"))
    return {
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": git_head(root),
        "freeze": freeze,
        "predictor": predictor,
        "training": training,
        "alarm": _yaml(root / "configs/alarm_policy.yaml"),
        "summary": _csv(root / "reports/tables/tab01_predictor_summary.csv"),
        "unseen": _csv(root / "reports/tables/tab03_unseen_family.csv"),
        "ablations": _csv(root / "reports/tables/tab04_ablations.csv"),
        "latency": _yaml(root / "reports/latency/inference_latency.yaml"),
        "recovery": _yaml(root / "reports/recovery/paired_recovery.yaml"),
        "rerun": _yaml(root / "reports/reproduction/independent_rerun.yaml"),
        "datasets": {
            name: _yaml(root / f"data/manifests/{name}.dataset.yaml")
            for name in ("balanced_pilot_v1", "balanced_validation_v1", "targeted_development_v1")
        },
        "held_out": _yaml(root / "reports/confirmatory/held_out_map.yaml"),
    }


def _rows(rows: list[dict[str, str]], columns: list[str]) -> list[str]:
    if not rows:
        return ["- pending: table not generated"]
    header = "| " + " | ".join(columns) + " |\n|" + "|".join("---" for _ in columns) + "|"
    body = "\n".join("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |" for row in rows)
    return [header, body]


def model_card_section(facts: dict[str, Any]) -> str:
    predictor, training, alarm = facts["predictor"], facts["training"], facts["alarm"]
    lines = [
        "", f"## Final record (generated {facts['utc']})", "",
        f"- Release commit: `{facts['commit']}`.",
        f"- Predictor: `{predictor.get('model_id', 'pending')}`; checkpoint `{predictor.get('checkpoint', 'pending')}` "
        f"(sha256 `{predictor.get('checkpoint_sha256', 'pending')}`).",
        f"- Normalisation bundle sha256 `{predictor.get('normalization_bundle_sha256', 'pending')}`; "
        f"training config sha256 `{predictor.get('training_config_sha256', 'pending')}`.",
        f"- Training seed {training.get('seed', 'pending')}; parameters {training.get('parameter_count', 'pending')}; "
        f"epochs {training.get('epochs_run', training.get('epochs', 'pending'))}; early stopping on validation AUPRC.",
        f"- Calibration `{facts['freeze'].get('calibration', {}).get('calibration_id', 'pending')}`; "
        f"frozen threshold {alarm.get('threshold', 'pending')} with "
        f"{alarm.get('persistence', {}).get('required_above_threshold', 2)}-of-"
        f"{alarm.get('persistence', {}).get('decisions_considered', 3)} persistence and "
        f"{alarm.get('cooldown_seconds', 10)} s cooldown at a budget of "
        f"{alarm.get('false_alert_budget_per_clean_mission', 0.1)} false alerts per clean mission.",
        "", "### Event-level evaluation (frozen threshold)", "",
        *_rows(facts["summary"], ["split", "model_id", "episodes", "events", "event_recall",
                                  "false_alerts_per_clean_mission", "median_useful_lead_seconds",
                                  "brier_calibrated", "ece_calibrated"]),
        "", "### Unseen-family folds", "",
        *_rows(facts["unseen"], ["model_id", "excluded_family", "events", "event_recall", "false_alerts_per_mission"]),
        "", "### Ablations", "",
        *_rows(facts["ablations"], ["ablation", "event_recall", "false_alerts_per_clean_mission"]),
    ]
    latency = facts["latency"].get("models") or facts["latency"].get("results")
    lines += ["", "### Latency", "", f"- {json.dumps(latency) if latency else 'pending: latency report not generated'}"]
    recovery = facts["recovery"]
    h6 = recovery.get("h6", {})
    lines += ["", "### Paired recovery", "",
              f"- Complete: {recovery.get('complete', 'pending')}; H6 supported: {h6.get('supported', 'pending')}; "
              f"completion difference R3−R0 {h6.get('completion_rate_difference', 'pending')} "
              f"(95% interval {h6.get('completion_interval_95', 'pending')}); collision difference "
              f"{h6.get('collision_rate_difference', 'pending')}; estimator {h6.get('estimator', 'pending')}.",
              f"- Guard rejections/violations by policy: {json.dumps(recovery.get('guard', {}))}."]
    return "\n".join(lines) + "\n"


def dataset_card_section(facts: dict[str, Any]) -> str:
    lines = ["", f"## Final record (generated {facts['utc']})", "", f"- Release commit: `{facts['commit']}`."]
    for name, document in facts["datasets"].items():
        inventory = document.get("episode_inventory", {})
        lines.append(f"- `{document.get('dataset_id', name + ' (pending)')}`: {inventory.get('rows', 'pending')} episodes; "
                     f"inventory sha256 `{inventory.get('sha256', 'pending')}`.")
    held = facts["held_out"]
    lines.append(f"- Held-out map dataset: `{held.get('dataset_id', 'pending')}`; result complete: {held.get('complete', 'pending')}.")
    rerun = facts["rerun"]
    lines.append(f"- Independent clean rerun passed: {rerun.get('passed', 'pending')} (commit `{rerun.get('git_commit', 'pending')}`).")
    lines.append("- Software: Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic, Nav2; exact package versions are "
                 "recorded per episode in the immutable summaries (`provenance` block).")
    return "\n".join(lines) + "\n"


def finalize_text(text: str, section: str, facts: dict[str, Any]) -> str:
    if "Status: final" in text:
        raise ValueError("card is already final")
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("Status:"):
            end = index
            while end + 1 < len(lines) and lines[end + 1].strip():
                end += 1
            lines[index:end + 1] = [f"Status: final ({facts['utc']}, release commit {facts['commit'][:12]})."]
            break
    else:
        raise ValueError("card has no Status line")
    return "\n".join(lines).rstrip("\n") + "\n" + section


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.root
    facts = collect(root)
    targets = {
        root / "docs/model-card.md": model_card_section(facts),
        root / "docs/dataset-card.md": dataset_card_section(facts),
    }
    rendered = {path: finalize_text(path.read_text(encoding="utf-8"), section, facts) for path, section in targets.items()}
    if args.dry_run:
        for path, text in rendered.items():
            print(f"===== {path} =====\n{text}")
        return 0
    blocking = blocking_findings(run_completion_audit(root), TOLERATED)
    if blocking:
        raise SystemExit("cards stay non-final; completion audit findings:\n- " + "\n- ".join(blocking))
    for path, text in rendered.items():
        before = sha256_file(path)
        path.write_text(text, encoding="utf-8")
        print(f"finalised {path} (previous sha256 {before})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
