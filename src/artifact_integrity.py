"""Metadata-only checks for lost raw artifact payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PRE_GOAL_TERMINATION_REASON = "pre_goal_process_termination"
PAYLOAD_LOSS_REASON = "summary_and_bag_payload_loss_after_validation"


def _document(path: Path) -> dict[str, Any] | None:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return None
    return value if isinstance(value, dict) else None


def audit_raw_payloads(root: Path) -> dict[str, Any]:
    raw = root / "data/raw"
    summaries = sorted((raw / "summaries").glob("*.yaml"))
    mcaps = sorted((raw / "bags").glob("*/*.mcap"))
    metadata = sorted((raw / "bags").glob("*/metadata.yaml"))

    zero_paths = [path for path in [*summaries, *mcaps, *metadata]
                  if path.is_file() and path.stat().st_size == 0]
    zero_runs = set()
    for path in zero_paths:
        zero_runs.add(path.stem if path.parent.name == "summaries" else path.parent.name)

    declared: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "data/manifests").glob("*.yaml")):
        document = _document(path) or {}
        for item in document.get("infrastructure_replacements", []):
            reason = item.get("invalid_reason")
            if reason in (PAYLOAD_LOSS_REASON, PRE_GOAL_TERMINATION_REASON):
                declared[str(item["original_run_id"])] = {
                    "manifest": str(path.relative_to(root)),
                    "episode_key": item["original_episode_key"],
                    "invalid_reason": reason,
                    "replacement_campaign_id": item["replacement_campaign_id"],
                    "replacement_episode_key": item["replacement_episode_key"],
                }

    # Post-validation payload loss leaves all three files present but empty. A
    # pre-goal process termination leaves only an empty MCAP: the recorder had opened
    # its file, but no summary, event sidecar or bag metadata was ever written.
    required_zero_names = {
        PAYLOAD_LOSS_REASON: {"summary", "mcap", "metadata"},
        PRE_GOAL_TERMINATION_REASON: {"mcap", "no_summary_file", "no_metadata_file"},
    }
    evidence: dict[str, list[str]] = {}
    for run_id, item in declared.items():
        names = []
        summary_path = raw / "summaries" / f"{run_id}.yaml"
        if summary_path.is_file() and summary_path.stat().st_size == 0:
            names.append("summary")
        if not summary_path.exists():
            names.append("no_summary_file")
        run_mcaps = list((raw / "bags" / run_id).glob("*.mcap"))
        if len(run_mcaps) == 1 and run_mcaps[0].stat().st_size == 0:
            names.append("mcap")
        bag_metadata = raw / "bags" / run_id / "metadata.yaml"
        if bag_metadata.is_file() and bag_metadata.stat().st_size == 0:
            names.append("metadata")
        if not bag_metadata.exists():
            names.append("no_metadata_file")
        evidence[run_id] = sorted(names)

    unexpected = sorted(zero_runs - set(declared))
    incomplete_declared = sorted(
        run_id for run_id, names in evidence.items()
        if set(names) != required_zero_names[declared[run_id]["invalid_reason"]]
    )
    return {
        "schema_version": 1,
        "passed": not unexpected and not incomplete_declared,
        "scope": "research2_preprotected_raw_artifact_file_metadata",
        "protected_outcomes_consulted": False,
        "counts": {
            "summary_paths": len(summaries),
            "mcap_paths": len(mcaps),
            "bag_metadata_paths": len(metadata),
            "zero_length_paths": len(zero_paths),
            "zero_length_runs": len(zero_runs),
            "declared_payload_loss_runs": len(declared),
            "unexpected_zero_length_runs": len(unexpected),
        },
        "declared_payload_loss": declared,
        "declared_evidence": evidence,
        "unexpected_zero_length_runs": unexpected,
        "incomplete_declared_evidence": incomplete_declared,
    }
