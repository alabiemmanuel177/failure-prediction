#!/usr/bin/env python3
"""Join a paired recovery campaign's episode summaries into the outcome CSV.

Consumed by ``scripts/analyze_paired_recovery.py`` (paired_recovery_v1) and by
``scripts/build_recovery_cost_table.py`` (recovery_pilot_v1). One row per valid
episode whose summary carries ``system.recovery_policy_id`` and whose
``identity.campaign_id`` matches the manifest:

  run_id, map_id, route_id, seed, fault_family, severity, policy_id, mission_complete,
  collision, guard_violation, guard_rejected, added_time_seconds, added_path_length_m,
  intervention_count, recovery_action, action_regret_vs_oracle, oracle_action, ...

``added_*`` are relative to the R0 episode of the same map/route/seed/fault pair
(0 for R0 itself; the signed differences are kept in ``time_difference_seconds`` and
``path_difference_m``, the clamped non-negative overheads feed the frozen cost).
``action_regret_vs_oracle`` = observed cost minus the lowest observed cost among the
policies of the same pair; ``oracle_action`` is that policy's executed action.
``unnecessary_intervention`` = an intervention on an episode whose R0 pair completed
its mission without collision. Rows without an R0 pair are excluded and listed in the
``<output>.provenance.json`` sidecar. Never overwrites.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments import declared_replacements, ledger_invalid_episode_keys  # noqa: E402
from src.dataset_inventory import publish_new_bytes, sha256_file  # noqa: E402
from src.models.common import sha256_bytes  # noqa: E402
from src.evaluation.recovery_metrics import PAIR_FIELDS  # noqa: E402
from src.protected_data import enforce_protected_boundary  # noqa: E402
from src.recovery.costs import (  # noqa: E402
    DEFAULT_COST_CONFIG, RecoveryOutcome, cost_breakdown, load_cost_weights, observed_cost,
)
from src.recovery.plumbing import DEFAULT_POLICY_ID, validate_policy_id  # noqa: E402


COLUMNS = (
    "run_id", *PAIR_FIELDS, "policy_id", "mission_complete", "collision", "guard_violation",
    "guard_rejected", "added_time_seconds", "added_path_length_m", "intervention_count",
    "recovery_action", "action_regret_vs_oracle", "oracle_action",
    "episode_key", "pair_key", "split", "protected_test_used", "terminal_state",
    "mission_abort", "failed_recovery", "unnecessary_intervention", "duration_s",
    "path_length_m", "time_difference_seconds", "path_difference_m", "warnings_issued",
    "first_warning_time", "forced_action", "engineering_smoke", "observed_cost",
    "cost_collision", "cost_mission_abort", "cost_failed_recovery", "cost_excessive_delay",
    "cost_path_overhead", "cost_unnecessary_intervention",
)


def _bool(value: object) -> str:
    return "true" if bool(value) else "false"


def _pair_key(summary: dict[str, Any]) -> tuple[str, ...]:
    environment = summary["environment"]
    label = summary["label_only"]
    return (
        str(environment["map_id"]), str(environment["route_id"]), str(environment["seed"]),
        str(label["fault_family"]), str(label["severity"]),
    )


def episode_record(summary: dict[str, Any]) -> dict[str, Any]:
    """Per-episode fields that do not depend on the pair."""
    system = summary.get("system") or {}
    recovery = system.get("recovery") or {}
    outcome = summary["outcome"]
    actions = [item for item in recovery.get("actions", []) if item.get("execution_performed")]
    executed = [str(item["action"]) for item in actions]
    return {
        "run_id": summary["identity"]["run_id"],
        "episode_key": summary["identity"].get("episode_key"),
        "policy_id": validate_policy_id(str(system["recovery_policy_id"])),
        "split": summary["environment"].get("split"),
        "protected_test_used": summary["environment"].get("protected_test_used"),
        "terminal_state": outcome["terminal_state"],
        "mission_complete": bool(outcome["success"]),
        "collision": bool(outcome["collision"]),
        "duration_s": float(outcome["duration_s"]),
        "path_length_m": float(outcome["path_length_m"]),
        "guard_violation": bool(recovery.get("guard_violation", False)),
        "guard_rejected": any(bool(item.get("guard_rejected")) for item in recovery.get("actions", [])),
        "intervention_count": len(actions),
        "recovery_action": executed[0] if executed else "none",
        "failed_recovery": any(item.get("failed") for item in recovery.get("execution_results", [])),
        "warnings_issued": int(recovery.get("warnings_issued", 0) or 0),
        "first_warning_time": recovery.get("first_warning_time"),
        "forced_action": system.get("recovery_forced_action"),
        "engineering_smoke": bool(recovery.get("engineering_smoke", False)),
    }


def build_outcome_rows(
    summaries: list[dict[str, Any]], *, campaign_id: str, expected_policies: list[str],
    weights, allow_protected: bool = False,
    ledger_invalid_keys: set[str] | frozenset[str] = frozenset(),
    replacements: dict[str, tuple[str, str]] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """``ledger_invalid_keys``: original episode keys whose recorded attempt failed
    (episode or artifact validation) in the campaign ledger; they are never rows.
    ``replacements``: original key -> (replacement campaign id, replacement episode key)
    from the declared child manifests; a replacement summary stands in for its cell."""
    replacements = replacements or {}
    replacement_keys = {(cid, key) for cid, key in replacements.values()}
    report: dict[str, Any] = {
        "summaries_seen": len(summaries), "excluded": [], "pairs": 0,
        "pairs_without_reference": [], "duplicate_cells": [],
        "ledger_invalid_originals_excluded": 0, "replacements_used": 0,
    }
    by_pair: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {}
    for summary in summaries:
        identity = summary.get("identity", {})
        is_replacement = (identity.get("campaign_id"), identity.get("episode_key")) in replacement_keys
        if identity.get("campaign_id") != campaign_id and not is_replacement:
            continue
        if identity.get("episode_key") in ledger_invalid_keys:
            report["ledger_invalid_originals_excluded"] += 1
            report["excluded"].append({"run_id": identity.get("run_id"),
                                       "reason": "ledger records a failed attempt (artifact validation)"})
            continue
        if is_replacement:
            report["replacements_used"] += 1
        system = summary.get("system") or {}
        if "recovery_policy_id" not in system:
            report["excluded"].append({"run_id": identity.get("run_id"), "reason": "no recovery_policy_id"})
            continue
        if summary["outcome"]["terminal_state"] == "invalid":
            report["excluded"].append({"run_id": identity.get("run_id"), "reason": "invalid episode"})
            continue
        protected = summary["environment"].get("protected_test_used")
        enforce_protected_boundary(protected, explicitly_allowed=allow_protected,
                                   confirmatory_gate_passed=allow_protected)
        record = episode_record(summary)
        if record["policy_id"] not in expected_policies:
            report["excluded"].append({"run_id": record["run_id"],
                                       "reason": f"policy {record['policy_id']} not in manifest"})
            continue
        key = _pair_key(summary)
        cell = by_pair.setdefault(key, {})
        if record["policy_id"] in cell:
            report["duplicate_cells"].append([*key, record["policy_id"]])
            continue
        record["pair_key"] = "-".join(key)
        cell[record["policy_id"]] = record
    rows: list[dict[str, str]] = []
    for key, cell in sorted(by_pair.items()):
        reference = cell.get(DEFAULT_POLICY_ID)
        if reference is None:
            report["pairs_without_reference"].append(list(key))
            continue
        report["pairs"] += 1
        costs: dict[str, tuple[float, dict[str, float], dict[str, Any]]] = {}
        for policy_id, record in cell.items():
            time_difference = record["duration_s"] - reference["duration_s"]
            path_difference = record["path_length_m"] - reference["path_length_m"]
            added_time = 0.0 if policy_id == DEFAULT_POLICY_ID else max(0.0, time_difference)
            added_path = 0.0 if policy_id == DEFAULT_POLICY_ID else max(0.0, path_difference)
            unnecessary = bool(
                record["intervention_count"] > 0 and reference["mission_complete"]
                and not reference["collision"]
            )
            outcome = RecoveryOutcome(
                collision=record["collision"], mission_abort=not record["mission_complete"],
                failed_recovery=record["failed_recovery"], added_time_seconds=added_time,
                added_path_length_m=added_path, unnecessary_intervention=unnecessary,
            )
            breakdown = cost_breakdown(outcome, weights)
            costs[policy_id] = (observed_cost(outcome, weights), breakdown, {
                **record, "added_time_seconds": added_time, "added_path_length_m": added_path,
                "time_difference_seconds": time_difference, "path_difference_m": path_difference,
                "unnecessary_intervention": unnecessary,
            })
        oracle_policy = min(costs.items(), key=lambda item: (item[1][0], item[0]))[0]
        oracle_cost = costs[oracle_policy][0]
        oracle_action = costs[oracle_policy][2]["recovery_action"]
        for policy_id in sorted(costs):
            cost, breakdown, record = costs[policy_id]
            rows.append({
                "run_id": record["run_id"],
                "map_id": key[0], "route_id": key[1], "seed": key[2],
                "fault_family": key[3], "severity": key[4],
                "policy_id": policy_id,
                "mission_complete": _bool(record["mission_complete"]),
                "collision": _bool(record["collision"]),
                "guard_violation": _bool(record["guard_violation"]),
                "guard_rejected": _bool(record["guard_rejected"]),
                "added_time_seconds": repr(float(record["added_time_seconds"])),
                "added_path_length_m": repr(float(record["added_path_length_m"])),
                "intervention_count": str(int(record["intervention_count"])),
                "recovery_action": record["recovery_action"],
                "action_regret_vs_oracle": repr(float(cost - oracle_cost)),
                "oracle_action": oracle_action,
                "episode_key": record["episode_key"] or "",
                "pair_key": record["pair_key"],
                "split": record["split"] or "",
                "protected_test_used": _bool(record["protected_test_used"]),
                "terminal_state": record["terminal_state"],
                "mission_abort": _bool(not record["mission_complete"]),
                "failed_recovery": _bool(record["failed_recovery"]),
                "unnecessary_intervention": _bool(record["unnecessary_intervention"]),
                "duration_s": repr(float(record["duration_s"])),
                "path_length_m": repr(float(record["path_length_m"])),
                "time_difference_seconds": repr(float(record["time_difference_seconds"])),
                "path_difference_m": repr(float(record["path_difference_m"])),
                "warnings_issued": str(record["warnings_issued"]),
                "first_warning_time": (
                    "" if record["first_warning_time"] is None else repr(float(record["first_warning_time"]))
                ),
                "forced_action": record["forced_action"] or "",
                "engineering_smoke": _bool(record["engineering_smoke"]),
                "observed_cost": repr(float(cost)),
                **{f"cost_{term}": repr(float(value)) for term, value in breakdown.items()},
            })
    report["rows"] = len(rows)
    report["engineering_smoke_rows"] = sum(1 for row in rows if row["engineering_smoke"] == "true")
    return rows, report


def rows_to_csv_bytes(rows: list[dict[str, str]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def load_summaries(directory: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    summaries = []
    digests = {}
    for path in sorted(directory.glob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(document, dict) and "identity" in document:
            summaries.append(document)
            digests[str(document["identity"].get("run_id"))] = sha256_file(path)
    return summaries, digests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summaries", type=Path, default=ROOT / "data/raw/summaries")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cost-config", type=Path, default=DEFAULT_COST_CONFIG)
    parser.add_argument("--allow-protected-after-freeze", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable table: {args.output}")
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    policies = [validate_policy_id(str(item)) for item in manifest.get("recovery_policies") or []]
    if not policies:
        raise SystemExit("manifest declares no recovery_policies")
    protected = manifest.get("protected_test_used") is True
    if protected and not args.allow_protected_after_freeze:
        raise SystemExit("protected campaign: pass --allow-protected-after-freeze after the freeze")
    weights = load_cost_weights(args.cost_config)
    summaries, digests = load_summaries(args.summaries)
    ledger_invalid = ledger_invalid_episode_keys(ROOT, manifest)
    replacements = {
        str(spec["original_episode_key"]): (str(spec["replacement_campaign_id"]), str(spec["replacement_episode_key"]))
        for spec in declared_replacements(ROOT, manifest)
    }
    try:
        rows, report = build_outcome_rows(
            summaries, campaign_id=str(manifest["campaign_id"]), expected_policies=policies,
            weights=weights, allow_protected=args.allow_protected_after_freeze,
            ledger_invalid_keys=ledger_invalid, replacements=replacements,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    payload = rows_to_csv_bytes(rows)
    publish_new_bytes(args.output, payload)
    provenance = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "manifest": str(args.manifest), "manifest_sha256": sha256_file(args.manifest),
        "cost_config": str(args.cost_config), "cost_config_sha256": sha256_file(args.cost_config),
        "summaries_dir": str(args.summaries),
        "summary_sha256_by_run": {row["run_id"]: digests.get(row["run_id"]) for row in rows},
        "output": str(args.output), "output_sha256": sha256_bytes(payload),
        "protected_test_used": protected,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **report,
    }
    publish_new_bytes(args.output.with_name(args.output.name + ".provenance.json"),
                      (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({"output": str(args.output), "rows": len(rows), "pairs": report["pairs"],
                      "pairs_without_reference": len(report["pairs_without_reference"]),
                      "excluded": len(report["excluded"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
