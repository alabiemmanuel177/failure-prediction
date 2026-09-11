#!/usr/bin/env python3
"""Turn recovery-pilot outcomes into the replay cost table the R3 selector trainer reads.

Inputs: the pilot manifest, the outcome CSV from ``scripts/build_recovery_outcome_table.py``
and the episode summaries (for each episode's recovery-manager sidecar, which holds the
post-safe-stop warning state the frozen guard evaluated). Output: one row per
(warning x candidate action) with exactly the columns
``src.recovery.selector_training.REQUIRED_COLUMNS`` plus identity fields:

  split, map_id, route_id, seed, fault_family, severity, warning_id, risk_score,
  diagnosed_signal_group, stopped, stop_allowed, localisation_poor,
  planning_stale_or_blocked, rear_clearance_m, rotation_clearance_m,
  immediate_collision_risk, obstruction_may_be_transient, relocalisation_available,
  repeated_recovery_count, candidate_action, guard_eligible, observed_cost

``warning_id`` is the pair key so the six forced-action episodes of one cell are the
candidates of one warning. ``guard_eligible`` is recomputed from the recorded state with
``src.recovery.guards.eligible_actions`` and must agree with the manager's decision;
a guard-rejected forced action yields its row (ineligible, never a training target)
plus a ``controlled_stop`` row for the action that actually ran. ``observed_cost`` is
the frozen ``src.recovery.costs`` cost from the outcome table. Episodes with no alarm
produce no row and are counted in the provenance sidecar. ``split`` is the
validation split by design of PA-2026-09-03-04.
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

from src.dataset_inventory import publish_new_bytes, sha256_file  # noqa: E402
from src.models.common import sha256_bytes  # noqa: E402
from src.recovery import GuardConfig  # noqa: E402
from src.recovery.guards import eligible_actions  # noqa: E402
from src.recovery.plumbing import DEFAULT_POLICY_ID, forced_action, load_sidecar  # noqa: E402
from src.recovery.selector_training import (  # noqa: E402
    REQUIRED_COLUMNS, STATE_BOOLEANS, STATE_CLEARANCES, state_from_row,
)


COLUMNS = (
    *REQUIRED_COLUMNS[:1], "map_id", "route_id", "seed", "fault_family", "severity",
    *REQUIRED_COLUMNS[1:], "run_id", "policy_id", "executed_action", "guard_rejected",
    "mission_complete", "collision", "engineering_smoke",
)


def guard_config_from(path: Path) -> GuardConfig:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return GuardConfig(
        minimum_rear_clearance_m=float(document["minimum_rear_clearance_m"]),
        minimum_rotation_clearance_m=float(document["minimum_rotation_clearance_m"]),
        maximum_repeated_recoveries=int(document["maximum_repeated_recoveries"]),
    )


def _text(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return repr(float(value)) if isinstance(value, float) else str(value)


def first_decision(manager: dict[str, Any] | None) -> dict[str, Any] | None:
    for decision in (manager or {}).get("decisions") or []:
        if decision.get("guard_results"):
            return decision
    return None


def cost_rows(
    outcomes: list[dict[str, str]], summaries_by_run: dict[str, dict[str, Any]],
    sidecars_by_run: dict[str, dict[str, Any] | None], guard_config: GuardConfig,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    report: dict[str, Any] = {"outcome_rows": len(outcomes), "no_alarm": [], "no_sidecar": [],
                              "guard_disagreements": [], "reference_rows_skipped": 0}
    rows: list[dict[str, str]] = []
    for outcome in outcomes:
        policy_id = outcome["policy_id"]
        if policy_id == DEFAULT_POLICY_ID:
            report["reference_rows_skipped"] += 1
            continue
        forced = forced_action(policy_id)
        if forced is None:
            continue
        run_id = outcome["run_id"]
        summary = summaries_by_run.get(run_id)
        manager = sidecars_by_run.get(run_id)
        if summary is None or manager is None:
            report["no_sidecar"].append(run_id)
            continue
        decision = first_decision(manager)
        if decision is None:
            report["no_alarm"].append(run_id)
            continue
        state_record = dict(decision["state"])
        state = state_from_row({**state_record, "relocalisation_available": state_record.get(
            "relocalisation_available", False)})
        guards = eligible_actions(state, guard_config)
        recorded = decision["guard_results"]
        for action, (eligible, _reason) in guards.items():
            if action in recorded and bool(recorded[action]["eligible"]) != eligible:
                report["guard_disagreements"].append({"run_id": run_id, "action": action})
        executed = str(decision.get("recommended_action"))
        base = {
            "split": str(summary["environment"].get("split", "")),
            "map_id": outcome["map_id"], "route_id": outcome["route_id"], "seed": outcome["seed"],
            "fault_family": outcome["fault_family"], "severity": outcome["severity"],
            "warning_id": outcome["pair_key"],
            "risk_score": _text(float(decision["risk_score"])),
            "diagnosed_signal_group": str(decision.get("diagnosed_signal_group", "unknown")),
            **{name: _text(bool(state_record[name])) for name in STATE_BOOLEANS},
            **{name: _text(state_record.get(name)) for name in STATE_CLEARANCES},
            "repeated_recovery_count": str(int(state_record["repeated_recovery_count"])),
            "observed_cost": outcome["observed_cost"],
            "run_id": run_id, "policy_id": policy_id, "executed_action": executed,
            "guard_rejected": _text(bool(decision.get("guard_rejected"))),
            "mission_complete": outcome["mission_complete"], "collision": outcome["collision"],
            "engineering_smoke": outcome.get("engineering_smoke", "false"),
        }
        rows.append({**base, "candidate_action": forced, "guard_eligible": _text(guards[forced][0])})
        if decision.get("guard_rejected") and executed != forced and executed in guards:
            rows.append({**base, "candidate_action": executed,
                         "guard_eligible": _text(guards[executed][0])})
    report["rows"] = len(rows)
    report["guard_eligible_rows"] = sum(1 for row in rows if row["guard_eligible"] == "true")
    return rows, report


def rows_to_csv_bytes(rows: list[dict[str, str]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summaries", type=Path, default=ROOT / "data/raw/summaries")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--guard-config", type=Path, default=ROOT / "configs/recovery_guards.yaml")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable table: {args.output}")
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protected_test_used") is not False or manifest.get("allowed_splits") != ["validation"]:
        raise SystemExit("cost tables are built from the validation-map recovery pilot only")
    with args.outcomes.open(newline="", encoding="utf-8") as stream:
        outcomes = list(csv.DictReader(stream))
    summaries_by_run: dict[str, dict[str, Any]] = {}
    sidecars_by_run: dict[str, dict[str, Any] | None] = {}
    for path in sorted(args.summaries.glob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or document.get("identity", {}).get("campaign_id") != manifest["campaign_id"]:
            continue
        run_id = str(document["identity"]["run_id"])
        summaries_by_run[run_id] = document
        sidecar = (document.get("system") or {}).get("manager_sidecar")
        sidecars_by_run[run_id] = load_sidecar(Path(sidecar)) if sidecar else None
    rows, report = cost_rows(outcomes, summaries_by_run, sidecars_by_run,
                             guard_config_from(args.guard_config))
    if report["guard_disagreements"]:
        raise SystemExit(f"recorded guard results disagree with the frozen guard: {report['guard_disagreements'][:5]}")
    payload = rows_to_csv_bytes(rows)
    publish_new_bytes(args.output, payload)
    provenance = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "manifest": str(args.manifest), "manifest_sha256": sha256_file(args.manifest),
        "outcomes": str(args.outcomes), "outcomes_sha256": sha256_file(args.outcomes),
        "guard_config": str(args.guard_config), "guard_config_sha256": sha256_file(args.guard_config),
        "output": str(args.output), "output_sha256": sha256_bytes(payload),
        "fit_split": "validation", "fit_split_admitted_by": "PA-2026-09-03-04",
        "selector_training_rule": "actions_rejected_by_guard_are_never_training_targets",
        "protected_test_used": False,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **report,
    }
    publish_new_bytes(args.output.with_name(args.output.name + ".provenance.json"),
                      (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({"output": str(args.output), "rows": len(rows),
                      "guard_eligible_rows": report["guard_eligible_rows"],
                      "no_alarm": len(report["no_alarm"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
