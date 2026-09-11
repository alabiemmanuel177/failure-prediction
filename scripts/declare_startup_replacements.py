#!/usr/bin/env python3
"""Declare exact same-cell replacements for pre-goal startup invalids of a campaign.

Scans the campaign ledger for invalid attempts whose retained summary says
``missing_mandatory_topic_before_goal`` with zero bag files (the simulator or Nav2
never came up before goal dispatch: a pure infrastructure failure), and appends one
preregistered replacement per affected design cell to the campaign's child
replacement manifest (``<campaign>_replacements_v1.yaml`` with
``parent_campaign_id`` = campaign). Idempotent; never touches attempts that already
have a declaration; refuses treatment-delivery or post-goal invalids, which need a
human decision. Every declaration is appended to the research log.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments import campaign_episodes, declared_replacements  # noqa: E402

ACTOR = "Claude Fable 5.1 (AI assistant, directed by the researcher)"
STARTUP_REASON = "missing_mandatory_topic_before_goal"
FIELDS = ("clean_prefix_seconds", "planned_onset_seconds", "maximum_duration_seconds",
          "maximum_wait_seconds", "recording_profile")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="parent campaign manifest")
    parser.add_argument("--replacements", type=Path, default=None,
                        help="child replacement manifest (default <stem without _v1>_replacements_v1.yaml)")
    parser.add_argument("--max-new", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    parent = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    campaign_id = parent["campaign_id"]
    child_path = args.replacements or (args.manifest.parent / f"{args.manifest.stem.replace('_v1', '')}_replacements_v1.yaml")
    child = yaml.safe_load(child_path.read_text(encoding="utf-8")) if child_path.exists() else None
    ledger = ROOT / "logs/campaigns" / f"{campaign_id}.jsonl"
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    declared = {item["original_episode_key"] for item in (child or {}).get("infrastructure_replacements", [])}
    # Cells declared in any other child manifest (treatment, artifact-loss, refusal
    # policies) are resolved elsewhere and never need a startup declaration here.
    declared |= {str(item["original_episode_key"]) for item in declared_replacements(ROOT, parent)}
    episodes = {item["episode_key"]: item for item in campaign_episodes(parent)}
    defaults = parent["episode_defaults"]
    new = []
    for index, row in enumerate(rows, 1):
        if not (row.get("returncode") or row.get("artifact_validation_returncode")):
            continue
        key = row["episode_key"]
        if key in declared or key not in episodes:
            continue
        log_path = ROOT / "logs/campaigns" / f"{campaign_id}.parallel" / f"slot{row.get('worker_slot', 0)}" / f"{key}.log"
        text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        run_id = row.get("run_id")
        if not run_id and "{" in text:
            run_id = json.loads(text[text.index("{"):text.index("}") + 1]).get("run_id")
        summary_path = ROOT / "data/raw/summaries" / f"{run_id}.yaml" if run_id else None
        if not (summary_path and summary_path.exists()):
            print(f"skip {key}: no retained summary (needs a human decision)")
            continue
        summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
        reason = summary["outcome"].get("invalid_reason")
        mcap = summary["provenance"].get("bag_mcap_count")
        if reason != STARTUP_REASON or mcap != 0:
            print(f"skip {key}: {reason} with {mcap} bag file(s) is not a pre-goal startup invalid")
            continue
        detail = text.splitlines()[1].strip() if len(text.splitlines()) > 1 else "pre-goal startup failure"
        ep = episodes[key]
        rep_key = f"{key}-infra-replacement-001"
        new.append((row, key, run_id, index, detail, ep, rep_key))
    if not new:
        print("no undeclared pre-goal startup invalids")
        return 0
    if len(new) > args.max_new:
        raise SystemExit(f"{len(new)} undeclared startup invalids exceed --max-new {args.max_new}; stop and inspect")
    if child is None:
        child = {
            "schema_version": 1, "protocol_version": "1.0",
            "campaign_id": f"{campaign_id.replace('_v1', '')}_replacements_v1", "parent_campaign_id": campaign_id,
            "campaign_kind": "held_out_confirmatory_replacement" if parent.get("protected_test_used") else "replacement",
            "status": "preregistered_pre_goal_startup_replacement",
            "allowed_splits": list(parent["allowed_splits"]), "protected_test_used": parent["protected_test_used"],
            "parallel_execution_admitted_by": parent.get("parallel_execution_admitted_by"),
            "replacement_policy": {
                "original_invalid_kind": "pre_goal_unmaterialized_attempt",
                "original_attempt_and_logs_are_retained": True,
                "original_scientific_admission": "excluded_because_the_simulator_or_nav2_did_not_come_up_before_goal_dispatch",
                "replacement_uses_same_map_route_system_condition_and_seed": True,
                "replacement_has_distinct_episode_key_and_run_id": True,
                "replacement_counts_once_as_the_original_scientific_design_cell": True,
                "maximum_replacements_for_this_design_cell": 1,
                "execution": "sequential_alone_after_all_phases_no_other_simulator_running",
            },
            "episodes": [], "infrastructure_replacements": [],
        }
    for row, key, run_id, index, detail, ep, rep_key in new:
        child["episodes"].append({
            "episode_key": rep_key, "replaces_episode_key": key, "replaces_run_id": run_id,
            "map": ep["map"], "route": ep["route"], "system": ep["system"], "family": ep["family"],
            "severity": ep["severity"], "seed": ep["seed"], **{k: defaults[k] for k in FIELDS},
            **({"placement_mode": ep["placement_mode"], "route_fraction": ep["route_fraction"]} if "placement_mode" in ep else {}),
            # Recovery campaigns: the replacement replays the same policy of the same pairing cell.
            **{k: ep[k] for k in ("recovery_policy_id", "pair_key", "recovery_live_execution",
                                  "recovery_selector_model") if k in ep},
        })
        child["infrastructure_replacements"].append({
            "original_episode_key": key, "original_run_id": run_id, "original_attempt_number": index,
            "invalid_reason": STARTUP_REASON, "invalid_detail": detail, "original_bag_mcap_count": 0,
            "replacement_campaign_id": child["campaign_id"], "replacement_episode_key": rep_key,
            "replacement_profile": defaults["recording_profile"], "retain_original_attempt": True,
            "scientific_episode_count_contribution": 1,
        })
        print(f"declare {rep_key} <- {run_id}: {detail}")
    if args.dry_run:
        return 0
    child_path.write_text(yaml.safe_dump(child, sort_keys=False), encoding="utf-8")
    keys = [item[1] for item in new]
    subprocess.run([sys.executable, str(ROOT / "scripts/research_log.py"), "add", "--kind", "exclusion", "--actor", ACTOR,
                    "--message", f"Declared {len(new)} exact same-cell replacement(s) for pre-goal startup invalid attempt(s) "
                    f"of {campaign_id} ({', '.join(keys)}): the simulator or Nav2 did not come up before goal dispatch "
                    f"(missing_mandatory_topic_before_goal, zero bag files); originals retained and excluded; "
                    f"declarations in {child_path.resolve().relative_to(ROOT)}.",
                    "--metadata", json.dumps({"campaign": campaign_id, "keys": keys, "protected_outcomes_consulted": False})],
                   check=True, stdout=subprocess.DEVNULL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
