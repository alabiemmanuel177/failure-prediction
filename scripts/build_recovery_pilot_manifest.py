#!/usr/bin/env python3
"""Build the validation-map recovery pilot manifest that trains the R3 selector.

Design (PA-2026-09-03-04, post-freeze, validation maps only):
  3 validation maps x 6 routes x 7 fault families (medium) x 1 seed = 126 cells,
  each executed under R0 (reference) and six forced-action policies
  RP_{controlled_stop, relocalise, replan_clear_costmaps, backup, spin_active_rescan,
  wait} triggered by the frozen predictor's first alarm: 7 policies x 126 = 882 episodes.
The guard keeps final authority: a rejected forced action is recorded as
``guard_rejected`` and executed as a controlled stop (``src.recovery.manager``).

The manifest reads no protected data (``allowed_splits: [validation]``,
``protected_test_used: false``) and is written immutably under ``data/manifests``
unless ``--output`` points elsewhere; ``--dry-run`` prints the design only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import publish_new_bytes, sha256_file  # noqa: E402
from src.experiments.campaigns import expand_balanced_pilot  # noqa: E402
from src.experiments.recovery_campaigns import (  # noqa: E402
    PILOT_POLICY_SET, RECOVERY_PILOT_CAMPAIGN_ID, RECOVERY_PILOT_KIND, expand_recovery_policies,
    validate_recovery_pilot,
)


ROUTES_PER_MAP = 6
FAULT_CONDITIONS = (
    {"id": "camera", "family": "camera_occlusion", "severity": "medium", "system": "s3"},
    {"id": "lidar", "family": "lidar_dropout", "severity": "medium", "system": "s0"},
    {"id": "wheel", "family": "wheel_slip", "severity": "medium", "system": "s0"},
    {"id": "localisation", "family": "localisation_perturbation", "severity": "medium",
     "system": "s0"},
    {"id": "blockage", "family": "dynamic_blockage", "severity": "medium", "system": "s0"},
    {"id": "oscillation", "family": "planner_oscillation", "severity": "medium", "system": "s0"},
    {"id": "semantic", "family": "semantic_corruption", "severity": "medium", "system": "s3"},
)
DEFAULT_OUTPUT = ROOT / "data/manifests/recovery_pilot_v1.yaml"


def validation_map_routes(splits: dict[str, Any], routes_per_map: int = ROUTES_PER_MAP) -> dict[str, list[str]]:
    split = splits.get("validation", {})
    maps = list(split.get("maps") or [])
    routes = list(split.get("routes") or [])
    if len(maps) != 3:
        raise ValueError(f"validation split must list exactly three maps, found {len(maps)}")
    selected = {}
    for map_id in maps:
        own = [route for route in routes if route.startswith(f"{map_id}_")]
        if len(own) < routes_per_map:
            raise ValueError(f"validation map {map_id} has {len(own)} routes; {routes_per_map} required")
        selected[map_id] = own[:routes_per_map]
    return selected


def build_manifest(map_routes: dict[str, list[str]], *, seed_base: int, gate: dict[str, Any]) -> dict[str, Any]:
    base = sum(len(routes) for routes in map_routes.values()) * len(FAULT_CONDITIONS)
    return {
        "schema_version": 1,
        "campaign_id": RECOVERY_PILOT_CAMPAIGN_ID,
        "campaign_kind": RECOVERY_PILOT_KIND,
        "status": "preregistered_post_freeze_validation_maps_only",
        "purpose": (
            "validation-map recovery pilot: observed cost of every guard-eligible forced "
            "action at the frozen predictor's first alarm, to fit the R3 selector "
            "(PA-2026-09-03-04)"
        ),
        "analysis_plan": "RESEARCH PROTOCOL - AUTONOMOUS ROBOT RELIABILITY.md",
        "allowed_splits": ["validation"],
        "protected_test_used": False,
        "parallel_execution_admitted_by": "PA-2026-09-04-02",
        "selector_training": {
            "fit_split": "validation",
            "fit_split_admitted_by": "PA-2026-09-03-04",
            "cost_table_builder": "scripts/build_recovery_cost_table.py",
            "trainer": "scripts/train_recovery_selector.py --fit-split validation --amendment-id PA-2026-09-03-04",
            "rule": "actions_rejected_by_guard_are_never_training_targets",
        },
        "freeze_gate": gate,
        "execution_policy": {
            "exact_once_per_episode_key": True,
            "retain_all_bags": True,
            "require_extraction_readiness_gate": True,
            "require_model_freeze": True,
            "require_recovery_live_evidence": "configs/recovery_live_evidence.yaml",
            "recovery_live_execution": True,
            "stop_on_first_invalid": True,
            "concurrency": 6,
            "ros_domains": [60, 61, 62, 63, 64, 65],
            "maximum_episodes_per_invocation": len(PILOT_POLICY_SET) * 6,
            "minimum_free_space_gib_before_episode": 100,
            "wave_design": "every_policy_of_one_pairing_cell_runs_consecutively",
            "threshold_adaptation": "forbidden_frozen_validation_threshold_only",
            "alarm_trigger": "first_alarm_of_the_frozen_predictor_under_the_frozen_alarm_policy",
        },
        "episode_defaults": {
            "clean_prefix_seconds": 5.0,
            "planned_onset_seconds": 8.0,
            "maximum_duration_seconds": 20.0,
            "maximum_wait_seconds": 5.0,
            "recording_profile": "compact_v2",
        },
        "infrastructure_replacements": [],
        "design": {
            "seed_base": seed_base,
            "seed_layout": "wide_v1",
            "map_routes": map_routes,
            "conditions": [{**condition, "replicates": [0]} for condition in FAULT_CONDITIONS],
            "environment_placement": {"placement_mode": "path_fraction", "route_fraction": 0.55},
        },
        "recovery_policies": list(PILOT_POLICY_SET),
        "pairing": {
            "fields": ["map", "route", "seed", "family", "severity"],
            "policies_share_seed": True,
            "episode_key_layout": "<map>-<route>-<condition>-r0-<policy>",
            "launch_argument": "recovery_policy",
            "cli_argument": "--recovery-policy",
            "reference_policy": "R0",
        },
        "expected_base_episode_count": base,
        "expected_episode_count": base * len(PILOT_POLICY_SET),
        "admission": {
            "protected_outcomes_consulted": False,
            "runs_only_after": [
                "configs/model_freeze.yaml frozen",
                "configs/alarm_policy.yaml threshold set by scripts/freeze_model.py",
                "configs/recovery_live_evidence.yaml frozen against configs/recovery_guards.yaml",
                "reports/integrity/concurrency_shift_check_v1.yaml passed (six workers)",
            ],
            "outcome_table": "scripts/build_recovery_outcome_table.py",
            "cost_table": "scripts/build_recovery_cost_table.py",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed-base", type=int, default=40_000_000)
    parser.add_argument("--freeze", type=Path, default=ROOT / "configs/model_freeze.yaml")
    parser.add_argument("--splits", type=Path, default=ROOT / "data/manifests/splits.template.yaml")
    args = parser.parse_args()
    dry_run_print_only = args.dry_run and args.output == DEFAULT_OUTPUT
    if args.dry_run and not dry_run_print_only and args.output.resolve().is_relative_to(
        (ROOT / "data/manifests").resolve()
    ):
        raise SystemExit("dry runs never write under data/manifests; pass --output elsewhere")
    freeze = yaml.safe_load(args.freeze.read_text(encoding="utf-8")) if args.freeze.exists() else None
    frozen = bool(freeze and freeze.get("frozen") is True)
    gate = {
        "model_freeze": str(args.freeze.relative_to(ROOT)) if args.freeze.is_relative_to(ROOT) else str(args.freeze),
        "model_freeze_sha256": sha256_file(args.freeze) if args.freeze.exists() else None,
        "frozen_at_manifest_build": frozen,
        "note": "the pilot executes only after the freeze; the manifest may be preregistered before it",
    }
    splits = yaml.safe_load(args.splits.read_text(encoding="utf-8"))
    try:
        map_routes = validation_map_routes(splits)
    except ValueError as error:
        raise SystemExit(f"recovery pilot manifest is blocked: {error}")
    document = build_manifest(map_routes, seed_base=args.seed_base, gate=gate)
    findings = validate_recovery_pilot(document, splits)
    if findings:
        raise SystemExit("invalid recovery pilot design:\n- " + "\n- ".join(findings))
    episodes = expand_recovery_policies(document, expand_balanced_pilot(document))
    print(json.dumps({
        "campaign_id": document["campaign_id"],
        "base_episodes": document["expected_base_episode_count"],
        "episodes": len(episodes),
        "policies": list(PILOT_POLICY_SET),
        "concurrency": document["execution_policy"]["concurrency"],
        "frozen_at_manifest_build": frozen,
        "dry_run": args.dry_run,
        "output": None if dry_run_print_only else str(args.output),
    }, indent=2))
    if dry_run_print_only:
        return 0
    publish_new_bytes(args.output, yaml.safe_dump(document, sort_keys=False).encode("utf-8"))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
