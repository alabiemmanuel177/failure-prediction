#!/usr/bin/env python3
"""Build the preregistered paired closed-loop recovery campaign manifest.

Design: 3 held-out maps x 8 routes x 3 seeds x 7 fault families = 504 base episodes,
each executed once under every admitted recovery policy (R0, R2, R3 under
PA-2026-09-03-04; 1,512 episodes with six routes and four seeds; the R0-vs-R3 pair is 1,008).

The manifest reads held-out maps and therefore fails closed unless
``configs/model_freeze.yaml`` declares ``frozen: true`` and the held-out split is
assigned. ``--dry-run --fake-split`` exercises the design with placeholder maps and
never writes under ``data/manifests``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import publish_new_bytes, sha256_file  # noqa: E402
from src.experiments.campaigns import expand_balanced_pilot  # noqa: E402
from src.protected_data import enforce_protected_boundary  # noqa: E402


CAMPAIGN_ID = "paired_recovery_v1"
# Protocol Amendment PA-2026-09-03-04: R0, R2 and R3 before the deadline; R1 deferred.
RECOVERY_POLICIES = ("R0", "R2", "R3")
ALL_POLICIES = ("R0", "R1", "R2", "R3")
PAIR_FIELDS = ("map", "route", "seed", "family", "severity")
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
DEFAULT_OUTPUT = ROOT / "data/manifests/paired_recovery_v1.yaml"


def fake_split(maps: int = 3, routes_per_map: int = 8) -> dict[str, Any]:
    map_ids = [f"fake_{index:02d}" for index in range(maps)]
    return {
        "held_out_map_test": {
            "maps": map_ids,
            "routes": [f"{map_id}_r{route}" for map_id in map_ids for route in range(routes_per_map)],
        }
    }


def held_out_map_routes(splits: dict[str, Any], routes_per_map: int) -> dict[str, list[str]]:
    split = splits.get("held_out_map_test", {})
    maps = list(split.get("maps") or [])
    routes = list(split.get("routes") or [])
    if len(maps) != 3:
        raise ValueError(f"held-out split must assign exactly three maps, found {len(maps)}")
    selected = {}
    for map_id in maps:
        own = [route for route in routes if route.startswith(f"{map_id}_")]
        if len(own) < routes_per_map:
            raise ValueError(
                f"held-out map {map_id} has {len(own)} routes; {routes_per_map} are required"
            )
        selected[map_id] = own[:routes_per_map]
    return selected


def build_manifest(
    map_routes: dict[str, list[str]], *, seeds_per_cell: int, seed_base: int,
    replicate_offset: int = 10,
    gate: dict[str, Any], selector_model: str, campaign_id: str = CAMPAIGN_ID,
) -> dict[str, Any]:
    routes_per_map = {len(routes) for routes in map_routes.values()}
    if len(routes_per_map) != 1:
        raise ValueError("every held-out map must contribute the same number of routes")
    base = len(map_routes) * routes_per_map.pop() * seeds_per_cell * len(FAULT_CONDITIONS)
    return {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "campaign_kind": "paired_recovery_confirmatory",
        "status": "preregistered_post_freeze_held_out_only",
        "purpose": (
            "paired closed-loop comparison of recovery policies R0-R3 on identical "
            "map/route/seed/fault episodes for confirmatory hypothesis H6"
        ),
        "analysis_plan": "RESEARCH PROTOCOL - AUTONOMOUS ROBOT RELIABILITY.md",
        "allowed_splits": ["held_out_map_test"],
        "protected_test_used": True,
        "parallel_execution_admitted_by": "PA-2026-09-04-02",
        "confirmatory_gate": gate,
        "execution_policy": {
            "exact_once_per_episode_key": True,
            "retain_all_bags": True,
            "require_confirmatory_readiness_gate": True,
            "require_recovery_live_evidence": "configs/recovery_live_evidence.yaml",
            "recovery_live_execution": True,
            "recovery_selector_model": selector_model,
            "stop_on_first_invalid": True,
            "concurrency": 1,
            "maximum_episodes_per_invocation": len(RECOVERY_POLICIES) * len(FAULT_CONDITIONS),
            "minimum_free_space_gib_before_episode": 100,
            "wave_design": "every_policy_of_one_pairing_cell_runs_consecutively",
            "threshold_adaptation": "forbidden_frozen_validation_threshold_only",
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
            "conditions": [
                {**condition, "replicates": list(range(replicate_offset, replicate_offset + seeds_per_cell))}
                for condition in FAULT_CONDITIONS
            ],
            "environment_placement": {"placement_mode": "path_fraction", "route_fraction": 0.55},
        },
        "recovery_policies": list(RECOVERY_POLICIES),
        "pairing": {
            "fields": list(PAIR_FIELDS),
            "policies_share_seed": True,
            "episode_key_layout": "<map>-<route>-<condition>-r<replicate>-<policy>",
            "launch_argument": "recovery_policy_id",
        },
        "expected_base_episode_count": base,
        "expected_episode_count": base * len(RECOVERY_POLICIES),
        "expected_pair_counts": {
            f"R3_vs_{policy}": base for policy in RECOVERY_POLICIES if policy != "R3"
        },
        "admission": {
            "protected_outcomes_consulted": False,
            "runs_only_after": [
                "configs/model_freeze.yaml frozen",
                "held_out_map_test split assigned",
                "scripts/check_readiness.py --stage confirmatory passes",
                "recovery live-execution evidence frozen",
            ],
            "analysis_script": "scripts/analyze_paired_recovery.py",
        },
    }


def expand_paired_recovery(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the manifest to per-policy episodes in pair-complete execution order."""
    base = expand_balanced_pilot(document)
    episodes = []
    for episode in base:
        for policy in document["recovery_policies"]:
            episodes.append({
                **episode,
                "episode_key": f"{episode['episode_key']}-{policy}",
                "pair_key": episode["episode_key"],
                "recovery_policy_id": policy,
            })
    return episodes


def validate_paired_recovery(document: dict[str, Any], splits: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    if document.get("allowed_splits") != ["held_out_map_test"]:
        findings.append("paired recovery must run on the held-out split only")
    if document.get("protected_test_used") is not True:
        findings.append("paired recovery must declare protected_test_used true")
    declared = list(document.get("recovery_policies", []))
    if not declared or any(policy not in ALL_POLICIES for policy in declared) \
            or len(set(declared)) != len(declared) or not {"R0", "R3"} <= set(declared):
        findings.append("recovery policies must be a unique subset of R0-R3 containing R0 and R3")
    split = splits.get("held_out_map_test", {})
    allowed_maps = set(split.get("maps", []))
    allowed_routes = set(split.get("routes", []))
    for map_id, routes in document.get("design", {}).get("map_routes", {}).items():
        if map_id not in allowed_maps:
            findings.append(f"map is outside held_out_map_test split: {map_id}")
        for route in routes:
            if route not in allowed_routes:
                findings.append(f"route is outside held_out_map_test split: {route}")
    families = [item["family"] for item in document.get("design", {}).get("conditions", [])]
    if sorted(families) != sorted(item["family"] for item in FAULT_CONDITIONS):
        findings.append("design must contain the seven frozen fault families exactly once")
    episodes = expand_paired_recovery(document)
    if len(episodes) != int(document.get("expected_episode_count", -1)):
        findings.append(f"expanded {len(episodes)} episodes, expected {document.get('expected_episode_count')}")
    if len({item["episode_key"] for item in episodes}) != len(episodes):
        findings.append("episode keys are not unique")
    per_policy = Counter(item["recovery_policy_id"] for item in episodes)
    if len(set(per_policy.values())) != 1:
        findings.append("policies do not receive identical episode counts")
    pairs: dict[tuple, set[str]] = {}
    for item in episodes:
        pairs.setdefault(tuple(item[field] for field in PAIR_FIELDS), set()).add(item["recovery_policy_id"])
    if any(policies != set(document.get("recovery_policies", RECOVERY_POLICIES)) for policies in pairs.values()):
        findings.append("some map/route/seed/fault cells are not paired across all policies")
    if len(pairs) != int(document.get("expected_base_episode_count", -1)):
        findings.append("pair count differs from expected_base_episode_count")
    if 2 * len(pairs) < 1008:
        findings.append("R0-vs-R3 comparison must contain at least 1,008 episodes")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--campaign-id", default=CAMPAIGN_ID,
                        help="campaign id (paired_recovery_v2 re-executes the v1 design with live recovery execution)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fake-split", action="store_true", help="dry-run only placeholder maps")
    parser.add_argument("--allow-protected-after-freeze", action="store_true")
    parser.add_argument("--routes-per-map", type=int, default=6)
    parser.add_argument("--seeds-per-cell", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=30_000_000)
    parser.add_argument("--replicate-offset", type=int, default=10,
                        help="first replicate number; keeps episode keys distinct from held_out_map_v1")
    parser.add_argument("--selector-model", default="models/recovery_selector/r3_cost_sensitive_ridge_v1.json")
    parser.add_argument("--freeze", type=Path, default=ROOT / "configs/model_freeze.yaml")
    parser.add_argument("--splits", type=Path, default=ROOT / "data/manifests/splits.template.yaml")
    args = parser.parse_args()
    if args.fake_split and not args.dry_run:
        raise SystemExit("--fake-split is permitted only with --dry-run")
    dry_run_print_only = args.dry_run and args.output == DEFAULT_OUTPUT
    if args.dry_run and not dry_run_print_only and args.output.resolve().is_relative_to(
        (ROOT / "data/manifests").resolve()
    ):
        raise SystemExit("dry runs never write under data/manifests; pass --output elsewhere")
    if args.fake_split:
        splits = fake_split(routes_per_map=args.routes_per_map)
        gate = {"model_freeze": None, "dry_run_fake_split": True, "frozen": False}
    else:
        freeze = yaml.safe_load(args.freeze.read_text(encoding="utf-8")) if args.freeze.exists() else None
        frozen = bool(freeze and freeze.get("frozen") is True)
        if not frozen:
            raise SystemExit("paired recovery manifest is blocked: configs/model_freeze.yaml is not frozen")
        if freeze.get("declaration", {}).get("protected_maps_or_outcomes_inspected") is not False:
            raise SystemExit("model freeze must declare protected maps and outcomes uninspected")
        enforce_protected_boundary(
            True, explicitly_allowed=args.allow_protected_after_freeze,
            confirmatory_gate_passed=frozen,
        )
        splits = yaml.safe_load(args.splits.read_text(encoding="utf-8"))
        gate = {
            "model_freeze": str(args.freeze.relative_to(ROOT)),
            "model_freeze_sha256": sha256_file(args.freeze),
            "splits_manifest": str(args.splits.relative_to(ROOT)),
            "splits_manifest_sha256": sha256_file(args.splits),
            "frozen": True,
        }
    try:
        map_routes = held_out_map_routes(splits, args.routes_per_map)
    except ValueError as error:
        raise SystemExit(f"paired recovery manifest is blocked: {error}")
    document = build_manifest(
        map_routes, seeds_per_cell=args.seeds_per_cell, seed_base=args.seed_base,
        replicate_offset=args.replicate_offset,
        gate=gate, selector_model=args.selector_model, campaign_id=args.campaign_id,
    )
    campaign_id = args.campaign_id
    findings = validate_paired_recovery(document, splits)
    if findings:
        raise SystemExit("invalid paired recovery design:\n- " + "\n- ".join(findings))
    episodes = expand_paired_recovery(document)
    summary = {
        "campaign_id": campaign_id,
        "base_episodes": document["expected_base_episode_count"],
        "episodes": len(episodes),
        "policies": list(RECOVERY_POLICIES),
        "r0_vs_r3_episodes": 2 * document["expected_base_episode_count"],
        "dry_run": args.dry_run,
        "fake_split": args.fake_split,
        "output": None if dry_run_print_only else str(args.output),
    }
    print(json.dumps(summary, indent=2))
    if dry_run_print_only:
        return 0
    payload = yaml.safe_dump(document, sort_keys=False).encode("utf-8")
    publish_new_bytes(args.output, payload)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
