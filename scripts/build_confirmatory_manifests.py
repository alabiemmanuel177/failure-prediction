#!/usr/bin/env python3
"""Build the post-freeze held-out-map and severity-stress campaign manifests.

Requires an assigned ``held_out_map_test`` split (``scripts/assign_protected_split.py``).
Writes two immutable manifests in the balanced-campaign schema so
``scripts/run_balanced_pilot_continuous.py`` can execute them after the confirmatory
readiness gate passes:

* ``data/manifests/held_out_map_v1.yaml``: 960 = 3 maps x 8 routes x 5 seeds x
  8 conditions (clean control + seven families at the frozen primary severity).
* ``data/manifests/severity_stress_v1.yaml``: 1512 = 3 maps x 8 routes x 3 seeds x
  7 families x 3 severities.

``--dry-run`` expands both designs, checks seed and episode-key collisions against
every existing campaign manifest, and writes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_inventory import publish_new_bytes
from src.experiments import (
    expand_balanced_pilot, targeted_execution_order, validate_confirmatory_campaign,
)
from src.protected_data import HELD_OUT_SPLIT, held_out_assignment_findings

FAMILIES = (
    ("camera", "camera_occlusion", "s3"),
    ("lidar", "lidar_dropout", "s0"),
    ("wheel", "wheel_slip", "s0"),
    ("localisation", "localisation_perturbation", "s0"),
    ("blockage", "dynamic_blockage", "s0"),
    ("oscillation", "planner_oscillation", "s0"),
    ("semantic", "semantic_corruption", "s3"),
)
PRIMARY_SEVERITY = "medium"
SEVERITIES = ("low", "medium", "high")
HELD_OUT_SEED_BASE = 10_000_000
SEVERITY_SEED_BASE = 20_000_000
# Protocol 1.0 minimum campaign sizes. Protocol Amendment PA-2026-09-03-02 fixes six
# routes per held-out map and keeps these minimums by adding seeds per cell.
HELD_OUT_MINIMUM_EPISODES = 960
SEVERITY_MINIMUM_EPISODES = 1512


def seeds_for_minimum(route_count: int, cells_per_seed: int, minimum: int) -> int:
    """Smallest seed count whose full matrix reaches the preregistered minimum."""
    if route_count <= 0 or cells_per_seed <= 0:
        raise ValueError("route and cell counts must be positive")
    return max(1, -(-minimum // (route_count * cells_per_seed)))
EPISODE_DEFAULTS = {
    "clean_prefix_seconds": 5.0,
    "planned_onset_seconds": 8.0,
    "maximum_duration_seconds": 20.0,
    "maximum_wait_seconds": 5.0,
    "recording_profile": "compact_v2",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def held_out_map_routes(splits: dict) -> dict[str, list[str]]:
    split = splits[HELD_OUT_SPLIT]
    map_routes: dict[str, list[str]] = {map_id: [] for map_id in split["maps"]}
    for route_id in split["routes"]:
        owner = [map_id for map_id in map_routes if route_id.startswith(f"{map_id}_")]
        if len(owner) != 1:
            raise ValueError(f"route {route_id} does not belong to exactly one held-out map")
        map_routes[owner[0]].append(route_id)
    return map_routes


def execution_policy(wave: int) -> dict:
    return {
        "exact_once_per_episode_key": True,
        "retain_all_bags": True,
        "refuse_while_research1_confirmatory_active": True,
        "require_confirmatory_readiness_gate": True,
        "stop_on_first_invalid": True,
        "concurrency": 1,
        "maximum_episodes_per_invocation": wave,
        "minimum_free_space_gib_before_episode": 100,
        "wave_design": "one_condition_and_replicate_across_all_held_out_routes",
    }


def provenance(splits_path: Path, record_path: Path | None, freeze_path: Path | None) -> dict:
    result = {
        "split_manifest": str(splits_path.relative_to(ROOT)) if splits_path.is_relative_to(ROOT)
        else str(splits_path),
        "split_manifest_sha256": sha256_file(splits_path),
    }
    if record_path is not None and record_path.exists():
        result["split_assignment_record"] = str(record_path)
        result["split_assignment_sha256"] = sha256_file(record_path)
    if freeze_path is not None and freeze_path.exists():
        result["model_freeze"] = str(freeze_path)
        result["model_freeze_sha256"] = sha256_file(freeze_path)
    return result


def held_out_manifest(map_routes: dict[str, list[str]], prov: dict, seeds: int = 5) -> dict:
    replicates = list(range(seeds))
    # Clean control per system: the single clean slot is split across S0 and S3 so
    # that both deployed systems contribute clean missions to the false-alert burden
    # while the protocol's 8-condition x 5-seed = 960 design is preserved exactly.
    clean_s0 = replicates[: (seeds + 1) // 2]
    clean_s3 = replicates[(seeds + 1) // 2:]
    conditions = [
        {"id": "clean_s0", "family": "none", "severity": "none", "system": "s0",
         "replicates": clean_s0},
        {"id": "clean_s3", "family": "none", "severity": "none", "system": "s3",
         "replicates": clean_s3},
    ]
    for condition_id, family, system in FAMILIES:
        conditions.append({
            "id": condition_id, "family": family, "severity": PRIMARY_SEVERITY,
            "system": system, "replicates": list(replicates),
        })
    route_count = sum(len(routes) for routes in map_routes.values())
    return {
        "schema_version": 1,
        "campaign_id": "held_out_map_v1",
        "campaign_kind": "held_out_confirmatory",
        "status": "preregistered_post_freeze_confirmatory",
        "purpose": (
            "one-time confirmatory held-out-map evaluation set: clean control plus seven "
            "fault families at the frozen primary severity on the protected maps"
        ),
        "protocol_reference": "section 8 held-out confirmatory set (960 episodes)",
        "allowed_splits": [HELD_OUT_SPLIT],
        "protected_test_used": True,
        "parallel_execution_admitted_by": "PA-2026-09-04-02",
        "provenance": prov,
        "execution_policy": execution_policy(route_count),
        "episode_defaults": dict(EPISODE_DEFAULTS),
        "infrastructure_replacements": [],
        "design": {
            "seed_base": HELD_OUT_SEED_BASE,
            "seed_layout": "wide_v1",
            "map_routes": {map_id: list(routes) for map_id, routes in map_routes.items()},
            "clean_control_design": "single_clean_slot_split_across_s0_and_s3_seeds",
            "conditions": conditions,
            "environment_placement": {"placement_mode": "path_fraction", "route_fraction": 0.55},
        },
        "expected_episode_count": route_count * seeds * (len(FAMILIES) + 1),
        "admission": {
            "collection_only_after_model_freeze_and_split_assignment": True,
            "fitting_or_selection_use": "forbidden",
            "evaluation_use": "one_time_confirmatory_after_predictions_are_immutable",
            "threshold_adaptation": "forbidden",
            "protected_outcomes_consulted_before_freeze": False,
        },
    }


def severity_stress_manifest(map_routes: dict[str, list[str]], prov: dict, seeds: int = 3) -> dict:
    replicates = list(range(seeds))
    conditions = []
    for condition_id, family, system in FAMILIES:
        for severity in SEVERITIES:
            conditions.append({
                "id": f"{condition_id}_{severity}", "family": family,
                "severity": severity, "system": system, "replicates": list(replicates),
            })
    route_count = sum(len(routes) for routes in map_routes.values())
    return {
        "schema_version": 1,
        "campaign_id": "severity_stress_v1",
        "campaign_kind": "held_out_confirmatory",
        "status": "preregistered_post_freeze_severity_stress",
        "purpose": (
            "severity stress set on the protected maps: seven families at low, medium "
            "and high severity; supporting analysis only"
        ),
        "protocol_reference": "section 8 severity stress set (1,512 episodes)",
        "allowed_splits": [HELD_OUT_SPLIT],
        "protected_test_used": True,
        "parallel_execution_admitted_by": "PA-2026-09-04-02",
        "provenance": prov,
        "execution_policy": execution_policy(route_count),
        "episode_defaults": dict(EPISODE_DEFAULTS),
        "infrastructure_replacements": [],
        "design": {
            "seed_base": SEVERITY_SEED_BASE,
            "seed_layout": "wide_v1",
            "map_routes": {map_id: list(routes) for map_id, routes in map_routes.items()},
            "conditions": conditions,
            "environment_placement": {"placement_mode": "path_fraction", "route_fraction": 0.55},
        },
        "expected_episode_count": route_count * seeds * len(FAMILIES) * len(SEVERITIES),
        "admission": {
            "collection_only_after_model_freeze_and_split_assignment": True,
            "fitting_or_selection_use": "forbidden",
            "evaluation_use": "supporting_severity_analysis_at_the_frozen_threshold",
            "threshold_adaptation": "forbidden",
            "protected_outcomes_consulted_before_freeze": False,
        },
    }


def existing_campaign_keys(manifest_dir: Path, skip: set[str]) -> dict[str, dict]:
    """Seeds and episode keys of every expandable campaign manifest already present."""
    result: dict[str, dict] = {}
    for path in sorted(manifest_dir.glob("*.yaml")):
        if path.name in skip:
            continue
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            continue
        if not isinstance(document, dict) or "campaign_id" not in document:
            continue
        design = document.get("design") or {}
        seeds: set[int] = set()
        keys: set[str] = set()
        if design.get("map_routes") and design.get("conditions") and document.get("episode_defaults"):
            try:
                episodes = expand_balanced_pilot(document)
            except (KeyError, TypeError, ValueError):
                episodes = []
            seeds = {int(item["seed"]) for item in episodes}
            keys = {item["episode_key"] for item in episodes}
        else:
            for item in document.get("episodes") or []:
                if isinstance(item, dict):
                    if "seed" in item:
                        seeds.add(int(item["seed"]))
                    if "episode_key" in item:
                        keys.add(str(item["episode_key"]))
        result[path.name] = {"campaign_id": document["campaign_id"], "seeds": seeds, "keys": keys,
                             "parent_campaign_id": document.get("parent_campaign_id")}
    return result


def collision_findings(manifests: list[dict], existing: dict[str, dict]) -> list[str]:
    findings: list[str] = []
    expanded = {item["campaign_id"]: expand_balanced_pilot(item) for item in manifests}
    for campaign_id, episodes in expanded.items():
        seeds = {int(item["seed"]) for item in episodes}
        keys = {item["episode_key"] for item in episodes}
        for other_id, other in expanded.items():
            if other_id <= campaign_id:
                continue
            other_seeds = {int(item["seed"]) for item in other}
            if seeds & other_seeds:
                findings.append(f"seed collision between {campaign_id} and {other_id}")
            if keys & {item["episode_key"] for item in other}:
                findings.append(f"episode-key collision between {campaign_id} and {other_id}")
        for name, info in existing.items():
            if info["campaign_id"] == campaign_id:
                findings.append(f"campaign id {campaign_id} already exists in {name}")
            if info.get("parent_campaign_id") == campaign_id:
                # A declared replacement campaign reuses its parent's design cell and
                # seed by construction; its keys carry a distinct replacement suffix.
                continue
            if seeds & info["seeds"]:
                findings.append(f"seed collision between {campaign_id} and {name}")
            if keys & info["keys"]:
                findings.append(f"episode-key collision between {campaign_id} and {name}")
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=ROOT / "data/manifests/splits.template.yaml")
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "data/manifests")
    parser.add_argument(
        "--assignment-record", type=Path,
        default=ROOT / "reports/confirmatory/protected_split_assignment.yaml",
    )
    parser.add_argument("--model-freeze", type=Path, default=ROOT / "configs/model_freeze.yaml")
    parser.add_argument("--held-out-seeds", type=int, default=None,
                        help="seeds per cell (default: smallest count reaching 960 episodes)")
    parser.add_argument("--severity-seeds", type=int, default=None,
                        help="seeds per cell (default: smallest count reaching 1,512 episodes)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    splits = yaml.safe_load(args.splits.read_text(encoding="utf-8")) or {}
    findings = held_out_assignment_findings(splits)
    if findings:
        raise SystemExit(
            "confirmatory manifests require an assigned held-out split:\n- "
            + "\n- ".join(findings)
        )
    map_routes = held_out_map_routes(splits)
    prov = provenance(args.splits, args.assignment_record, args.model_freeze)
    route_count = sum(len(routes) for routes in map_routes.values())
    held_out_seeds = args.held_out_seeds or seeds_for_minimum(
        route_count, len(FAMILIES) + 1, HELD_OUT_MINIMUM_EPISODES)
    severity_seeds = args.severity_seeds or seeds_for_minimum(
        route_count, len(FAMILIES) * len(SEVERITIES), SEVERITY_MINIMUM_EPISODES)
    manifests = [held_out_manifest(map_routes, prov, seeds=held_out_seeds),
                 severity_stress_manifest(map_routes, prov, seeds=severity_seeds)]
    outputs = {item["campaign_id"]: args.manifest_dir / f"{item['campaign_id']}.yaml"
               for item in manifests}
    problems: list[str] = []
    for item in manifests:
        problems.extend(f"{item['campaign_id']}: {finding}"
                        for finding in validate_confirmatory_campaign(item, splits))
    existing = existing_campaign_keys(args.manifest_dir, skip={path.name for path in outputs.values()})
    problems.extend(collision_findings(manifests, existing))
    for campaign_id, path in outputs.items():
        if path.exists():
            problems.append(f"{campaign_id}: refusing to overwrite immutable manifest {path}")
    summary = {}
    for item in manifests:
        episodes = targeted_execution_order(expand_balanced_pilot(item))
        wave = item["execution_policy"]["maximum_episodes_per_invocation"]
        summary[item["campaign_id"]] = {
            "expected_episode_count": item["expected_episode_count"],
            "expanded": len(episodes),
            "unique_keys": len({episode["episode_key"] for episode in episodes}),
            "unique_seeds": len({episode["seed"] for episode in episodes}),
            "seed_range": [min(e["seed"] for e in episodes), max(e["seed"] for e in episodes)],
            "waves": len(episodes) // wave,
            "wave_size": wave,
            "conditions": len(item["design"]["conditions"]),
            "output": str(outputs[item["campaign_id"]]),
        }
    print(json.dumps({
        "dry_run": args.dry_run,
        "held_out_map_test": {"maps": list(map_routes), "routes": sum(map(len, map_routes.values()))},
        "campaigns": summary,
        "existing_campaigns_checked": sorted(existing),
        "findings": problems,
        "protected_test_used": True,
        "parallel_execution_admitted_by": "PA-2026-09-04-02",
    }, indent=2, sort_keys=True))
    if problems:
        return 1
    if args.dry_run:
        return 0
    for item in manifests:
        payload = yaml.safe_dump(item, sort_keys=False, width=100).encode("utf-8")
        publish_new_bytes(outputs[item["campaign_id"]], payload)
        print(f"wrote {outputs[item['campaign_id']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
