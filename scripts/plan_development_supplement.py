#!/usr/bin/env python3
"""Preregister the route-balanced development supplement from the adapter audit.

episodes = max(324, 3000 - 648 - 1212 - N_r1), rounded up to complete 12-route blocks,
where N_r1 is the number of Research 1 development rows admitted by
``reports/integrity/research1_causal_adapter_audit_v1.yaml``. Blocks are spread across
the seven fault families and the two clean-control systems in the balanced-pilot
condition schema; replicates continue after every replicate used by
``balanced_pilot_v1`` and ``targeted_development_v1`` and the seed base is distinct, so
no episode key or seed can collide. Collision freedom is verified by expanding all
three manifests with ``src.experiments.expand_balanced_pilot`` (the same logic
``scripts/run_balanced_pilot.py`` executes).

    python3 scripts/plan_development_supplement.py [--dry-run]
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import publish_new_bytes, sha256_file  # noqa: E402
from src.experiments import expand_balanced_pilot, validate_balanced_pilot  # noqa: E402
from src.research1_adapter import (  # noqa: E402
    BALANCED_DEVELOPMENT_EPISODES, MINIMUM_SUPPLEMENT_EPISODES, PROTOCOL_FITTING_TARGET,
    ROUTES_PER_BLOCK, TARGETED_DEVELOPMENT_EPISODES, distribute_blocks,
    supplement_episode_count,
)


AUDIT = ROOT / "reports/integrity/research1_causal_adapter_audit_v1.yaml"
OUTPUT = ROOT / "data/manifests/development_supplement_v1.yaml"
BALANCED = ROOT / "data/manifests/balanced_pilot_v1.yaml"
TARGETED = ROOT / "data/manifests/targeted_development_v1.yaml"
SPLITS = ROOT / "data/manifests/splits.template.yaml"
SEED_BASE = 2_000_000
CAMPAIGN_ID = "development_supplement_v1"

# Same condition schema as the balanced pilot (ids, families, systems, severity).
CONDITIONS = [
    {"id": "clean_s0", "family": "none", "severity": "none", "system": "s0"},
    {"id": "clean_s3", "family": "none", "severity": "none", "system": "s3"},
    {"id": "camera", "family": "camera_occlusion", "severity": "medium", "system": "s3"},
    {"id": "lidar", "family": "lidar_dropout", "severity": "medium", "system": "s0"},
    {"id": "wheel", "family": "wheel_slip", "severity": "medium", "system": "s0"},
    {"id": "localisation", "family": "localisation_perturbation", "severity": "medium", "system": "s0"},
    {"id": "blockage", "family": "dynamic_blockage", "severity": "medium", "system": "s0"},
    {"id": "oscillation", "family": "planner_oscillation", "severity": "medium", "system": "s0"},
    {"id": "semantic", "family": "semantic_corruption", "severity": "medium", "system": "s3"},
]


def max_replicate(document: dict) -> int:
    design = document["design"]
    values = list(design.get("replicates", []))
    for condition in design.get("conditions", []):
        values.extend(condition.get("replicates", []))
    return max(int(value) for value in values)


def build_supplement(
    admitted_research1: int, balanced: dict, targeted: dict, *, audit_path: str,
    audit_sha256: str, now_utc: str,
) -> dict:
    total = supplement_episode_count(admitted_research1)
    blocks = total // ROUTES_PER_BLOCK
    per_condition = distribute_blocks(blocks, len(CONDITIONS))
    first_replicate = max(max_replicate(balanced), max_replicate(targeted)) + 1
    conditions = []
    for condition, count in zip(CONDITIONS, per_condition):
        conditions.append({
            **condition,
            "replicates": [first_replicate + index for index in range(count)],
        })
    conditions = [item for item in conditions if item["replicates"]]
    document = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "campaign_kind": "development_supplement",
        "status": "preregistered_waiting_for_targeted_completion",
        "purpose": (
            f"close the fitting-volume gap to {PROTOCOL_FITTING_TARGET} development "
            "episodes after the Research 1 causal-adapter audit, route-balanced across the "
            "seven fault families and two clean-control systems"
        ),
        "planning_source": audit_path,
        "planning_source_sha256": audit_sha256,
        "split_accounting_correction": "reports/pilot/training_volume_split_correction_v1.yaml",
        "preregistered_utc": now_utc,
        "allowed_splits": ["development"],
        "protected_test_used": False,
        "sizing": {
            "formula": "max(324, 3000 - 648 - 1212 - N_r1) rounded up to complete 12-route blocks",
            "protocol_fitting_target": PROTOCOL_FITTING_TARGET,
            "balanced_development_episodes": BALANCED_DEVELOPMENT_EPISODES,
            "targeted_development_episodes": TARGETED_DEVELOPMENT_EPISODES,
            "research1_development_admitted": admitted_research1,
            "raw_shortfall": PROTOCOL_FITTING_TARGET - BALANCED_DEVELOPMENT_EPISODES
            - TARGETED_DEVELOPMENT_EPISODES - admitted_research1,
            "minimum_supplement": MINIMUM_SUPPLEMENT_EPISODES,
            "routes_per_block": ROUTES_PER_BLOCK,
            "complete_blocks": blocks,
            "block_distribution_rule": "round-robin over conditions in listed order; counts differ by at most one",
            "replicates_per_condition": {
                item["id"]: len(item["replicates"]) for item in conditions
            },
            "first_replicate": first_replicate,
        },
        "execution_policy": copy.deepcopy(targeted["execution_policy"]),
        "episode_defaults": copy.deepcopy(targeted["episode_defaults"]),
        "infrastructure_replacements": [],
        "design": {
            "seed_base": SEED_BASE,
            "seed_layout": "wide_v1",
            "map_routes": copy.deepcopy(targeted["design"]["map_routes"]),
            "conditions": conditions,
            "environment_placement": copy.deepcopy(targeted["design"]["environment_placement"]),
        },
        "expected_episode_count": total,
        "admission": {
            "raw_collection_before_human_gate": "permitted_under_the_approved_dependency_boundary",
            "collection_start_condition": "targeted_development_v1 complete and inventoried",
            "fitting_use": "only_after_training_readiness_and_research1_adapter_audits_pass",
            "protected_outcomes_consulted": False,
            "research1_validation_episodes_reserved_for_selection": True,
            "key_collision_check": "expanded against balanced_pilot_v1 and targeted_development_v1 with src.experiments.expand_balanced_pilot",
        },
    }
    if len(set(per_condition)) == 1:
        document["expected_per_condition"] = per_condition[0] * ROUTES_PER_BLOCK
    document["execution_policy"]["wave_design"] = (
        "one_condition_and_replicate_across_all_12_development_routes"
    )
    return document


def collision_findings(supplement: dict, balanced: dict, targeted: dict) -> list[str]:
    findings: list[str] = []
    ours = expand_balanced_pilot(supplement)
    keys = [item["episode_key"] for item in ours]
    seeds = [item["seed"] for item in ours]
    if len(set(keys)) != len(keys):
        findings.append("supplement episode keys are not unique")
    if len(set(seeds)) != len(seeds):
        findings.append("supplement seeds are not unique")
    if len(ours) != int(supplement["expected_episode_count"]):
        findings.append("expanded supplement count differs from expected_episode_count")
    for name, other in (("balanced_pilot_v1", balanced), ("targeted_development_v1", targeted)):
        theirs = expand_balanced_pilot(other)
        shared_keys = set(keys) & {item["episode_key"] for item in theirs}
        shared_seeds = set(seeds) & {item["seed"] for item in theirs}
        if shared_keys:
            findings.append(f"episode keys collide with {name}: {sorted(shared_keys)[:3]}")
        if shared_seeds:
            findings.append(f"seeds collide with {name}: {sorted(shared_seeds)[:3]}")
        replacement_keys = {
            item.get("replacement_episode_key") for item in other.get("infrastructure_replacements", [])
        }
        if set(keys) & replacement_keys:
            findings.append(f"episode keys collide with {name} replacement keys")
    if any(key.startswith("r1-") for key in keys):
        findings.append("supplement keys must not use the Research 1 adapter prefix")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.audit.is_file():
        raise SystemExit(f"adapter audit not found: {args.audit}; run the audit first")
    audit = yaml.safe_load(args.audit.read_text(encoding="utf-8"))
    if audit.get("status") != "complete" or audit.get("protected_outcomes_consulted") is not False:
        raise SystemExit("adapter audit is not a complete, outcome-safe report; refusing")
    admitted = int(audit["counts"]["admitted"])
    balanced = yaml.safe_load(BALANCED.read_text(encoding="utf-8"))
    targeted = yaml.safe_load(TARGETED.read_text(encoding="utf-8"))
    audit_path = str(args.audit.relative_to(ROOT)) if args.audit.is_relative_to(ROOT) else str(args.audit)
    document = build_supplement(
        admitted, balanced, targeted, audit_path=audit_path, audit_sha256=sha256_file(args.audit),
        now_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    splits = yaml.safe_load(SPLITS.read_text(encoding="utf-8"))
    findings = collision_findings(document, balanced, targeted)
    if "expected_per_condition" in document:
        findings.extend(validate_balanced_pilot(document, splits))
    if findings:
        raise SystemExit("supplement plan is invalid:\n- " + "\n- ".join(findings))
    payload = yaml.safe_dump(document, sort_keys=False).encode("utf-8")
    summary = {
        "campaign_id": CAMPAIGN_ID,
        "research1_development_admitted": admitted,
        "supplement_episodes": document["expected_episode_count"],
        "complete_blocks": document["sizing"]["complete_blocks"],
        "replicates_per_condition": document["sizing"]["replicates_per_condition"],
        "first_replicate": document["sizing"]["first_replicate"],
        "collision_free_against": ["balanced_pilot_v1", "targeted_development_v1"],
    }
    if args.dry_run:
        print(yaml.safe_dump({"dry_run": True, **summary}, sort_keys=False))
        return 0
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    publish_new_bytes(args.output, payload)
    print(yaml.safe_dump({"written": str(args.output), **summary}, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
