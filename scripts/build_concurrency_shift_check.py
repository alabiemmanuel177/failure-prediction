#!/usr/bin/env python3
"""Preregister a sequential-versus-six-worker health-feature shift check.

Every variant re-runs the same 36 development episodes (12 development routes x
{clean S0, clean S3, medium planner oscillation on S0}) under six concurrent worker
slots so ``scripts/evaluate_concurrency_shift.py`` can compare the deployable health
channels against the sequential development dataset
``balanced_pilot_v1-development-648``.

Variants:

* ``six_workers`` (``concurrency_shift_check_v1``, seed base 4,000,000): six workers
  with no per-system cap. Its report failed: clean S3 (the only system whose
  perception runs on the single GPU) produced 15 false alerts in 12 missions and
  ``inference_latency_ms`` shifted beyond 0.5 SMD, while clean S0 was unchanged.
* ``s3_serialised`` (``concurrency_shift_check_v2``, seed base 4,500,000): six
  workers with ``execution_policy.system_concurrency: {s3: 1}``, so at most one S3
  episode runs at a time and the GPU-contention hypothesis is tested directly.

Each variant's seed base uses the ``wide_v1`` layout and is disjoint from every
existing campaign; disjointness is verified by expanding every manifest under
``data/manifests`` (including the other variant) with the same helpers the runners
execute, and against the reserved confirmatory seed bases. The v2 base is 4,500,000
rather than 5,000,000 because under ``wide_v1`` a base of 5,000,000 lands map index
1 on v1's seeds (v1 map index 1 starts at 5,000,000) and map index 5 on the reserved
held-out base 10,000,000.

    python3 scripts/build_concurrency_shift_check.py [--variant s3_serialised] [--dry-run]
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_live_integrity_campaign import expand as expand_integrity  # noqa: E402
from src.dataset_inventory import publish_new_bytes, sha256_file  # noqa: E402
from src.experiments import (  # noqa: E402
    CONCURRENCY_CHECK_KIND, expand_balanced_pilot, targeted_execution_order,
    validate_concurrency_check,
)

VARIANTS: dict[str, dict] = {
    "six_workers": {
        "campaign_id": "concurrency_shift_check_v1",
        "seed_base": 4_000_000,
        "purpose": "measure the health-feature shift between sequential and six-worker execution",
        "system_concurrency": None,
        "wave_design": "one_condition_and_replicate_across_all_12_development_routes_by_six_workers",
    },
    "s3_serialised": {
        "campaign_id": "concurrency_shift_check_v2",
        "seed_base": 4_500_000,
        "purpose": "six workers with at most one S3 episode at a time",
        "system_concurrency": {"s3": 1},
        "wave_design": (
            "one_condition_and_replicate_across_all_12_development_routes_by_six_workers_"
            "with_at_most_one_s3_episode_in_flight"
        ),
        "hypothesis": (
            "GPU contention among concurrent S3 perception nodes is the entire "
            "six-worker shift seen in concurrency_shift_check_v1: clean S3 produced "
            "15 false alerts in 12 missions with inference_latency_ms beyond 0.5 SMD "
            "while clean S0 produced 2 in 12 (0.17, sequential reference 0.18) with no "
            "feature beyond bound"
        ),
        "preceding_check": {
            "campaign_id": "concurrency_shift_check_v1",
            "report": "reports/integrity/concurrency_shift_check_v1.yaml",
            "passed": False,
        },
    },
}
DEFAULT_VARIANT = "six_workers"
CAMPAIGN_ID = VARIANTS[DEFAULT_VARIANT]["campaign_id"]
SEED_BASE = VARIANTS[DEFAULT_VARIANT]["seed_base"]
OUTPUT = ROOT / "data/manifests" / f"{CAMPAIGN_ID}.yaml"
TARGETED = ROOT / "data/manifests/targeted_development_v1.yaml"
SPLITS = ROOT / "data/manifests/splits.template.yaml"
POLICY = ROOT / "configs/concurrency_shift_policy.yaml"
WORKERS = 6
REPLICATES = [0]
# Seed bases reserved by scripts/build_confirmatory_manifests.py for post-freeze work.
RESERVED_SEED_BASES = {"held_out_map_v1": 10_000_000, "severity_stress_v1": 20_000_000}
CONDITIONS = [
    {"id": "clean_s0", "family": "none", "severity": "none", "system": "s0"},
    {"id": "clean_s3", "family": "none", "severity": "none", "system": "s3"},
    {"id": "oscillation", "family": "planner_oscillation", "severity": "medium", "system": "s0"},
]


def build_manifest(
    targeted: dict, *, now_utc: str, policy_sha256: str, variant: str = DEFAULT_VARIANT,
) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    spec = VARIANTS[variant]
    campaign_id = spec["campaign_id"]
    map_routes = copy.deepcopy(targeted["design"]["map_routes"])
    routes = sum(len(items) for items in map_routes.values())
    conditions = [{**item, "replicates": list(REPLICATES)} for item in CONDITIONS]
    expected = routes * len(REPLICATES) * len(conditions)
    execution_policy = {
        "exact_once_per_episode_key": True,
        "retain_all_bags": True,
        "refuse_while_research1_confirmatory_active": True,
        "require_extraction_readiness_gate": True,
        "stop_on_first_invalid": True,
        "concurrency": WORKERS,
        "dispatcher": "scripts/run_campaign_parallel.py",
        "ros_domain_base": 60,
        "gz_partition_prefix": "research2_w",
        "maximum_episodes_per_invocation": routes,
        "minimum_free_space_gib_before_episode": 100,
        "wave_design": spec["wave_design"],
    }
    if spec["system_concurrency"] is not None:
        execution_policy["system_concurrency"] = dict(spec["system_concurrency"])
    document = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "campaign_kind": CONCURRENCY_CHECK_KIND,
        "status": "preregistered_engineering_check_not_in_fitting_pool",
        "purpose": spec["purpose"],
        "preregistered_utc": now_utc,
        "comparison_baseline": {
            "dataset_id": "balanced_pilot_v1-development-648",
            "execution": "sequential_single_simulator_ros_domain_52",
        },
        "pass_rule": {
            "policy": str(POLICY.relative_to(ROOT)),
            "policy_sha256": policy_sha256,
            "evaluator": "scripts/evaluate_concurrency_shift.py",
            "report": f"reports/integrity/{campaign_id}.yaml",
        },
        "allowed_splits": ["development"],
        "protected_test_used": False,
        "execution_policy": execution_policy,
        "episode_defaults": copy.deepcopy(targeted["episode_defaults"]),
        "infrastructure_replacements": [],
        "design": {
            "seed_base": spec["seed_base"],
            "seed_layout": "wide_v1",
            "map_routes": map_routes,
            "conditions": conditions,
            "environment_placement": copy.deepcopy(targeted["design"]["environment_placement"]),
        },
        "expected_episode_count": expected,
        "expected_per_condition": routes * len(REPLICATES),
        "admission": {
            "raw_collection_before_human_gate": "permitted_under_the_approved_dependency_boundary",
            "fitting_use": "not_admitted_engineering_evidence_only_unless_a_protocol_amendment_admits_parallel_collection",
            "protected_outcomes_consulted": False,
            "seed_collision_check": "expanded against every manifest under data/manifests and the reserved confirmatory seed bases",
        },
    }
    if "hypothesis" in spec:
        document["hypothesis"] = spec["hypothesis"]
    if "preceding_check" in spec:
        document["preceding_check"] = dict(spec["preceding_check"])
    return document


def existing_seed_sets(
    manifest_root: Path, *, exclude_campaign_id: str = CAMPAIGN_ID,
) -> dict[str, set[int]]:
    """Seeds of every existing campaign manifest, expanded as its runner would.

    ``exclude_campaign_id`` names the manifest being (re)built so a rebuilt document
    is not compared against its own file; every other variant is included.
    """
    result: dict[str, set[int]] = {}
    for path in sorted(manifest_root.glob("*.yaml")):
        if path.name == f"{exclude_campaign_id}.yaml" or path.name.endswith(".dataset.yaml"):
            continue
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            continue
        seeds: set[int] = set()
        if isinstance(document.get("design"), dict) and document["design"].get("conditions"):
            seeds |= {int(item["seed"]) for item in expand_balanced_pilot(document)}
        elif "fault_matrix" in document and "controls" in document:
            seeds |= {int(item["seed"]) for item in expand_integrity(document)}
        for item in document.get("episodes", []) or []:
            if isinstance(item, dict) and "seed" in item:
                seeds.add(int(item["seed"]))
        if seeds:
            result[path.stem] = seeds
    return result


def collision_findings(document: dict, manifest_root: Path) -> list[str]:
    findings: list[str] = []
    ours = expand_balanced_pilot(document)
    keys = [item["episode_key"] for item in ours]
    seeds = [int(item["seed"]) for item in ours]
    if len(set(keys)) != len(keys):
        findings.append("episode keys are not unique")
    if len(set(seeds)) != len(seeds):
        findings.append("seeds are not unique")
    if len(ours) != int(document["expected_episode_count"]):
        findings.append("expanded count differs from expected_episode_count")
    others = existing_seed_sets(manifest_root, exclude_campaign_id=str(document["campaign_id"]))
    for name, theirs in others.items():
        shared = set(seeds) & theirs
        if shared:
            findings.append(f"seeds collide with {name}: {sorted(shared)[:3]}")
    for name, base in RESERVED_SEED_BASES.items():
        if max(seeds) >= base:
            findings.append(f"seeds reach the reserved {name} base {base}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default=DEFAULT_VARIANT,
                        help="six_workers writes concurrency_shift_check_v1; "
                             "s3_serialised writes concurrency_shift_check_v2")
    parser.add_argument("--output", type=Path, default=None,
                        help="defaults to data/manifests/<campaign_id>.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    spec = VARIANTS[args.variant]
    campaign_id = spec["campaign_id"]
    output = args.output or ROOT / "data/manifests" / f"{campaign_id}.yaml"
    targeted = yaml.safe_load(TARGETED.read_text(encoding="utf-8"))
    document = build_manifest(
        targeted, now_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        policy_sha256=sha256_file(POLICY), variant=args.variant,
    )
    splits = yaml.safe_load(SPLITS.read_text(encoding="utf-8"))
    findings = validate_concurrency_check(document, splits) + collision_findings(
        document, ROOT / "data/manifests"
    )
    if findings:
        raise SystemExit("concurrency check plan is invalid:\n- " + "\n- ".join(findings))
    ordered = targeted_execution_order(expand_balanced_pilot(document))
    summary = {
        "variant": args.variant,
        "campaign_id": campaign_id,
        "episodes": document["expected_episode_count"],
        "conditions": [item["id"] for item in document["design"]["conditions"]],
        "seed_base": spec["seed_base"],
        "seed_range": [min(int(item["seed"]) for item in ordered), max(int(item["seed"]) for item in ordered)],
        "workers": WORKERS,
        "system_concurrency": document["execution_policy"].get("system_concurrency"),
        "waves": document["expected_episode_count"] // document["execution_policy"]["maximum_episodes_per_invocation"],
        "collision_free_against": sorted(
            existing_seed_sets(ROOT / "data/manifests", exclude_campaign_id=campaign_id)
        ),
    }
    if args.dry_run:
        print(yaml.safe_dump({"dry_run": True, **summary}, sort_keys=False))
        return 0
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    publish_new_bytes(output, yaml.safe_dump(document, sort_keys=False).encode("utf-8"))
    print(yaml.safe_dump({"written": str(output), **summary}, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
