"""Recovery-policy expansion and validation for campaign manifests (no ROS access).

A manifest that declares ``recovery_policies`` runs every base episode once per
policy with the same map/route/seed/fault pairing. Only such manifests produce
episodes carrying ``recovery_policy_id``; every other campaign expands exactly as
before, so the sequential R0 path stays unchanged.
"""

from __future__ import annotations

from collections import Counter

from src.recovery.plumbing import (
    DEFAULT_POLICY_ID, PILOT_POLICY_IDS, POLICY_IDS, validate_policy_id,
)


RECOVERY_PILOT_KIND = "recovery_pilot"
RECOVERY_PILOT_CAMPAIGN_ID = "recovery_pilot_v1"
PAIRED_RECOVERY_KIND = "paired_recovery_confirmatory"
RECOVERY_KINDS = (RECOVERY_PILOT_KIND, PAIRED_RECOVERY_KIND)
PILOT_POLICY_SET = (DEFAULT_POLICY_ID, *PILOT_POLICY_IDS)
PILOT_FAULT_FAMILIES = (
    "camera_occlusion", "lidar_dropout", "wheel_slip", "localisation_perturbation",
    "dynamic_blockage", "planner_oscillation", "semantic_corruption",
)


# Concurrency amendments that may admit the recovery pilot (PA 1.6 supersedes PA 1.4).
PARALLEL_ADMISSION_AMENDMENTS = ("PA-2026-09-10-01", "PA-2026-09-04-02", "PA-2026-09-03-04")


def declares_recovery_policies(document: dict) -> bool:
    return bool(document.get("recovery_policies"))


def is_recovery_kind(document: dict) -> bool:
    return document.get("campaign_kind") in RECOVERY_KINDS


def campaign_episodes(document: dict) -> list[dict]:
    """Every episode a manifest expands to, recovery policies included.

    Plain manifests return the balanced-pilot expansion unchanged (same objects, same
    order). A manifest that declares ``recovery_policies`` returns the per-policy
    episodes in pair-complete execution order, so summaries, inventories, monitors and
    replacement declarations count the same 882/1,512 keys the dispatcher runs.
    """
    from .campaigns import expand_balanced_pilot, targeted_execution_order

    base = expand_balanced_pilot(document)
    if not declares_recovery_policies(document):
        return base
    return expand_recovery_policies(document, targeted_execution_order(base))


def validate_paired_recovery(document: dict, splits: dict) -> list[str]:
    """Validate the protected paired recovery campaign (delegates to its builder)."""
    from scripts.build_recovery_campaign_manifest import validate_paired_recovery as _validate

    findings: list[str] = []
    if document.get("campaign_kind") != PAIRED_RECOVERY_KIND:
        findings.append(f"campaign_kind must be {PAIRED_RECOVERY_KIND}")
    if document.get("parallel_execution_admitted_by") not in PARALLEL_ADMISSION_AMENDMENTS:
        findings.append("paired recovery must cite parallel_execution_admitted_by "
                        + " or ".join(PARALLEL_ADMISSION_AMENDMENTS))
    policy = document.get("execution_policy", {}) or {}
    if policy.get("exact_once_per_episode_key") is not True or policy.get("stop_on_first_invalid") is not True:
        findings.append("paired recovery must be exact-once and stop on the first invalid episode")
    if policy.get("threshold_adaptation") != "forbidden_frozen_validation_threshold_only":
        findings.append("paired recovery must forbid threshold adaptation")
    if document.get("infrastructure_replacements"):
        findings.append("paired recovery declares replacements in a separate manifest")
    if policy.get("recovery_live_execution") is not True:
        # Without this flag the recovery manager runs in recommendation-only mode and
        # the R2/R3 arms deliver no treatment (paired_recovery_v1, 8-10 September 2026).
        findings.append("paired recovery must request live recovery execution (execution_policy.recovery_live_execution: true)")
    return findings + _validate(document, splits)


def expand_recovery_policies(document: dict, episodes: list[dict]) -> list[dict]:
    """Per-policy episodes in pair-complete order; identity when no policies declared."""
    policies = document.get("recovery_policies")
    if not policies:
        return list(episodes)
    for policy in policies:
        validate_policy_id(str(policy))
    if len(set(policies)) != len(policies):
        raise ValueError("recovery_policies must be unique")
    settings = document.get("execution_policy", {}) or {}
    live = settings.get("recovery_live_execution") is True
    selector = settings.get("recovery_selector_model")
    expanded = []
    for episode in episodes:
        for policy in policies:
            item = {
                **episode,
                "episode_key": f"{episode['episode_key']}-{policy}",
                "pair_key": episode["episode_key"],
                "recovery_policy_id": str(policy),
            }
            if live and policy != DEFAULT_POLICY_ID:
                item["recovery_live_execution"] = True
            if policy == "R3" and selector:
                item["recovery_selector_model"] = str(selector)
            expanded.append(item)
    return expanded


def pair_completeness_findings(episodes: list[dict], policies: list[str]) -> list[str]:
    findings: list[str] = []
    if len({item["episode_key"] for item in episodes}) != len(episodes):
        findings.append("episode keys are not unique")
    seed_policy = Counter((item["seed"], item["recovery_policy_id"]) for item in episodes)
    if any(count > 1 for count in seed_policy.values()):
        findings.append("seed/policy combinations are not unique")
    pairs: dict[tuple, set[str]] = {}
    for item in episodes:
        key = (item["map"], item["route"], item["seed"], item["family"], item["severity"])
        pairs.setdefault(key, set()).add(item["recovery_policy_id"])
    if any(members != set(policies) for members in pairs.values()):
        findings.append("some map/route/seed/fault cells are not paired across all policies")
    return findings


def validate_recovery_pilot(document: dict, splits: dict) -> list[str]:
    """Validate the validation-map recovery pilot that trains the R3 selector."""
    from .campaigns import expand_balanced_pilot

    findings: list[str] = []
    if document.get("campaign_kind") != RECOVERY_PILOT_KIND:
        findings.append(f"campaign_kind must be {RECOVERY_PILOT_KIND}")
    if document.get("allowed_splits") != ["validation"]:
        findings.append("recovery pilot must run on validation maps only")
    if document.get("protected_test_used") is not False:
        findings.append("recovery pilot must declare protected_test_used false")
    policies = list(document.get("recovery_policies") or [])
    if policies != list(PILOT_POLICY_SET):
        findings.append(
            "recovery pilot policies must be R0 plus the six forced actions "
            f"{list(PILOT_POLICY_IDS)}"
        )
    unknown = [policy for policy in policies if policy not in POLICY_IDS]
    if unknown:
        findings.append(f"unknown recovery policies: {unknown}")
    split = splits.get("validation", {})
    allowed_maps = set(split.get("maps", []))
    allowed_routes = set(split.get("routes", []))
    design = document.get("design", {})
    map_routes = design.get("map_routes", {})
    if len(map_routes) != 3:
        findings.append("recovery pilot must cover the three validation maps")
    for map_id, routes in map_routes.items():
        if map_id not in allowed_maps:
            findings.append(f"map is outside validation split: {map_id}")
        if len(routes) != 6:
            findings.append(f"recovery pilot needs six routes on {map_id}")
        for route in routes:
            if route not in allowed_routes:
                findings.append(f"route is outside validation split: {route}")
            if not route.startswith(f"{map_id}_"):
                findings.append(f"route/map mismatch: {map_id}/{route}")
    conditions = design.get("conditions", [])
    families = sorted(item.get("family") for item in conditions)
    if families != sorted(PILOT_FAULT_FAMILIES):
        findings.append("recovery pilot must contain the seven fault families exactly once")
    if any(item.get("severity") != "medium" for item in conditions):
        findings.append("recovery pilot conditions must use medium severity")
    if any(list(item.get("replicates", [])) != [0] for item in conditions):
        findings.append("recovery pilot uses exactly one seed per cell (replicates [0])")
    if design.get("seed_layout") != "wide_v1":
        findings.append("recovery pilot requires collision-safe wide_v1 seeds")
    policy = document.get("execution_policy", {})
    if int(policy.get("concurrency", 1)) != 6:
        findings.append("recovery pilot declares execution_policy.concurrency 6")
    if document.get("parallel_execution_admitted_by") not in PARALLEL_ADMISSION_AMENDMENTS:
        findings.append(
            "recovery pilot must cite parallel_execution_admitted_by "
            + " or ".join(PARALLEL_ADMISSION_AMENDMENTS)
        )
    if policy.get("exact_once_per_episode_key") is not True:
        findings.append("recovery pilot must be exact-once per episode key")
    if policy.get("stop_on_first_invalid") is not True:
        findings.append("recovery pilot must stop on the first invalid episode")
    if policy.get("threshold_adaptation") != "forbidden_frozen_validation_threshold_only":
        findings.append("recovery pilot must forbid threshold adaptation")
    if policy.get("recovery_live_execution") is not True:
        findings.append("recovery pilot must request live recovery execution (guarded, evidence-gated)")
    if document.get("infrastructure_replacements"):
        findings.append("recovery pilot declares replacements in a separate manifest")
    try:
        base = expand_balanced_pilot(document)
    except (KeyError, TypeError, ValueError) as error:
        findings.append(f"design does not expand: {error}")
        return findings
    if len(base) != int(document.get("expected_base_episode_count", -1)):
        findings.append(
            f"expanded {len(base)} base episodes, expected {document.get('expected_base_episode_count')}"
        )
    episodes = expand_recovery_policies(document, base) if policies else base
    if len(episodes) != int(document.get("expected_episode_count", -1)):
        findings.append(
            f"expanded {len(episodes)} episodes, expected {document.get('expected_episode_count')}"
        )
    if policies:
        findings.extend(pair_completeness_findings(episodes, policies))
    return findings


__all__ = [
    "PAIRED_RECOVERY_KIND", "PILOT_FAULT_FAMILIES", "PILOT_POLICY_SET",
    "RECOVERY_KINDS", "RECOVERY_PILOT_CAMPAIGN_ID", "RECOVERY_PILOT_KIND",
    "declares_recovery_policies", "expand_recovery_policies", "pair_completeness_findings",
    "validate_recovery_pilot",
]
