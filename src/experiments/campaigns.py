"""Pure manifest expansion helpers; no simulator or protected-data access."""

from __future__ import annotations

from pathlib import Path

import yaml

from collections import Counter


def expand_balanced_pilot(document: dict) -> list[dict]:
    design = document["design"]
    defaults = document["episode_defaults"]
    episodes = []
    for map_index, (map_id, routes) in enumerate(design["map_routes"].items()):
        for route_index, route_id in enumerate(routes):
            for condition_index, condition in enumerate(design["conditions"]):
                replicates = condition.get("replicates", design.get("replicates", []))
                for replicate in replicates:
                    family = condition["family"]
                    if design.get("seed_layout") == "wide_v1":
                        seed = (
                            int(design["seed_base"])
                            + map_index * 1_000_000
                            + route_index * 100_000
                            + int(replicate) * 100
                            + condition_index
                        )
                    else:
                        seed = (
                            int(design["seed_base"])
                            + map_index * 10_000
                            + route_index * 1_000
                            + int(replicate) * 100
                            + condition_index
                        )
                    episode = {
                        **defaults,
                        "episode_key": (
                            f"{map_id}-{route_id}-{condition['id']}-r{replicate}"
                        ),
                        "map": map_id,
                        "route": route_id,
                        "system": condition["system"],
                        "family": family,
                        "severity": condition["severity"],
                        "seed": seed,
                    }
                    if family in {"dynamic_blockage", "planner_oscillation"}:
                        episode.update(design["environment_placement"])
                    episodes.append(episode)
    return episodes


def balanced_execution_order(episodes: list[dict]) -> list[dict]:
    """Create 18-run waves: every condition twice, spanning two maps/routes."""
    by_key = {
        (item["map"], item["route"], int(item["episode_key"].rsplit("-r", 1)[1]),
         item["episode_key"].rsplit("-", 2)[-2]): item
        for item in episodes
    }
    route_pairs = sorted({(item["map"], item["route"]) for item in episodes})
    replicates = sorted({int(item["episode_key"].rsplit("-r", 1)[1]) for item in episodes})
    conditions = sorted({item["episode_key"].rsplit("-", 2)[-2] for item in episodes})
    if len(route_pairs) % 2:
        raise ValueError("balanced execution ordering requires an even number of routes")
    half = len(route_pairs) // 2
    ordered = []
    for replicate in replicates:
        for wave in range(half):
            for condition in conditions:
                for map_id, route_id in (route_pairs[wave], route_pairs[wave + half]):
                    ordered.append(by_key[(map_id, route_id, replicate, condition)])
    if len(ordered) != len(episodes) or {id(item) for item in ordered} != {
        id(item) for item in episodes
    }:
        raise ValueError("balanced execution ordering lost or duplicated episodes")
    return ordered


def targeted_execution_order(episodes: list[dict]) -> list[dict]:
    """Order one family/replicate across all 12 routes per exact-once wave."""
    def parts(item: dict) -> tuple[str, int, str, str]:
        condition = item["episode_key"].rsplit("-", 2)[-2]
        replicate = int(item["episode_key"].rsplit("-r", 1)[1])
        return condition, replicate, item["map"], item["route"]

    ordered = sorted(episodes, key=parts)
    if len(ordered) != len(episodes) or len({item["episode_key"] for item in ordered}) != len(episodes):
        raise ValueError("targeted execution ordering lost or duplicated episodes")
    return ordered


def validate_targeted_campaign(document: dict, splits: dict) -> list[str]:
    """Validate the frozen post-pilot family-targeted development design."""
    findings = validate_balanced_pilot(document, splits)
    balanced_only = {
        "pilot must contain exactly two clean control conditions",
        "pilot must contain exactly the seven frozen fault families",
        "expanded condition counts do not match expected_per_condition",
    }
    findings = [item for item in findings if item not in balanced_only]
    if document.get("campaign_kind") != "targeted_event_floor":
        findings.append("campaign_kind must be targeted_event_floor")
    if document.get("allowed_splits") != ["development"]:
        findings.append("targeted additions must remain development-only")
    design = document.get("design", {})
    if design.get("seed_layout") != "wide_v1":
        findings.append("targeted additions require collision-safe wide_v1 seeds")
    conditions = design.get("conditions", [])
    if any(not item.get("replicates") for item in conditions):
        findings.append("every targeted condition requires an explicit replicate list")
    expected_faults = {
        "camera_occlusion", "lidar_dropout", "wheel_slip",
        "localisation_perturbation", "dynamic_blockage", "semantic_corruption",
    }
    if {item.get("family") for item in conditions} != expected_faults:
        findings.append("targeted design must contain the six pilot-deficit families")
    if int(document.get("expected_episode_count", -1)) != 1212:
        findings.append("targeted design must contain the frozen 1,212 additions")
    if int(document.get("execution_policy", {}).get("maximum_episodes_per_invocation", -1)) != 12:
        findings.append("targeted exact-once waves must cover all 12 development routes")
    return findings


def validate_development_supplement(document: dict, splits: dict) -> list[str]:
    """Validate the post-adapter-audit route-balanced development supplement.

    The supplement reuses the targeted exact-once 12-route wave design but its
    per-condition replicate counts may differ by at most one, so the equal-count and
    fixed-total rules of the targeted validator are replaced by the sizing declared in
    the manifest itself.
    """
    findings = validate_balanced_pilot(document, splits)
    balanced_only = {
        "pilot must contain exactly two clean control conditions",
        "pilot must contain exactly the seven frozen fault families",
        "expanded condition counts do not match expected_per_condition",
    }
    findings = [item for item in findings if item not in balanced_only]
    if document.get("campaign_kind") != "development_supplement":
        findings.append("campaign_kind must be development_supplement")
    if document.get("allowed_splits") != ["development"]:
        findings.append("supplement additions must remain development-only")
    if document.get("protected_test_used") is not False:
        findings.append("supplement must declare protected_test_used false")
    design = document.get("design", {})
    if design.get("seed_layout") != "wide_v1":
        findings.append("supplement requires collision-safe wide_v1 seeds")
    conditions = design.get("conditions", [])
    if any(not item.get("replicates") for item in conditions):
        findings.append("every supplement condition requires an explicit replicate list")
    families = Counter(item.get("family") for item in conditions)
    expected_faults = {
        "camera_occlusion", "lidar_dropout", "wheel_slip", "localisation_perturbation",
        "dynamic_blockage", "planner_oscillation", "semantic_corruption",
    }
    if set(families) - {"none"} != expected_faults or families.get("none", 0) != 2:
        findings.append("supplement must cover the seven fault families and two clean controls")
    counts = sorted(len(item["replicates"]) for item in conditions if item.get("replicates"))
    if counts and counts[-1] - counts[0] > 1:
        findings.append("supplement replicate counts may differ by at most one")
    sizing = document.get("sizing", {})
    expected = int(document.get("expected_episode_count", -1))
    routes = sum(len(routes) for routes in design.get("map_routes", {}).values())
    if expected != sum(counts) * routes:
        findings.append("expected_episode_count does not match the replicate design")
    if int(sizing.get("routes_per_block", -1)) != routes or int(sizing.get("complete_blocks", -1)) * routes != expected:
        findings.append("sizing block accounting does not match the design")
    if expected < int(sizing.get("minimum_supplement", 324)):
        findings.append("supplement is smaller than the preregistered minimum")
    if int(document.get("execution_policy", {}).get("maximum_episodes_per_invocation", -1)) != routes:
        findings.append("supplement exact-once waves must cover all development routes")
    return findings


def validate_balanced_pilot(document: dict, splits: dict) -> list[str]:
    """Validate a balanced development or validation campaign without touching ROS."""
    findings: list[str] = []
    if document.get("protected_test_used") is not False:
        findings.append("protected_test_used must be false")
    allowed_splits = document.get("allowed_splits")
    if allowed_splits not in (["development"], ["validation"]):
        findings.append("allowed_splits must contain exactly development or validation")
    split_name = (
        allowed_splits[0]
        if allowed_splits in (["development"], ["validation"])
        else "development"
    )

    design = document.get("design", {})
    map_routes = design.get("map_routes", {})
    split = splits.get(split_name, {})
    allowed_maps = set(split.get("maps", []))
    allowed_routes = set(split.get("routes", []))
    for map_id, routes in map_routes.items():
        if map_id not in allowed_maps:
            findings.append(f"map is outside {split_name} split: {map_id}")
        for route_id in routes:
            if route_id not in allowed_routes:
                findings.append(f"route is outside {split_name} split: {route_id}")
            if not route_id.startswith(f"{map_id}_"):
                findings.append(f"route/map mismatch: {map_id}/{route_id}")

    defaults = document.get("episode_defaults", {})
    clean_prefix = float(defaults.get("clean_prefix_seconds", -1))
    onset = float(defaults.get("planned_onset_seconds", -1))
    duration = float(defaults.get("maximum_duration_seconds", -1))
    if clean_prefix < 0 or onset < clean_prefix:
        findings.append("planned onset must follow the non-negative clean prefix")
    if duration <= 0:
        findings.append("maximum duration must be positive")

    conditions = design.get("conditions", [])
    condition_ids = [condition.get("id") for condition in conditions]
    if len(condition_ids) != len(set(condition_ids)):
        findings.append("condition ids must be unique")
    family_counts = Counter(condition.get("family") for condition in conditions)
    if family_counts.get("none") != 2:
        findings.append("pilot must contain exactly two clean control conditions")
    expected_faults = {
        "camera_occlusion", "lidar_dropout", "wheel_slip",
        "localisation_perturbation", "dynamic_blockage",
        "planner_oscillation", "semantic_corruption",
    }
    if set(family_counts) - {"none"} != expected_faults:
        findings.append("pilot must contain exactly the seven frozen fault families")

    placement = design.get("environment_placement", {})
    if placement.get("placement_mode") != "path_fraction":
        findings.append("environment faults must use path_fraction placement")
    fraction = float(placement.get("route_fraction", -1))
    if not 0.0 < fraction < 1.0:
        findings.append("environment route_fraction must lie strictly between 0 and 1")

    episodes = expand_balanced_pilot(document)
    expected = int(document.get("expected_episode_count", -1))
    if len(episodes) != expected:
        findings.append(f"expanded {len(episodes)} episodes, expected {expected}")
    if len({item["episode_key"] for item in episodes}) != len(episodes):
        findings.append("pilot episode keys are not unique")
    if len({item["seed"] for item in episodes}) != len(episodes):
        findings.append("pilot seeds are not unique")
    episode_keys = {item["episode_key"] for item in episodes}
    replacements = document.get("infrastructure_replacements", [])
    original_keys = [item.get("original_episode_key") for item in replacements]
    if len(original_keys) != len(set(original_keys)):
        findings.append("infrastructure replacement original keys must be unique")
    for replacement in replacements:
        original = replacement.get("original_episode_key")
        if original not in episode_keys:
            findings.append(f"replacement original is outside pilot design: {original}")
        if replacement.get("replacement_campaign_id") == document.get("campaign_id"):
            findings.append("infrastructure replacement must use a distinct campaign")
        if replacement.get("replacement_episode_key") in episode_keys:
            findings.append("infrastructure replacement key collides with pilot design")
        if replacement.get("retain_original_attempt") is not True:
            findings.append("infrastructure replacement must retain the original attempt")
        if replacement.get("scientific_episode_count_contribution") != 1:
            findings.append("infrastructure replacement must contribute exactly one design cell")
        reason = replacement.get("invalid_reason")
        mcap_count = replacement.get("original_bag_mcap_count")
        permitted = (
            (reason == "missing_mandatory_topic_before_goal" and mcap_count == 0)
            or (reason == "missing_injection_started_event" and mcap_count == 1)
            or (reason == "ros_middleware_initialization_failure" and mcap_count == 0)
            or (reason == "summary_and_bag_payload_loss_after_validation" and mcap_count == 1)
            or (
                reason == "teardown_worker_isolation_failure_before_summary_publication"
                and mcap_count == 1
            )
        )
        if not permitted:
            findings.append(
                "replacement must be a declared zero-bag startup invalid, retained "
                "one-bag treatment-delivery/teardown invalid, or post-validation payload loss"
            )
    per_condition = Counter(
        item["episode_key"].rsplit("-", 1)[0].rsplit("-", 1)[-1]
        for item in episodes
    )
    expected_per_condition = int(document.get("expected_per_condition", -1))
    if per_condition and set(per_condition.values()) != {expected_per_condition}:
        findings.append("expanded condition counts do not match expected_per_condition")
    return findings


def validate_confirmatory_campaign(document: dict, splits: dict) -> list[str]:
    """Validate a post-freeze held-out-map campaign without touching ROS or bags.

    The design reuses the balanced expansion (`expand_balanced_pilot`) and the
    targeted execution order, so every exact-once wave covers one condition and
    replicate across all frozen held-out routes.
    """
    findings: list[str] = []
    if document.get("campaign_kind") != "held_out_confirmatory":
        findings.append("campaign_kind must be held_out_confirmatory")
    if document.get("allowed_splits") != ["held_out_map_test"]:
        findings.append("confirmatory campaigns must declare allowed_splits [held_out_map_test]")
    if document.get("protected_test_used") is not True:
        findings.append("confirmatory campaigns must declare protected_test_used: true")
    split = splits.get("held_out_map_test", {}) or {}
    if split.get("status") != "assigned_after_model_freeze":
        findings.append("held_out_map_test split is not assigned after the model freeze")
    allowed_maps = set(split.get("maps") or [])
    allowed_routes = set(split.get("routes") or [])
    design = document.get("design", {})
    map_routes = design.get("map_routes", {})
    for map_id, routes in map_routes.items():
        if map_id not in allowed_maps:
            findings.append(f"map is outside held_out_map_test split: {map_id}")
        for route_id in routes:
            if route_id not in allowed_routes:
                findings.append(f"route is outside held_out_map_test split: {route_id}")
            if not route_id.startswith(f"{map_id}_"):
                findings.append(f"route/map mismatch: {map_id}/{route_id}")
    if allowed_maps and set(map_routes) != allowed_maps:
        findings.append("confirmatory design must cover every assigned held-out map")
    designed_routes = {route for routes in map_routes.values() for route in routes}
    if allowed_routes and designed_routes != allowed_routes:
        findings.append("confirmatory design must cover every frozen held-out route")
    if design.get("seed_layout") != "wide_v1":
        findings.append("confirmatory campaigns require collision-safe wide_v1 seeds")

    defaults = document.get("episode_defaults", {})
    clean_prefix = float(defaults.get("clean_prefix_seconds", -1))
    onset = float(defaults.get("planned_onset_seconds", -1))
    duration = float(defaults.get("maximum_duration_seconds", -1))
    if clean_prefix < 0 or onset < clean_prefix:
        findings.append("planned onset must follow the non-negative clean prefix")
    if duration <= 0:
        findings.append("maximum duration must be positive")

    conditions = design.get("conditions", [])
    condition_ids = [condition.get("id") for condition in conditions]
    if len(condition_ids) != len(set(condition_ids)):
        findings.append("condition ids must be unique")
    if any(not item.get("replicates") for item in conditions):
        findings.append("every confirmatory condition requires an explicit replicate list")
    expected_faults = {
        "camera_occlusion", "lidar_dropout", "wheel_slip",
        "localisation_perturbation", "dynamic_blockage",
        "planner_oscillation", "semantic_corruption",
    }
    families = {item.get("family") for item in conditions}
    if families - {"none"} != expected_faults:
        findings.append("confirmatory design must contain exactly the seven frozen fault families")
    for item in conditions:
        if item.get("family") == "none" and item.get("severity") != "none":
            findings.append("clean control conditions must declare severity none")
        if item.get("family") != "none" and item.get("severity") not in {"low", "medium", "high"}:
            findings.append(f"condition {item.get('id')} has an unknown severity")
    placement = design.get("environment_placement", {})
    if placement.get("placement_mode") != "path_fraction":
        findings.append("environment faults must use path_fraction placement")
    fraction = float(placement.get("route_fraction", -1))
    if not 0.0 < fraction < 1.0:
        findings.append("environment route_fraction must lie strictly between 0 and 1")

    episodes = expand_balanced_pilot(document) if map_routes and conditions else []
    expected = int(document.get("expected_episode_count", -1))
    if len(episodes) != expected:
        findings.append(f"expanded {len(episodes)} episodes, expected {expected}")
    if len({item["episode_key"] for item in episodes}) != len(episodes):
        findings.append("confirmatory episode keys are not unique")
    if len({item["seed"] for item in episodes}) != len(episodes):
        findings.append("confirmatory seeds are not unique")
    policy = document.get("execution_policy", {})
    wave = int(policy.get("maximum_episodes_per_invocation", -1))
    route_count = sum(len(routes) for routes in map_routes.values())
    if wave != route_count or route_count <= 0:
        findings.append("confirmatory exact-once waves must cover all held-out routes")
    if policy.get("exact_once_per_episode_key") is not True:
        findings.append("confirmatory campaigns must be exact-once per episode key")
    if policy.get("require_confirmatory_readiness_gate") is not True:
        findings.append("confirmatory campaigns must require the confirmatory readiness gate")
    if document.get("infrastructure_replacements"):
        findings.append("confirmatory campaigns declare replacements in a separate manifest")
    return findings


CONCURRENCY_CHECK_KIND = "concurrency_check"
CONCURRENCY_CHECK_CONDITIONS = (
    ("none", "none", "s0"),
    ("none", "none", "s3"),
    ("planner_oscillation", "medium", "s0"),
)
AMENDMENT_ID_PATTERN = r"^PA-\d{4}-\d{2}-\d{2}-\d{2}$"
SYSTEM_ID_PATTERN = r"^s\d+$"


def system_concurrency_findings(block: object, *, where: str) -> list[str]:
    """Format findings for a ``system_concurrency`` cap block (``{s3: 1}``).

    A cap block maps lower-case system ids to positive integers: the largest number
    of episodes of that system that may run at the same time. It is shared by the
    concurrency-check manifest (``execution_policy.system_concurrency``), a protocol
    amendment (``decisions.concurrency.system_concurrency``) and the dispatcher's
    ``--max-system-concurrency`` flag.
    """
    import re

    if block is None:
        return []
    if not isinstance(block, dict):
        return [f"{where} must be a mapping of system id to a positive integer cap"]
    findings: list[str] = []
    for system, cap in block.items():
        if not isinstance(system, str) or not re.match(SYSTEM_ID_PATTERN, system):
            findings.append(f"{where}: system id must look like s0..s9, got {system!r}")
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            findings.append(f"{where}: cap for {system!r} must be a positive integer")
    return findings


def normalise_system_concurrency(block: object) -> dict[str, int]:
    """Lower-cased, integer-valued copy of a cap block that already validated."""
    if not isinstance(block, dict):
        return {}
    return {str(system).lower(): int(cap) for system, cap in block.items()}


def validate_concurrency_check(document: dict, splits: dict) -> list[str]:
    """Validate the sequential-versus-parallel health-feature shift check.

    The check reuses the targeted exact-once 12-route wave design and the
    development-supplement admission rules, but its condition set is fixed to the two
    clean controls and the medium planner-oscillation environment fault, and it must
    declare a multi-worker execution policy: the whole point of the campaign is to
    measure what concurrent simulators do to the deployable health channels.
    """
    findings = validate_balanced_pilot(document, splits)
    balanced_only = {
        "pilot must contain exactly two clean control conditions",
        "pilot must contain exactly the seven frozen fault families",
        "expanded condition counts do not match expected_per_condition",
    }
    findings = [item for item in findings if item not in balanced_only]
    if document.get("campaign_kind") != CONCURRENCY_CHECK_KIND:
        findings.append(f"campaign_kind must be {CONCURRENCY_CHECK_KIND}")
    if document.get("allowed_splits") != ["development"]:
        findings.append("concurrency check must remain development-only")
    if document.get("protected_test_used") is not False:
        findings.append("concurrency check must declare protected_test_used false")
    design = document.get("design", {})
    if design.get("seed_layout") != "wide_v1":
        findings.append("concurrency check requires collision-safe wide_v1 seeds")
    conditions = design.get("conditions", [])
    if any(not item.get("replicates") for item in conditions):
        findings.append("every concurrency-check condition requires an explicit replicate list")
    cells = Counter(
        (item.get("family"), item.get("severity"), item.get("system")) for item in conditions
    )
    if cells != Counter(CONCURRENCY_CHECK_CONDITIONS):
        findings.append(
            "concurrency check must contain exactly clean s0, clean s3 and medium "
            "planner_oscillation s0 conditions"
        )
    counts = sorted(len(item["replicates"]) for item in conditions if item.get("replicates"))
    if counts and counts[-1] != counts[0]:
        findings.append("concurrency-check conditions must share one replicate count")
    routes = sum(len(routes) for routes in design.get("map_routes", {}).values())
    expected = int(document.get("expected_episode_count", -1))
    if expected != sum(counts) * routes:
        findings.append("expected_episode_count does not match the replicate design")
    policy = document.get("execution_policy", {})
    if int(policy.get("maximum_episodes_per_invocation", -1)) != routes:
        findings.append("concurrency-check exact-once waves must cover all development routes")
    try:
        concurrency = int(policy.get("concurrency", 1))
    except (TypeError, ValueError):
        concurrency = 1
    if concurrency < 2:
        findings.append("concurrency check must declare execution_policy.concurrency of at least 2")
    if policy.get("exact_once_per_episode_key") is not True:
        findings.append("concurrency check must be exact-once per episode key")
    if policy.get("stop_on_first_invalid") is not True:
        findings.append("concurrency check must stop on the first invalid episode")
    caps = policy.get("system_concurrency")
    cap_findings = system_concurrency_findings(
        caps, where="execution_policy.system_concurrency",
    )
    findings.extend(cap_findings)
    if caps is not None and not cap_findings:
        designed_systems = {str(item.get("system")).lower() for item in conditions}
        for system, cap in normalise_system_concurrency(caps).items():
            if system not in designed_systems:
                findings.append(
                    f"execution_policy.system_concurrency caps {system}, which no condition uses"
                )
            if cap > concurrency:
                findings.append(
                    f"execution_policy.system_concurrency cap for {system} ({cap}) exceeds "
                    f"the declared concurrency {concurrency}"
                )
    if document.get("infrastructure_replacements"):
        findings.append("concurrency check declares replacements in a separate manifest")
    return findings


def parallel_admission_terms(amendment: dict) -> dict:
    """Concurrency terms of an approved amendment (PA-2026-09-03-04 schema).

    Returns ``campaigns`` (admitted campaign ids), ``admitted_workers`` (or None),
    ``ros_domains`` (or None), ``admission_condition`` (``<report path> passed true``
    or None), ``system_concurrency`` (the amendment's per-system cap block, or
    None) and ``phased_execution`` (the per-system phase worker caps, ``{s0: 6,
    s3: 1}``, or None). The flat ``decisions.parallel_execution_admitted_campaigns``
    form is accepted as well. Nothing here is tied to one check report: whichever
    report path the amendment names is the one the dispatcher loads and requires to
    pass.
    """
    decisions = amendment.get("decisions") if isinstance(amendment.get("decisions"), dict) else {}
    concurrency = decisions.get("concurrency") if isinstance(decisions.get("concurrency"), dict) else {}
    campaigns = concurrency.get("campaigns_admitted_if_check_passes")
    if not isinstance(campaigns, list):
        campaigns = decisions.get("parallel_execution_admitted_campaigns")
    return {
        "campaigns": list(campaigns) if isinstance(campaigns, list) else [],
        "admitted_workers": concurrency.get("admitted_workers"),
        "ros_domains": concurrency.get("ros_domains"),
        "admission_condition": concurrency.get("admission_condition"),
        "system_concurrency": concurrency.get("system_concurrency"),
        "phased_execution": concurrency.get("phased_execution"),
    }


def admission_condition_report_path(condition: object) -> str | None:
    """Report path named by an ``<path> passed true`` admission condition."""
    if not isinstance(condition, str) or not condition.strip():
        return None
    return condition.split()[0]


def validate_parallel_execution_admission(
    document: dict, amendments: dict | None = None, *, workers: int | None = None,
    check_reports: dict | None = None, systems: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Validate ``parallel_execution_admitted_by`` for a sequential-policy manifest.

    A manifest whose ``execution_policy.concurrency`` is 1 was preregistered for
    sequential collection. Running it with several workers changes the compute-load
    and timing conditions that are candidate health signals, so it needs an approved
    protocol amendment naming the campaign (``decisions.concurrency.
    campaigns_admitted_if_check_passes`` as in PA-2026-09-03-04). When the amendment
    declares an ``admission_condition`` of the form ``<report> passed true`` the named
    report (any path, e.g. the v1 or the v2 shift check) must be supplied in
    ``check_reports`` with ``passed: true``; when it declares ``admitted_workers`` the
    requested ``workers`` may not exceed it; when it declares
    ``decisions.concurrency.system_concurrency`` the block must be well formed (the
    dispatcher then enforces it as a floor of restrictions for admitted campaigns).

    An optional ``decisions.concurrency.phased_execution`` block (``{s0: 6, s3: 1}``)
    admits running the campaign one system at a time: phase A claims only the
    episodes of one system with at most that system's worker cap, phase B the next.
    When the dispatcher restricts claims with ``--systems`` it passes them here as
    ``systems``: every requested system must have a declared phase cap and
    ``workers`` may not exceed the smallest cap among the requested systems. A
    ``--systems`` run against an amendment without a phased block is refused, and no
    phase cap may exceed ``admitted_workers``. ``amendments`` maps amendment ids to
    their loaded documents; when it is None only the field format is checked.
    """
    import re

    findings: list[str] = []
    amendment_id = document.get("parallel_execution_admitted_by")
    if not isinstance(amendment_id, str) or not re.match(AMENDMENT_ID_PATTERN, amendment_id):
        findings.append(
            "parallel_execution_admitted_by must name one protocol amendment id (PA-YYYY-MM-DD-NN)"
        )
        return findings
    if amendments is None:
        return findings
    amendment = amendments.get(amendment_id)
    if not isinstance(amendment, dict):
        findings.append(f"protocol amendment {amendment_id} is not on file under configs/")
        return findings
    if amendment.get("amendment_id") != amendment_id:
        findings.append(f"amendment file for {amendment_id} declares a different amendment_id")
    if amendment.get("status") != "approved":
        findings.append(f"protocol amendment {amendment_id} is not approved")
    terms = parallel_admission_terms(amendment)
    if document.get("campaign_id") not in terms["campaigns"]:
        findings.append(
            f"protocol amendment {amendment_id} does not admit {document.get('campaign_id')!r} "
            "to parallel execution"
        )
    report_path = admission_condition_report_path(terms["admission_condition"])
    if report_path is not None:
        report = (check_reports or {}).get(report_path)
        if not isinstance(report, dict) or report.get("passed") is not True:
            findings.append(
                f"admission condition of {amendment_id} is not met: {terms['admission_condition']}"
            )
    admitted_workers = terms["admitted_workers"]
    if workers is not None and isinstance(admitted_workers, int) and workers > admitted_workers:
        findings.append(
            f"requested {workers} workers exceeds the {admitted_workers} admitted by {amendment_id}"
        )
    findings.extend(system_concurrency_findings(
        terms["system_concurrency"],
        where=f"{amendment_id} decisions.concurrency.system_concurrency",
    ))
    findings.extend(phased_execution_findings(
        terms["phased_execution"], amendment_id=amendment_id,
        admitted_workers=admitted_workers, workers=workers, systems=systems,
    ))
    return findings


def phased_execution_findings(
    block: object, *, amendment_id: str, admitted_workers: object = None,
    workers: int | None = None, systems: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Findings for an amendment's ``phased_execution`` block and a requested phase.

    ``block`` is the ``{s0: 6, s3: 1}`` phase-cap mapping (or None when the amendment
    declares no phased execution); ``systems`` the systems a dispatcher run is
    restricted to (None for an unrestricted run, which the block never constrains).
    """
    where = f"{amendment_id} decisions.concurrency.phased_execution"
    findings = system_concurrency_findings(block, where=where)
    caps = normalise_system_concurrency(block) if not findings else {}
    if isinstance(admitted_workers, int) and not isinstance(admitted_workers, bool):
        for system, cap in caps.items():
            if cap > admitted_workers:
                findings.append(
                    f"{where}: phase cap for {system} ({cap}) exceeds the "
                    f"{admitted_workers} admitted workers"
                )
    if systems is None:
        return findings
    requested = sorted({str(system).lower() for system in systems})
    if not requested:
        findings.append("a phased run must name at least one system")
        return findings
    if block is None:
        findings.append(
            f"{amendment_id} declares no decisions.concurrency.phased_execution; a run "
            f"restricted to {', '.join(requested)} is not admitted"
        )
        return findings
    if findings:
        return findings
    for system in requested:
        if system not in caps:
            findings.append(f"{where} declares no phase for {system}")
    if workers is not None:
        declared = [caps[system] for system in requested if system in caps]
        if declared and workers > min(declared):
            tightest = min(requested, key=lambda system: caps.get(system, workers + 1))
            findings.append(
                f"requested {workers} workers exceeds the phase cap {caps[tightest]} for "
                f"{tightest} in {amendment_id}"
            )
    return findings


def declared_replacements(root: Path, document: dict) -> list[dict]:
    """Infrastructure replacement declarations for a campaign.

    Preprotected campaigns declare them inline; confirmatory and concurrency-check
    campaigns must declare them in a separate replacement manifest whose
    ``parent_campaign_id`` names the campaign. Both sources are merged here so every
    consumer (reports, inventories, audits) sees the same declarations.
    """
    specs = list(document.get("infrastructure_replacements", []) or [])
    campaign_id = document.get("campaign_id")
    for path in sorted((root / "data/manifests").glob("*.yaml")):
        try:
            other = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - non-manifest YAML is ignored
            continue
        if isinstance(other, dict) and other.get("parent_campaign_id") == campaign_id \
                and other.get("campaign_id") != campaign_id:
            specs.extend(other.get("infrastructure_replacements", []) or [])
    return specs


def ledger_invalid_episode_keys(root: Path, manifest: dict) -> set[str]:
    """Episode keys (parent campaign and its declared replacement campaigns) whose
    recorded attempt has a non-zero return code in the campaign ledger: the episode
    process failed, or the post-episode artifact validator rejected the recording."""
    import json

    campaign_ids = {str(manifest["campaign_id"])}
    for spec in declared_replacements(root, manifest):
        if spec.get("replacement_campaign_id"):
            campaign_ids.add(str(spec["replacement_campaign_id"]))
    invalid: set[str] = set()
    for campaign_id in sorted(campaign_ids):
        ledger = root / "logs/campaigns" / f"{campaign_id}.jsonl"
        if not ledger.exists():
            continue
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("returncode", 0) != 0:
                    invalid.add(str(row["episode_key"]))
    return invalid
