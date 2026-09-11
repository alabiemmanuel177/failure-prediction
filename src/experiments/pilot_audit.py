"""Episode-level balanced-pilot inventory without window-level pseudoreplication."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import median


def _artifact_usable(summary: dict) -> bool:
    return (
        summary.get("outcome", {}).get("terminal_state") != "invalid"
        and summary.get("provenance", {}).get("bag_mcap_count") == 1
        and bool(summary.get("provenance", {}).get("bag_checksum_sha256"))
    )


def _replacement_index(
    summaries: list[dict], replacement_specs: list[dict] | tuple[dict, ...],
    ledger_invalid_keys: set[str] | frozenset[str] = frozenset(),
) -> tuple[dict[str, dict], list[str]]:
    """``ledger_invalid_keys`` may include replacement episode keys whose attempt failed
    artifact validation in the replacement campaign's ledger: such a replacement does not
    resolve its cell (the cell stays unresolved; it is not a manifest error)."""
    replacements: dict[str, dict] = {}
    errors: list[str] = []
    for spec in replacement_specs:
        original_key = str(spec["original_episode_key"])
        matches = [
            item for item in summaries
            if item.get("identity", {}).get("campaign_id")
            == spec.get("replacement_campaign_id")
            and item.get("identity", {}).get("episode_key")
            == spec.get("replacement_episode_key")
        ]
        if len(matches) != 1:
            errors.append(
                f"{original_key}: expected one declared replacement, found {len(matches)}"
            )
            continue
        replacement = matches[0]
        identity = replacement.get("identity", {})
        if identity.get("replacement_for_episode_key") != original_key \
                or identity.get("replacement_for_run_id") != spec.get("original_run_id"):
            errors.append(f"{original_key}: replacement identity linkage differs from manifest")
            continue
        if not _artifact_usable(replacement):
            errors.append(f"{original_key}: declared replacement is not artifact-usable")
            continue
        if str(spec.get("replacement_episode_key")) in ledger_invalid_keys:
            continue  # attempted once, failed validation; the cell remains unresolved
        replacements[original_key] = replacement
    return replacements, errors


def _allows_missing_original(spec: dict) -> bool:
    return (
        (spec.get("invalid_reason") == "ros_middleware_initialization_failure"
         and spec.get("original_bag_mcap_count") == 0)
        or (spec.get("invalid_reason") == "summary_and_bag_payload_loss_after_validation"
            and spec.get("original_bag_mcap_count") == 1)
        or (spec.get("invalid_reason")
            == "teardown_worker_isolation_failure_before_summary_publication"
            and spec.get("original_bag_mcap_count") == 1)
        or (spec.get("invalid_reason") == "platform_boundary_refusal_before_launch"
            and spec.get("original_bag_mcap_count") == 0)
    )


def summarize_pilot(
    campaign_id: str, expected: list[dict], summaries: list[dict],
    replacement_specs: list[dict] | tuple[dict, ...] = (),
    ledger_invalid_keys: set[str] | frozenset[str] = frozenset(),
) -> dict:
    """``ledger_invalid_keys``: episode keys whose recorded attempt failed post-episode
    artifact validation in the campaign ledger (the episode reached a terminal state but
    the validator rejected its artifacts, e.g. an undelivered treatment). Such originals
    are never scientific episodes; only a declared replacement resolves their cell."""
    summaries = [summary for summary in summaries if isinstance(summary, dict)]
    ledger_invalid_keys = set(ledger_invalid_keys)
    expected_keys = {episode["episode_key"] for episode in expected}
    selected = [
        summary for summary in summaries
        if summary.get("identity", {}).get("campaign_id") == campaign_id
    ]
    observed_keys = [summary.get("identity", {}).get("episode_key") for summary in selected]
    observed_counter = Counter(observed_keys)
    duplicate_keys = sorted(key for key, count in observed_counter.items() if count > 1)
    unexpected_keys = sorted(set(observed_keys) - expected_keys)

    attempt_terminal = Counter(
        summary.get("outcome", {}).get("terminal_state", "missing") for summary in selected
    )
    replacement_index, replacement_errors = _replacement_index(
        summaries, replacement_specs, ledger_invalid_keys
    )
    replacement_specs_by_key = {
        str(item["original_episode_key"]): item for item in replacement_specs
    }
    unmaterialized_replaced = {
        key for key, spec in replacement_specs_by_key.items()
        if key not in observed_counter and key in replacement_index
        and _allows_missing_original(spec)
    }
    missing_keys = sorted(expected_keys - set(observed_keys) - unmaterialized_replaced)
    attempt_terminal.update({"invalid": len(unmaterialized_replaced)})
    selected_by_key = {
        str(summary.get("identity", {}).get("episode_key")): summary
        for summary in selected if observed_counter[
            summary.get("identity", {}).get("episode_key")
        ] == 1
    }
    scientific: list[dict] = []
    replacements_used = 0
    for key in expected_keys:
        original = selected_by_key.get(key)
        if key in replacement_index and (original is not None or key in unmaterialized_replaced):
            scientific.append(replacement_index[key])
            replacements_used += 1
        elif original is not None and _artifact_usable(original) and key not in ledger_invalid_keys:
            scientific.append(original)
    terminal = Counter(
        summary.get("outcome", {}).get("terminal_state", "missing")
        for summary in scientific
    )
    families = Counter(
        summary.get("label_only", {}).get("fault_family", "missing") for summary in scientific
    )
    maps = Counter(
        summary.get("environment", {}).get("map_id", "missing") for summary in scientific
    )
    durations = [
        float(summary.get("outcome", {}).get("duration_s", 0.0)) for summary in scientific
    ]
    event_count = sum(
        summary.get("label_only", {}).get("primary_event_class") is not None
        for summary in scientific
    )
    protected = any(
        summary.get("environment", {}).get("protected_test_used") is not False
        for summary in [*selected, *replacement_index.values()]
    )
    complete = (
        len(selected) + len(unmaterialized_replaced) == len(expected)
        and not duplicate_keys and not missing_keys and not unexpected_keys
        and len(scientific) == len(expected) and not protected and not replacement_errors
    )
    return {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "analysis_unit": "episode",
        "protected_test_used": protected,
        "complete_and_artifact_valid": complete,
        "counts": {
            "expected": len(expected),
            "observed": len(selected) + len(unmaterialized_replaced),
            "usable": len(scientific),
            "ledger_artifact_invalid_attempts": len(ledger_invalid_keys & set(selected_by_key)),
            "infrastructure_invalid_attempts": sum(
                summary.get("outcome", {}).get("terminal_state") == "invalid"
                or str(summary.get("identity", {}).get("episode_key")) in replacement_index
                for summary in selected
            ) + len(unmaterialized_replaced),
            "infrastructure_replacements": replacements_used,
            "terminal_events": event_count,
            "non_events": len(scientific) - event_count,
        },
        "event_prevalence": event_count / len(scientific) if scientific else None,
        "duration_seconds": {
            "total": sum(durations),
            "median": median(durations) if durations else None,
        },
        "by_terminal_state": dict(sorted(terminal.items())),
        "by_attempt_terminal_state": dict(sorted(attempt_terminal.items())),
        "by_fault_family": dict(sorted(families.items())),
        "by_map": dict(sorted(maps.items())),
        "integrity": {
            "missing_episode_keys": missing_keys,
            "unexpected_episode_keys": unexpected_keys,
            "duplicate_episode_keys": duplicate_keys,
            "replacement_errors": replacement_errors,
        },
    }


def summarize_pilot_wave(
    *, campaign_id: str, wave: int, ledger_rows: list[dict], summaries: list[dict],
    episodes_per_wave: int = 18,
    replacement_specs: list[dict] | tuple[dict, ...] = (),
) -> dict:
    """Summarize exactly one completed ledger wave at the episode level."""

    summaries = [summary for summary in summaries if isinstance(summary, dict)]
    if wave < 1 or episodes_per_wave < 1:
        raise ValueError("wave and episodes_per_wave must be positive")
    start = (wave - 1) * episodes_per_wave
    selected_ledger = ledger_rows[start:start + episodes_per_wave]
    if len(selected_ledger) != episodes_per_wave:
        raise ValueError(
            f"wave {wave} has {len(selected_ledger)} ledger rows, expected {episodes_per_wave}"
        )
    keys = [str(row["episode_key"]) for row in selected_ledger]
    if len(set(keys)) != len(keys):
        raise ValueError(f"wave {wave} contains duplicate episode keys")
    replacement_index, replacement_errors = _replacement_index(
        summaries, replacement_specs
    )
    replacement_specs_by_key = {
        str(item["original_episode_key"]): item for item in replacement_specs
    }
    candidates: dict[str, list[dict]] = {}
    for summary in summaries:
        identity = summary.get("identity", {})
        if identity.get("campaign_id") == campaign_id:
            candidates.setdefault(str(identity.get("episode_key")), []).append(summary)
    unmaterialized_replaced = {
        key for key in keys
        if len(candidates.get(key, [])) == 0
        and key in replacement_index
        and _allows_missing_original(replacement_specs_by_key[key])
    }
    missing = [
        key for key in keys
        if len(candidates.get(key, [])) != 1 and key not in unmaterialized_replaced
    ]
    if missing:
        raise ValueError(f"wave {wave} lacks exactly one summary for: {missing}")
    attempts = [
        candidates[key][0] if key not in unmaterialized_replaced else {
            "identity": {"episode_key": key},
            "environment": {"protected_test_used": False},
            "outcome": {"terminal_state": "invalid", "duration_s": 0.0},
            "provenance": {"bag_mcap_count": 0},
            "label_only": {},
        }
        for key in keys
    ]
    ledger_status = {str(row["episode_key"]): int(row.get("returncode", 1))
                     for row in selected_ledger}
    declared_replacements = {str(item["original_episode_key"]) for item in replacement_specs}
    inconsistent_failed_rows = [
        str(item["identity"]["episode_key"])
        for item in attempts
        if ledger_status[str(item["identity"]["episode_key"])] != 0
        and item.get("outcome", {}).get("terminal_state") != "invalid"
        and str(item["identity"]["episode_key"]) not in declared_replacements
    ]
    if inconsistent_failed_rows:
        raise ValueError(
            f"wave {wave} has failed ledger rows without invalid summaries: "
            f"{inconsistent_failed_rows}"
        )
    selected: list[dict] = []
    replacements_used = 0
    unreplaced_invalid = []
    for item in attempts:
        key = str(item["identity"]["episode_key"])
        if key in replacement_index:
            selected.append(replacement_index[key])
            replacements_used += 1
        elif _artifact_usable(item):
            selected.append(item)
        else:
            unreplaced_invalid.append(key)
    if replacement_errors or unreplaced_invalid:
        raise ValueError(
            f"wave {wave} replacement errors={replacement_errors}, "
            f"unreplaced_invalid={unreplaced_invalid}"
        )
    if any(item.get("environment", {}).get("protected_test_used") is not False
           for item in [*attempts, *selected]):
        raise ValueError(f"wave {wave} contains protected or unmarked data")
    usable = selected
    terminal_states = Counter(item["outcome"]["terminal_state"] for item in selected)
    event_classes = Counter(
        item.get("label_only", {}).get("primary_event_class")
        for item in usable if item.get("label_only", {}).get("primary_event_class") is not None
    )
    durations = [float(item["outcome"]["duration_s"]) for item in selected]
    bag_bytes = 0
    for item in selected:
        bag = Path(item["provenance"]["bag_path"])
        bag_bytes += sum(path.stat().st_size for path in bag.rglob("*") if path.is_file())
    profiles = Counter(item.get("provenance", {}).get("recording_profile", "unknown")
                       for item in selected)
    report = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "wave": wave,
        "recording_profiles": dict(sorted(profiles.items())),
        "analysis_unit": "episode",
        "protected_test_used": False,
        "counts": {
            "episodes": len(selected),
            "usable": len(usable),
            "invalid": len(selected) - len(usable),
            "infrastructure_invalid_attempts": sum(
                item.get("outcome", {}).get("terminal_state") == "invalid"
                or ledger_status[str(item["identity"]["episode_key"])] != 0
                for item in attempts
            ),
            "infrastructure_replacements": replacements_used,
            "terminal_events": sum(event_classes.values()),
            "successes": terminal_states.get("success", 0),
            "collisions": terminal_states.get("collision", 0),
        },
        "event_prevalence": sum(event_classes.values()) / len(usable) if usable else None,
        "bag_gib": bag_bytes / 2**30,
        "duration_seconds": {
            "total": sum(durations),
            "median": median(durations) if durations else None,
        },
        "by_terminal_state": dict(sorted(terminal_states.items())),
        "by_event_class": dict(sorted(event_classes.items())),
        "perception_metric_summaries": sum(
            item.get("label_only", {}).get("perception_metrics") is not None
            for item in selected
        ),
        "episode_keys": keys,
        "training_admission": "human_gate_passed_dataset_campaign_in_progress",
    }
    provenance = wave_execution_provenance(selected_ledger)
    if provenance is not None:
        report["execution_provenance"] = provenance
    return report


def wave_execution_provenance(ledger_rows: list[dict]) -> dict | None:
    """Dispatcher provenance of one wave, from the ledger rows it was built from.

    Returns None for waves collected by the sequential runner (no ``dispatcher``
    field), so sequential wave reports keep their exact form. For parallel waves it
    records the worker count, the worker slots, ROS domains and Gazebo partitions
    used, and the per-system concurrency caps in force (the caps are recorded per
    row, so a wave that ran under differing caps lists every distinct block).
    """
    dispatched = [row for row in ledger_rows if row.get("dispatcher")]
    if not dispatched:
        return None
    caps: list[dict] = []
    for row in dispatched:
        block = row.get("system_concurrency")
        block = dict(sorted(block.items())) if isinstance(block, dict) else {}
        if block not in caps:
            caps.append(block)
    systems = Counter(str(row.get("system")) for row in dispatched if row.get("system") is not None)
    return {
        "dispatcher": sorted({str(row["dispatcher"]) for row in dispatched}),
        "dispatched_rows": len(dispatched),
        "sequential_rows": len(ledger_rows) - len(dispatched),
        "concurrency_workers": sorted({
            int(row["concurrency_workers"]) for row in dispatched
            if row.get("concurrency_workers") is not None
        }),
        "system_concurrency": caps[0] if len(caps) == 1 else caps,
        "worker_slots": sorted({
            int(row["worker_slot"]) for row in dispatched if row.get("worker_slot") is not None
        }),
        "ros_domain_ids": sorted({
            int(row["ros_domain_id"]) for row in dispatched if row.get("ros_domain_id") is not None
        }),
        "gz_partitions": sorted({
            str(row["gz_partition"]) for row in dispatched if row.get("gz_partition") is not None
        }),
        "episodes_by_system": dict(sorted(systems.items())),
    }
