"""Paired episode-level outcomes for guarded recovery policies."""

from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Mapping, Sequence


PAIR_FIELDS = ("map_id", "route_id", "seed", "fault_family", "severity")


def _truth(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def evaluate_paired_recovery(
    records: Sequence[Mapping[str, object]], baseline_policy: str, proposed_policy: str
) -> dict[str, object]:
    by_policy: dict[str, dict[tuple[object, ...], Mapping[str, object]]] = {
        baseline_policy: {}, proposed_policy: {},
    }
    for row in records:
        policy = str(row["policy_id"])
        if policy not in by_policy:
            continue
        key = tuple(row[field] for field in PAIR_FIELDS)
        if key in by_policy[policy]:
            raise ValueError(f"duplicate recovery outcome for {policy}/{key}")
        by_policy[policy][key] = row
    baseline_keys = set(by_policy[baseline_policy])
    proposed_keys = set(by_policy[proposed_policy])
    if baseline_keys != proposed_keys:
        missing_proposed = sorted(baseline_keys - proposed_keys, key=str)
        missing_baseline = sorted(proposed_keys - baseline_keys, key=str)
        raise ValueError(
            f"unmatched recovery pairs; missing proposed={missing_proposed}, "
            f"missing baseline={missing_baseline}"
        )
    if not baseline_keys:
        raise ValueError("no paired recovery episodes")

    def summarize(policy: str) -> dict[str, object]:
        rows = list(by_policy[policy].values())
        completion = sum(_truth(row["mission_complete"]) for row in rows)
        collisions = sum(_truth(row["collision"]) for row in rows)
        guard_violations = sum(_truth(row.get("guard_violation", False)) for row in rows)
        rejected = sum(_truth(row.get("guard_rejected", False)) for row in rows)
        times = [float(row["added_time_seconds"]) for row in rows]
        paths = [float(row["added_path_length_m"]) for row in rows]
        interventions = [int(row["intervention_count"]) for row in rows]
        actions = Counter(str(row.get("recovery_action", "none")) for row in rows)
        return {
            "episode_count": len(rows),
            "mission_completion_count": completion,
            "mission_completion_rate": completion / len(rows),
            "collision_count": collisions,
            "collision_rate": collisions / len(rows),
            "guard_violation_count": guard_violations,
            "guard_rejection_count": rejected,
            "median_added_time_seconds": median(times),
            "median_added_path_length_m": median(paths),
            "mean_intervention_count": sum(interventions) / len(interventions),
            "action_counts": dict(sorted(actions.items())),
        }

    baseline = summarize(baseline_policy)
    proposed = summarize(proposed_policy)
    return {
        "analysis_unit": "paired_map_route_seed_fault_episode",
        "pair_fields": list(PAIR_FIELDS),
        "pair_count": len(baseline_keys),
        "baseline_policy": baseline_policy,
        "proposed_policy": proposed_policy,
        "baseline": baseline,
        "proposed": proposed,
        "paired_completion_rate_difference": (
            proposed["mission_completion_rate"] - baseline["mission_completion_rate"]
        ),
        "paired_collision_rate_difference": (
            proposed["collision_rate"] - baseline["collision_rate"]
        ),
        "safety_gate_passed": proposed["guard_violation_count"] == 0,
    }
