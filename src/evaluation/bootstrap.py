"""Hierarchical paired bootstrap over maps, routes, then independent episodes."""

from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import Mapping, Sequence


def _truth(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def hierarchical_paired_binary_bootstrap(
    records: Sequence[Mapping[str, object]],
    outcome_a: str,
    outcome_b: str,
    *,
    include_field: str | None = None,
    replicates: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260824,
) -> dict[str, object]:
    """Estimate A-minus-B while preserving paired episode comparisons."""
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    selected = [record for record in records if include_field is None or _truth(record[include_field])]
    if not selected:
        raise ValueError("no eligible paired records")
    identities = [(row["map_id"], row["route_id"], row["episode_id"]) for row in selected]
    if len(identities) != len(set(identities)):
        raise ValueError("paired records must be unique by map, route, and episode")

    hierarchy: dict[object, dict[object, list[Mapping[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    differences = []
    for row in selected:
        hierarchy[row["map_id"]][row["route_id"]].append(row)
        differences.append(float(_truth(row[outcome_a])) - float(_truth(row[outcome_b])))
    point = sum(differences) / len(differences)
    maps = sorted(hierarchy, key=str)
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        sampled_differences = []
        for _map_draw in range(len(maps)):
            map_id = rng.choice(maps)
            routes = sorted(hierarchy[map_id], key=str)
            for _route_draw in range(len(routes)):
                route_id = rng.choice(routes)
                episodes = hierarchy[map_id][route_id]
                for _episode_draw in range(len(episodes)):
                    row = rng.choice(episodes)
                    sampled_differences.append(
                        float(_truth(row[outcome_a])) - float(_truth(row[outcome_b]))
                    )
        draws.append(sum(sampled_differences) / len(sampled_differences))
    alpha = (1.0 - confidence) / 2.0
    return {
        "analysis_unit": "episode",
        "hierarchy": ["map", "route", "episode"],
        "contrast": f"{outcome_a}_minus_{outcome_b}",
        "episode_count": len(selected),
        "map_count": len(maps),
        "point_estimate": point,
        "confidence": confidence,
        "confidence_interval": [_percentile(draws, alpha), _percentile(draws, 1.0 - alpha)],
        "bootstrap_replicates": replicates,
        "seed": seed,
    }
