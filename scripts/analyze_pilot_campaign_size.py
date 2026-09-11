#!/usr/bin/env python3
"""Freeze the post-pilot, pre-model campaign-size decision from episode evidence."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import publish_new_bytes  # noqa: E402


EVENT_FLOOR = 30
ROUTE_BLOCK = 12
PROTOCOL_TARGET = 3000


def round_up(value: int, block: int) -> int:
    return int(math.ceil(value / block) * block)


def main() -> int:
    rows = [
        json.loads(line)
        for line in (ROOT / "data/manifests/balanced_pilot_v1.episodes.jsonl")
        .read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reuse = yaml.safe_load(
        (ROOT / "reports/integrity/research1_reuse_compatibility_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    families = sorted({row["fault_family"] for row in rows if row["fault_family"] != "none"})
    family_plan = {}
    targeted_total = 0
    for family in families:
        family_rows = [row for row in rows if row["fault_family"] == family]
        events = sum(row["primary_event_class"] is not None for row in family_rows)
        rate = events / len(family_rows)
        estimated_total = math.ceil(EVENT_FLOOR / rate) if rate else len(family_rows)
        route_balanced_total = max(len(family_rows), round_up(estimated_total, ROUTE_BLOCK))
        additional = route_balanced_total - len(family_rows)
        targeted_total += additional
        family_plan[family] = {
            "pilot_episodes": len(family_rows),
            "pilot_terminal_events": events,
            "observed_event_prevalence": rate,
            "planning_event_floor": EVENT_FLOOR,
            "route_balanced_total_episodes": route_balanced_total,
            "additional_episodes": additional,
            "additional_replicates_per_map_route": additional // ROUTE_BLOCK,
        }
    compact = [row["bag_bytes"] for row in rows if row["recording_profile"] == "compact_v2"]
    compatible = reuse["counts"]["core_temporal_topics_by_split"]
    r1_development = int(compatible["development"])
    r1_validation = int(compatible["validation"])
    pre_supplement_fitting = len(rows) + targeted_total + r1_development
    fitting_shortfall = max(0, PROTOCOL_TARGET - pre_supplement_fitting)
    route_balanced_supplement = (
        round_up(fitting_shortfall, ROUTE_BLOCK) if fitting_shortfall else 0
    )
    report = {
        "schema_version": 2,
        "decision_type": "post_pilot_pre_model_campaign_size_split_corrected",
        "protected_outcomes_consulted": False,
        "inputs": {
            "development_pilot_episodes": len(rows),
            "development_pilot_terminal_events": sum(
                row["primary_event_class"] is not None for row in rows
            ),
            "research1_development_structurally_eligible_bags": r1_development,
            "research1_validation_reserved_for_selection": r1_validation,
            "protocol_predictor_dataset_target": PROTOCOL_TARGET,
        },
        "planning_rule": (
            "raise each injected family to at least 30 expected terminal events using "
            "its medium-severity pilot prevalence, rounding totals to complete blocks "
            "of 12 development map-route pairs"
        ),
        "family_plan": family_plan,
        "recommended_targeted_addition_episodes": targeted_total,
        "maximum_fitting_episodes_before_supplement_if_all_research1_development_admitted": (
            pre_supplement_fitting
        ),
        "minimum_fitting_shortfall": fitting_shortfall,
        "minimum_route_balanced_development_supplement": route_balanced_supplement,
        "storage_projection": {
            "compact_v2_mean_bag_mib": sum(compact) / len(compact) / 2**20,
            "targeted_addition_gib": targeted_total * sum(compact) / len(compact) / 2**30,
            "operational_reserve_gib": 100,
            "decision": "execute only while the prospective 100 GiB reserve passes",
        },
        "interpretation": {
            "event_floor_is_campaign_planning_not_a_confirmatory_endpoint": True,
            "research1_admission_is_contingent": True,
            "research1_validation_is_selection_only": True,
            "supplement_is_materialized_only_after_human_and_adapter_admission": True,
            "if_research1_adapter_is_not_admitted": (
                "use post-admission learning curves and add development episodes; do not "
                "count validation or protected episodes toward fitting volume"
            ),
            "planner_oscillation_addition": "none because the pilot already exceeds the event floor",
        },
    }
    output = ROOT / "reports/pilot/post_pilot_campaign_size_v2.yaml"
    payload = yaml.safe_dump(report, sort_keys=False).encode("utf-8")
    if output.exists():
        if output.read_bytes() != payload:
            raise SystemExit(f"refusing to overwrite differing planning decision: {output}")
    else:
        publish_new_bytes(output, payload)
    print(f"wrote planning decision to {output}")
    print(f"targeted development addition: {targeted_total} episodes")
    print(
        "maximum split-safe fitting pool before supplement: "
        f"{pre_supplement_fitting}; route-balanced supplement: {route_balanced_supplement}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
