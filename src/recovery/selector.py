"""Interpretable and cost-sensitive selectors constrained by guard results."""

from __future__ import annotations

from typing import Mapping


SIGNAL_ACTIONS = {
    "localisation": "relocalise",
    "planning": "replan_clear_costmaps",
    "frontal_blockage": "backup",
    "perception": "spin_active_rescan",
    "transient_obstruction": "wait",
    "unknown": "controlled_stop",
}


def rule_matched_action(
    signal_group: str, guards: Mapping[str, tuple[bool, str]]
) -> tuple[str, str]:
    preferred = SIGNAL_ACTIONS.get(signal_group, "controlled_stop")
    if guards.get(preferred, (False, "unknown action"))[0]:
        return preferred, "preferred action eligible"
    if guards.get("controlled_stop", (False, ""))[0]:
        return "controlled_stop", f"{preferred} rejected by guard"
    return "request_assistance", f"{preferred} and controlled_stop rejected by guard"


def select_lowest_cost(
    predicted_costs: Mapping[str, float], guards: Mapping[str, tuple[bool, str]]
) -> tuple[str, float]:
    eligible = {
        action: float(cost) for action, cost in predicted_costs.items()
        if guards.get(action, (False, ""))[0]
    }
    if not eligible:
        return "request_assistance", float(predicted_costs.get("request_assistance", 0.0))
    return min(eligible.items(), key=lambda item: (item[1], item[0]))

