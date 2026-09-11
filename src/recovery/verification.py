"""Exhaustive discrete-state verification for the independent recovery guard."""

from __future__ import annotations

from dataclasses import asdict
from itertools import product
from typing import Any

from .guards import ACTIONS, GuardConfig, RobotState, eligible_actions
from .manager import RecoveryRequest, decide_recovery
from .selector import SIGNAL_ACTIONS


def state_space(config: GuardConfig):
    rear_values = (None, config.minimum_rear_clearance_m - 0.01,
                   config.minimum_rear_clearance_m)
    rotation_values = (None, config.minimum_rotation_clearance_m - 0.01,
                       config.minimum_rotation_clearance_m)
    for values in product(
        (False, True), (False, True), (False, True), (False, True),
        rear_values, rotation_values, (False, True), (False, True), (False, True),
        (0, max(0, config.maximum_repeated_recoveries - 1),
         config.maximum_repeated_recoveries),
    ):
        yield RobotState(
            stopped=values[0], stop_allowed=values[1], localisation_poor=values[2],
            planning_stale_or_blocked=values[3], rear_clearance_m=values[4],
            rotation_clearance_m=values[5], immediate_collision_risk=values[6],
            obstruction_may_be_transient=values[7], relocalisation_available=values[8],
            repeated_recovery_count=values[9],
        )


def _invariant_errors(
    state: RobotState, config: GuardConfig, guards: dict[str, tuple[bool, str]]
) -> list[str]:
    eligible = {name for name, result in guards.items() if result[0]}
    errors = []
    exhausted = state.repeated_recovery_count >= config.maximum_repeated_recoveries
    if "request_assistance" not in eligible:
        errors.append("request assistance was rejected")
    if exhausted and eligible != {"request_assistance"}:
        errors.append("exhausted budget admitted an automated action")
    if not exhausted:
        implications = {
            "controlled_stop": state.stop_allowed,
            "relocalise": (
                state.stopped and state.localisation_poor and state.relocalisation_available
            ),
            "replan_clear_costmaps": (
                not state.immediate_collision_risk and state.planning_stale_or_blocked
            ),
            "backup": (
                state.stopped and state.rear_clearance_m is not None
                and state.rear_clearance_m >= config.minimum_rear_clearance_m
            ),
            "spin_active_rescan": (
                state.stopped and state.rotation_clearance_m is not None
                and state.rotation_clearance_m >= config.minimum_rotation_clearance_m
            ),
            "wait": (
                not state.immediate_collision_risk and state.obstruction_may_be_transient
            ),
        }
        for action, condition in implications.items():
            if (action in eligible) != condition:
                errors.append(f"{action} eligibility differs from its frozen condition")
    return errors


def verify_guard_space(config: GuardConfig) -> dict[str, Any]:
    config.validate()
    findings = []
    states_checked = 0
    decisions_checked = 0
    eligibility_counts = {action: 0 for action in ACTIONS}
    signal_groups = sorted({*SIGNAL_ACTIONS, "unrecognised"})
    for index, state in enumerate(state_space(config)):
        states_checked += 1
        guards = eligible_actions(state, config)
        for action, (eligible, _reason) in guards.items():
            eligibility_counts[action] += int(eligible)
        for error in _invariant_errors(state, config, guards):
            findings.append({"state_index": index, "error": error, "state": asdict(state)})
        request = RecoveryRequest(
            run_id=f"verification-{index}", warning_id=f"warning-{index}",
            risk_score=0.75, diagnosed_signal_group="unknown", state=state,
        )
        for policy in ("R1",):
            decision = decide_recovery(request, config, policy)
            decisions_checked += 1
            if not guards[decision["recommended_action"]][0]:
                findings.append({"state_index": index, "error": f"{policy} bypassed guard"})
        for signal in signal_groups:
            signal_request = RecoveryRequest(
                run_id=request.run_id, warning_id=f"{request.warning_id}-{signal}",
                risk_score=request.risk_score, diagnosed_signal_group=signal, state=state,
            )
            decision = decide_recovery(signal_request, config, "R2")
            decisions_checked += 1
            if not guards[decision["recommended_action"]][0]:
                findings.append({"state_index": index, "error": "R2 bypassed guard"})
        for cheapest in ACTIONS:
            costs = {action: (0.0 if action == cheapest else float(position + 1))
                     for position, action in enumerate(ACTIONS)}
            decision = decide_recovery(request, config, "R3", predicted_costs=costs)
            decisions_checked += 1
            if not guards[decision["recommended_action"]][0]:
                findings.append({"state_index": index, "error": "R3 bypassed guard"})
    return {
        "states_checked": states_checked,
        "policy_decisions_checked": decisions_checked,
        "eligibility_counts": eligibility_counts,
        "violation_count": len(findings),
        "violations": findings,
        "passed": not findings,
    }
