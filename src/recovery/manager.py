"""Auditable recovery-policy decisions with guards holding final authority."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from .guards import GuardConfig, RobotState, eligible_actions
from .selector import rule_matched_action, select_lowest_cost


@dataclass(frozen=True)
class RecoveryRequest:
    run_id: str
    warning_id: str
    risk_score: float
    diagnosed_signal_group: str
    state: RobotState

    def validate(self) -> None:
        if not self.run_id or not self.warning_id:
            raise ValueError("run_id and warning_id are required")
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError("risk_score must lie in [0, 1]")


def decide_recovery(
    request: RecoveryRequest,
    guard_config: GuardConfig,
    policy_id: str,
    predicted_costs: Mapping[str, float] | None = None,
) -> dict[str, object]:
    request.validate()
    guards = eligible_actions(request.state, guard_config)
    forced = None
    guard_rejected = False
    if policy_id.startswith("RP_"):
        # Recovery pilot: the frozen predictor's alarm forces one action; the guard keeps
        # final authority and a rejected forced action degrades to a controlled stop.
        forced = policy_id[len("RP_"):]
        if forced not in guards or forced == "request_assistance":
            raise ValueError(f"unsupported forced pilot action: {forced}")
        if guards[forced][0]:
            action, reason = forced, "forced pilot action eligible"
        else:
            guard_rejected = True
            if guards["controlled_stop"][0]:
                action = "controlled_stop"
                reason = f"forced {forced} rejected by guard; controlled stop executed"
            else:
                action = "request_assistance"
                reason = f"forced {forced} and controlled_stop rejected by guard"
    elif policy_id == "R1":
        if guards["controlled_stop"][0]:
            action, reason = "controlled_stop", "fixed conservative policy"
        else:
            action, reason = "request_assistance", "controlled stop rejected by guard"
    elif policy_id == "R2":
        action, reason = rule_matched_action(request.diagnosed_signal_group, guards)
    elif policy_id == "R3":
        if predicted_costs is None:
            raise ValueError("R3 requires predicted action costs")
        action, cost = select_lowest_cost(predicted_costs, guards)
        reason = f"lowest eligible predicted cost={cost}"
    else:
        raise ValueError(f"unsupported recovery policy: {policy_id}")
    if not guards.get(action, (False, "unknown action"))[0]:
        raise AssertionError("selector returned an action rejected by the independent guard")
    return {
        "run_id": request.run_id,
        "warning_id": request.warning_id,
        "risk_score": request.risk_score,
        "diagnosed_signal_group": request.diagnosed_signal_group,
        "policy_id": policy_id,
        "recommended_action": action,
        "selection_reason": reason,
        "forced_action": forced,
        "guard_rejected": guard_rejected,
        "abstained": action == "request_assistance",
        "eligible_actions": [name for name, result in guards.items() if result[0]],
        "guard_results": {
            name: {"eligible": result[0], "reason": result[1]}
            for name, result in guards.items()
        },
        "state": asdict(request.state),
        "execution_performed": False,
    }
