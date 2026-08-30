from src.recovery import (
    GuardConfig, RobotState, eligible_actions, rule_matched_action, select_lowest_cost,
)


def state(**changes):
    values = dict(
        stopped=True, stop_allowed=True, localisation_poor=False,
        planning_stale_or_blocked=False, rear_clearance_m=1.0,
        rotation_clearance_m=1.0, immediate_collision_risk=False,
        obstruction_may_be_transient=False, repeated_recovery_count=0,
    )
    values.update(changes)
    return RobotState(**values)


def test_backup_and_spin_require_verified_clearance():
    guards = eligible_actions(
        state(rear_clearance_m=None, rotation_clearance_m=0.2), GuardConfig()
    )
    assert not guards["backup"][0]
    assert not guards["spin_active_rescan"][0]
    assert guards["controlled_stop"][0]


def test_immediate_collision_risk_rejects_wait_and_replan():
    guards = eligible_actions(
        state(immediate_collision_risk=True, planning_stale_or_blocked=True,
              obstruction_may_be_transient=True), GuardConfig()
    )
    assert not guards["wait"][0]
    assert not guards["replan_clear_costmaps"][0]


def test_exhausted_recovery_budget_allows_only_assistance():
    guards = eligible_actions(state(repeated_recovery_count=2), GuardConfig())
    assert {name for name, result in guards.items() if result[0]} == {"request_assistance"}


def test_guard_overrides_rule_and_cost_selectors():
    guards = eligible_actions(
        state(localisation_poor=True, stopped=False, rear_clearance_m=None), GuardConfig()
    )
    assert rule_matched_action("localisation", guards)[0] == "controlled_stop"
    action, _cost = select_lowest_cost(
        {"backup": 0.0, "controlled_stop": 2.0, "request_assistance": 5.0}, guards
    )
    assert action == "controlled_stop"
