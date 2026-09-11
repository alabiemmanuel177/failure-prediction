"""Independent eligibility guards with final authority over learned recommendations."""

from __future__ import annotations

from dataclasses import dataclass


ACTIONS = (
    "controlled_stop", "relocalise", "replan_clear_costmaps", "backup",
    "spin_active_rescan", "wait", "request_assistance",
)


@dataclass(frozen=True)
class GuardConfig:
    minimum_rear_clearance_m: float = 0.35
    minimum_rotation_clearance_m: float = 0.40
    maximum_repeated_recoveries: int = 2

    def validate(self) -> None:
        if self.minimum_rear_clearance_m <= 0 or self.minimum_rotation_clearance_m <= 0:
            raise ValueError("clearance guards must be positive")
        if self.maximum_repeated_recoveries < 0:
            raise ValueError("maximum repeated recoveries must be nonnegative")


@dataclass(frozen=True)
class RobotState:
    stopped: bool
    stop_allowed: bool
    localisation_poor: bool
    planning_stale_or_blocked: bool
    rear_clearance_m: float | None
    rotation_clearance_m: float | None
    immediate_collision_risk: bool
    obstruction_may_be_transient: bool
    relocalisation_available: bool = False
    repeated_recovery_count: int = 0


def eligible_actions(state: RobotState, config: GuardConfig) -> dict[str, tuple[bool, str]]:
    config.validate()
    repeated = state.repeated_recovery_count >= config.maximum_repeated_recoveries
    if repeated:
        return {
            action: (
                action == "request_assistance",
                "recovery budget exhausted" if action != "request_assistance" else "always eligible",
            )
            for action in ACTIONS
        }
    results = {
        "controlled_stop": (state.stop_allowed, "stop disallowed by frozen safety rule"),
        "relocalise": (
            state.stopped and state.localisation_poor and state.relocalisation_available,
            "requires stopped robot, poor localisation health, and an available procedure",
        ),
        "replan_clear_costmaps": (
            not state.immediate_collision_risk and state.planning_stale_or_blocked,
            "requires safe state and stale or blocked planning",
        ),
        "backup": (
            state.stopped and state.rear_clearance_m is not None
            and state.rear_clearance_m >= config.minimum_rear_clearance_m,
            "requires stopped robot and verified rear clearance",
        ),
        "spin_active_rescan": (
            state.stopped and state.rotation_clearance_m is not None
            and state.rotation_clearance_m >= config.minimum_rotation_clearance_m,
            "requires stopped robot and verified rotation clearance",
        ),
        "wait": (
            not state.immediate_collision_risk and state.obstruction_may_be_transient,
            "requires no immediate collision risk and possibly transient obstruction",
        ),
        "request_assistance": (True, "always eligible"),
    }
    return results
