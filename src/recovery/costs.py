"""Frozen numeric cost of one observed recovery outcome (Protocol section 10)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_COST_CONFIG = Path(__file__).resolve().parents[2] / "configs/recovery_costs.yaml"
COST_TERMS = (
    "collision", "mission_abort", "failed_recovery", "excessive_delay",
    "path_overhead", "unnecessary_intervention",
)


@dataclass(frozen=True)
class CostWeights:
    collision: float
    mission_abort: float
    failed_recovery: float
    excessive_delay_per_second: float
    excessive_delay_cap: float
    path_overhead_per_metre: float
    path_overhead_cap: float
    unnecessary_intervention: float

    def validate(self, dominance_order: tuple[str, ...] = COST_TERMS) -> None:
        values = {
            "collision": self.collision,
            "mission_abort": self.mission_abort,
            "failed_recovery": self.failed_recovery,
            "excessive_delay": self.excessive_delay_cap,
            "path_overhead": self.path_overhead_cap,
            "unnecessary_intervention": self.unnecessary_intervention,
        }
        if tuple(dominance_order) != COST_TERMS:
            raise ValueError(f"dominance order must be {COST_TERMS}")
        if any(value <= 0 for value in values.values()):
            raise ValueError("every maximum cost contribution must be positive")
        if self.excessive_delay_per_second <= 0 or self.path_overhead_per_metre <= 0:
            raise ValueError("per-unit overhead rates must be positive")
        maxima = [values[term] for term in dominance_order]
        if any(earlier <= later for earlier, later in zip(maxima, maxima[1:])):
            raise ValueError(
                "maximum contributions must strictly decrease along the frozen dominance order"
            )

    def maximum_contribution(self, term: str) -> float:
        return {
            "collision": self.collision,
            "mission_abort": self.mission_abort,
            "failed_recovery": self.failed_recovery,
            "excessive_delay": self.excessive_delay_cap,
            "path_overhead": self.path_overhead_cap,
            "unnecessary_intervention": self.unnecessary_intervention,
        }[term]


@dataclass(frozen=True)
class RecoveryOutcome:
    """Observed, label-only outcome of one warning-triggered recovery."""

    collision: bool
    mission_abort: bool
    failed_recovery: bool
    added_time_seconds: float
    added_path_length_m: float
    unnecessary_intervention: bool

    def validate(self) -> None:
        if self.added_time_seconds < 0 or self.added_path_length_m < 0:
            raise ValueError("overheads must be nonnegative")


def load_cost_config(path: Path = DEFAULT_COST_CONFIG) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or "numeric_weights" not in document:
        raise ValueError(f"{path} has no numeric_weights section")
    if document.get("selector_training_rule") != (
        "actions_rejected_by_guard_are_never_training_targets"
    ):
        raise ValueError("selector_training_rule must forbid guard-rejected training targets")
    return document


def cost_weights_from_document(document: Mapping[str, Any]) -> CostWeights:
    weights = CostWeights(**{
        key: float(value) for key, value in document["numeric_weights"].items()
    })
    weights.validate(tuple(document.get("dominance_order", COST_TERMS)))
    return weights


def load_cost_weights(path: Path = DEFAULT_COST_CONFIG) -> CostWeights:
    return cost_weights_from_document(load_cost_config(path))


def cost_breakdown(outcome: RecoveryOutcome, weights: CostWeights) -> dict[str, float]:
    outcome.validate()
    weights.validate()
    return {
        "collision": weights.collision if outcome.collision else 0.0,
        "mission_abort": weights.mission_abort if outcome.mission_abort else 0.0,
        "failed_recovery": weights.failed_recovery if outcome.failed_recovery else 0.0,
        "excessive_delay": min(
            weights.excessive_delay_cap,
            weights.excessive_delay_per_second * outcome.added_time_seconds,
        ),
        "path_overhead": min(
            weights.path_overhead_cap,
            weights.path_overhead_per_metre * outcome.added_path_length_m,
        ),
        "unnecessary_intervention": (
            weights.unnecessary_intervention if outcome.unnecessary_intervention else 0.0
        ),
    }


def observed_cost(outcome: RecoveryOutcome, weights: CostWeights) -> float:
    return float(sum(cost_breakdown(outcome, weights).values()))


def outcome_from_row(row: Mapping[str, object]) -> RecoveryOutcome:
    """Build an outcome from a CSV/YAML row using the paired-recovery field names."""
    def truth(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes"}
        return bool(value)

    return RecoveryOutcome(
        collision=truth(row.get("collision", False)),
        mission_abort=truth(row.get("mission_abort", not truth(row.get("mission_complete", True)))),
        failed_recovery=truth(row.get("failed_recovery", False)),
        added_time_seconds=float(row.get("added_time_seconds", 0.0)),
        added_path_length_m=float(row.get("added_path_length_m", 0.0)),
        unnecessary_intervention=truth(row.get("unnecessary_intervention", False)),
    )
