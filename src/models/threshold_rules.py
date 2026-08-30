"""Interpretable threshold-rule baseline with explicit missing-data behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Rule:
    name: str
    feature: str
    direction: str
    threshold: float
    gate_feature: str | None = None
    gate_direction: str = "above"
    gate_threshold: float | None = None

    def validate(self) -> None:
        if self.direction not in {"above", "below"}:
            raise ValueError(f"{self.name}: direction must be above or below")
        if self.gate_feature and self.gate_threshold is None:
            raise ValueError(f"{self.name}: gated rule requires gate_threshold")
        if self.gate_direction not in {"above", "below"}:
            raise ValueError(f"{self.name}: invalid gate_direction")

    @staticmethod
    def _compare(value: float, direction: str, threshold: float) -> bool:
        return value >= threshold if direction == "above" else value <= threshold

    def fires(self, row: Mapping[str, object]) -> bool:
        self.validate()
        if int(row.get(f"{self.feature}__missing", 0)):
            return False
        if self.gate_feature:
            if int(row.get(f"{self.gate_feature}__missing", 0)):
                return False
            if not self._compare(
                float(row[self.gate_feature]), self.gate_direction, float(self.gate_threshold)
            ):
                return False
        return self._compare(float(row[self.feature]), self.direction, self.threshold)


class ThresholdRuleSet:
    def __init__(self, rules: Sequence[Rule]):
        if not rules:
            raise ValueError("at least one threshold rule is required")
        if len({rule.name for rule in rules}) != len(rules):
            raise ValueError("rule names must be unique")
        self.rules = tuple(rules)
        for rule in self.rules:
            rule.validate()

    def predict(self, row: Mapping[str, object]) -> dict[str, object]:
        fired = [rule.name for rule in self.rules if rule.fires(row)]
        return {
            "risk_score": 1.0 if fired else 0.0,
            "fired_rules": fired,
        }

