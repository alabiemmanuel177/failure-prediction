"""Fail-closed checks for deployable feature names and sources."""

from __future__ import annotations

from dataclasses import dataclass
import fnmatch
from pathlib import Path
from typing import Iterable

import yaml


class LeakageError(ValueError):
    """A label-only or provenance-only field entered the deployable matrix."""


@dataclass(frozen=True)
class LeakagePolicy:
    forbidden_topic_patterns: tuple[str, ...]
    forbidden_field_patterns: tuple[str, ...]

    @classmethod
    def from_yaml(cls, path: Path) -> "LeakagePolicy":
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if document.get("policy") != "fail_closed":
            raise LeakageError("leakage policy must be fail_closed")
        return cls(
            tuple(str(item) for item in document["forbidden_topic_patterns"]),
            tuple(str(item) for item in document["forbidden_field_patterns"]),
        )

    @staticmethod
    def _matches(value: str, patterns: Iterable[str]) -> bool:
        lowered = value.lower()
        for pattern in patterns:
            candidate = pattern.lower()
            if any(char in candidate for char in "*?["):
                if fnmatch.fnmatch(lowered, candidate):
                    return True
            elif lowered == candidate or candidate in lowered.split("."):
                return True
        return False

    def validate(self, *, feature_name: str, source: str) -> None:
        if self._matches(source, self.forbidden_topic_patterns):
            raise LeakageError(f"forbidden deployable source: {source}")
        if self._matches(feature_name, self.forbidden_field_patterns):
            raise LeakageError(f"forbidden deployable field: {feature_name}")

