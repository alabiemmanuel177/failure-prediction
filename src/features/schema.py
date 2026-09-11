"""Validation and resolution of the executable raw-feature contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .causal import FeatureSpec


def load_raw_feature_contract(path: Path) -> dict[str, dict[str, Any]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    contract = document.get("raw_feature_contract") if isinstance(document, dict) else None
    if not isinstance(contract, dict) or not contract:
        raise ValueError("feature schema lacks a non-empty raw_feature_contract")
    validated: dict[str, dict[str, Any]] = {}
    for name, raw in contract.items():
        if not isinstance(raw, dict):
            raise ValueError(f"{name}: raw feature contract must be a mapping")
        sources = raw.get("sources")
        maximum_age = raw.get("max_age_seconds")
        if not isinstance(sources, list) or not sources or any(
            not isinstance(source, str) or not source.startswith("/") for source in sources
        ):
            raise ValueError(f"{name}: sources must be a non-empty ROS topic list")
        try:
            maximum_age = float(maximum_age)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name}: max_age_seconds must be numeric") from error
        if maximum_age <= 0:
            raise ValueError(f"{name}: max_age_seconds must be positive")
        validated[str(name)] = {
            "sources": tuple(sources),
            "max_age_seconds": maximum_age,
        }
    return validated


def load_primary_feature_set(path: Path) -> tuple[str, ...]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("feature schema must be a mapping")
    primary = document.get("primary_feature_set", {})
    values = primary.get("values") if isinstance(primary, dict) else None
    if not isinstance(values, list) or not values or any(not isinstance(name, str) for name in values):
        raise ValueError("primary_feature_set.values must be a non-empty string list")
    if len(values) != len(set(values)):
        raise ValueError("primary_feature_set.values contains duplicates")
    companions = primary.get("include_companion_channels")
    if companions != ["age_seconds", "missing"]:
        raise ValueError("primary feature companions must be age_seconds and missing")
    raw = set(load_raw_feature_contract(path))
    derived_values = document.get("derived_feature_contract")
    if not isinstance(derived_values, list) or any(
        not isinstance(name, str) for name in derived_values
    ):
        raise ValueError("derived_feature_contract must be a string list")
    available = raw | set(derived_values)
    unknown = sorted(set(values) - available)
    if unknown:
        raise ValueError(f"primary feature set contains undeclared values: {unknown}")
    return tuple(values)


def resolve_feature_specs(
    contract: Mapping[str, Mapping[str, Any]], telemetry_rows: Iterable[Mapping[str, str]],
) -> tuple[list[FeatureSpec], dict[str, list[Mapping[str, str]]]]:
    """Return a stable full schema plus rows grouped by feature.

    An observed feature must use exactly one declared transport source and the frozen
    maximum age. Features absent for the whole episode still receive a canonical
    source so causal resampling emits explicit missing channels.
    """
    grouped: dict[str, list[Mapping[str, str]]] = {}
    observed_sources: dict[str, set[str]] = {}
    for row in telemetry_rows:
        name = str(row.get("feature", ""))
        if name not in contract:
            raise ValueError(f"telemetry feature is outside raw_feature_contract: {name!r}")
        source = str(row.get("source", ""))
        allowed = tuple(contract[name]["sources"])
        if source not in allowed:
            raise ValueError(f"{name}: undeclared source {source!r}")
        maximum_age = float(row["max_age_seconds"])
        if maximum_age != float(contract[name]["max_age_seconds"]):
            raise ValueError(f"{name}: maximum age differs from frozen feature contract")
        grouped.setdefault(name, []).append(row)
        observed_sources.setdefault(name, set()).add(source)
    specs = []
    for name, definition in contract.items():
        sources = observed_sources.get(name, set())
        if len(sources) > 1:
            raise ValueError(f"{name}: mixed transport sources in one episode")
        source = next(iter(sources)) if sources else str(definition["sources"][0])
        specs.append(FeatureSpec(name, source, float(definition["max_age_seconds"])))
    return specs, grouped
