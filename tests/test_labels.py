from pathlib import Path

import yaml

from src.labels.annotations import validate_annotation
from src.labels.causal_windows import LabelConfig, label_decision_times


ROOT = Path(__file__).resolve().parents[1]


def read_yaml(path):
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def by_time(records):
    return {record["decision_time"]: record for record in records}


def test_collision_fixture_is_valid():
    annotation = read_yaml("tests/fixtures/episode_collision.yaml")
    taxonomy = read_yaml("configs/failure_taxonomy.yaml")
    assert validate_annotation(annotation, taxonomy) == []


def test_exact_causal_label_boundaries():
    records = label_decision_times(
        episode_start=0,
        episode_end=45,
        events=[{"class": "collision", "time": 40.0, "terminal": True}],
        injection_onsets=[22.0],
        precedence=["collision", "mission_timeout"],
        config=LabelConfig(),
    )
    rows = by_time(records)
    assert rows[5.0]["window_start"] == 0.0
    assert rows[30.0]["eligibility"] == "eligible_positive"
    assert rows[30.0]["label"] == 1
    assert rows[39.0]["eligibility"] == "eligible_positive"
    assert rows[39.0]["label"] == 1
    assert rows[39.5]["eligibility"] == "excluded_too_late"
    assert rows[40.0]["eligibility"] == "excluded_too_late"
    assert rows[40.5]["eligibility"] == "excluded_post_event"


def test_guard_excludes_early_decisions_near_injection():
    records = label_decision_times(
        episode_start=0,
        episode_end=45,
        events=[{"class": "collision", "time": 40.0, "terminal": True}],
        injection_onsets=[22.0],
        precedence=["collision"],
        config=LabelConfig(),
    )
    rows = by_time(records)
    assert rows[5.0]["eligibility"] == "excluded_near_event_or_injection"
    assert rows[29.5]["eligibility"] == "excluded_near_event_or_injection"


def test_clean_episode_produces_only_eligible_negatives():
    records = label_decision_times(
        episode_start=0,
        episode_end=10,
        events=[],
        injection_onsets=[],
        precedence=["collision"],
        config=LabelConfig(),
    )
    assert records
    assert {record["label"] for record in records} == {0}
    assert {record["eligibility"] for record in records} == {"eligible_negative"}


def test_same_time_event_uses_frozen_precedence():
    records = label_decision_times(
        episode_start=0,
        episode_end=20,
        events=[
            {"class": "mission_timeout", "time": 20.0, "terminal": True},
            {"class": "collision", "time": 20.0, "terminal": True},
        ],
        injection_onsets=[],
        precedence=["collision", "mission_timeout"],
        config=LabelConfig(),
    )
    assert records[0]["primary_event_class"] == "collision"

