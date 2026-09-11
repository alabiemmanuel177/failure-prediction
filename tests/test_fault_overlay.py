from pathlib import Path

import numpy as np
import pytest
import yaml

from failure_experiment.parameters import FAMILIES, load_fault_parameters
from failure_experiment.schedule import FaultSchedule
from failure_experiment.transforms import (
    biased_progress,
    camera_occlusion,
    lidar_dropout,
    semantic_risk_corruption,
)
from failure_experiment.events import EVENT_TOPIC
from failure_experiment.environment_fault import point_along_path


ROOT = Path(__file__).resolve().parents[1]


def test_exactly_seven_fault_families_plus_control():
    assert FAMILIES - {"none"} == {
        "camera_occlusion", "lidar_dropout", "wheel_slip",
        "localisation_perturbation", "dynamic_blockage",
        "planner_oscillation", "semantic_corruption",
    }
    for family in FAMILIES - {"none"}:
        for severity in ("low", "medium", "high"):
            assert load_fault_parameters(ROOT, family, severity)


def test_schedule_enforces_clean_prefix_and_delayed_eligibility():
    schedule = FaultSchedule(10.0, 15.0, 20.0, 5.0)
    assert schedule.update(100.0, eligible=True) == "waiting_for_goal"
    assert schedule.arm(100.0)
    assert schedule.update(109.9, eligible=True) == "clean_prefix"
    assert schedule.update(110.0, eligible=True) == "waiting_for_onset"
    assert schedule.update(115.0, eligible=False) == "waiting_for_eligibility"
    assert schedule.update(117.0, eligible=True) == "activated"
    assert schedule.actual_onset_time == 117.0
    assert schedule.update(137.0, eligible=True) == "active"
    assert schedule.update(137.1, eligible=True) == "complete"


def test_schedule_rejects_when_eligibility_never_occurs():
    schedule = FaultSchedule(1.0, 2.0, 3.0, 1.0)
    schedule.arm(10.0)
    assert schedule.update(13.0, eligible=False) == "waiting_for_eligibility"
    assert schedule.update(13.01, eligible=False) == "rejected"
    assert schedule.update(20.0, eligible=True) == "ineligible"


def test_camera_occlusion_is_seed_and_timestamp_deterministic():
    image = np.full((4, 10, 3), 255, dtype=np.uint8)
    first = camera_occlusion(
        image, mask_fraction=0.5, dropout_probability=0.0, seed=7, stamp_ns=100
    )
    second = camera_occlusion(
        image, mask_fraction=0.5, dropout_probability=0.0, seed=7, stamp_ns=100
    )
    assert np.array_equal(first, second)
    assert np.all(first[:, :5] == 255)
    assert np.all(first[:, 5:] == 0)


def test_lidar_dropout_is_contiguous_and_deterministic():
    first = lidar_dropout([1.0] * 20, invalid_fraction=0.35, seed=11)
    second = lidar_dropout([1.0] * 20, invalid_fraction=0.35, seed=11)
    assert first == second
    assert sum(np.isinf(first)) == 7


def test_semantic_corruption_preserves_unknown_cells():
    output = semantic_risk_corruption(
        [-1, 100, 100, 100], corruption_probability=1.0,
        risk_scale=0.5, seed=3, stamp_ns=10,
    )
    assert output == [-1, 50, 50, 50]


def test_biased_progress_is_incremental_not_absolute_scaling():
    assert biased_progress((10.0, 5.0), (12.0, 7.0), (20.0, 30.0), 0.5) == (21.0, 31.0)
    with pytest.raises(ValueError):
        biased_progress((0, 0), (1, 1), (0, 0), 1.1)


def test_route_relative_fault_placement_uses_polyline_arc_length():
    x, y, yaw = point_along_path([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)], 0.75)
    assert (x, y) == pytest.approx((2.0, 1.0))
    assert yaw == pytest.approx(np.pi / 2)
    with pytest.raises(ValueError):
        point_along_path([(0.0, 0.0)], 0.5)


def test_research1_lock_forbids_protected_test_split():
    lock = yaml.safe_load((ROOT / "integration" / "research1.lock.yaml").read_text())
    assert lock["allowed_splits"] == ["development", "validation"]
    assert lock["forbidden_splits"] == ["test"]
    assert len(lock["commit_sha"]) == 40


def test_research_event_topic_is_label_only_not_a_feature_source():
    features = yaml.safe_load((ROOT / "configs/feature_schema.yaml").read_text())
    sources = {
        source
        for group in features["groups"].values()
        for source in group.get("sources", [])
    }
    assert EVENT_TOPIC not in sources
