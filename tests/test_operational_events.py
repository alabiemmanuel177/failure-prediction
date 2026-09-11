import pytest

from src.labels.operational_events import (
    MotionSample,
    PerceptionSafetySample,
    PoseErrorSample,
    first_immobilisation,
    first_localisation_loss,
    first_primary_event,
    first_unsafe_perception,
)


def pose(t, translation=0.0, yaw=0.0):
    return PoseErrorSample(t, translation, yaw)


def motion(t, command=0.1, x=0.0, y=0.0):
    return MotionSample(t, command, x, y)


def test_localisation_requires_uninterrupted_channel_specific_persistence():
    samples = [pose(0.0, 0.51), pose(0.5, 0.51), pose(1.0, 0.49),
               pose(1.5, 0.51), pose(2.0, 0.51), pose(2.5, 0.51),
               pose(3.0, 0.51), pose(3.5, 0.51)]
    assert first_localisation_loss(samples) == 3.5


def test_localisation_does_not_combine_translation_and_yaw_intervals():
    samples = [pose(0.0, 0.51), pose(0.5, 0.51), pose(1.0, yaw=0.51),
               pose(1.5, yaw=0.51), pose(2.0), pose(2.5)]
    assert first_localisation_loss(samples) is None


def test_localisation_gap_breaks_continuity():
    samples = [pose(0.0, 0.51), pose(0.5, 0.51), pose(2.0, 0.51),
               pose(2.5, 0.51), pose(3.0, 0.51), pose(3.5, 0.51),
               pose(4.0, 0.51)]
    assert first_localisation_loss(samples, maximum_gap_seconds=0.5) == 4.0


def test_immobilisation_detects_ten_seconds_without_half_metre_progress():
    samples = [motion(t * 0.5, x=0.01 * t) for t in range(21)]
    assert first_immobilisation(samples) == 10.0


def test_immobilisation_resets_after_progress_or_command_interruption():
    progress = [motion(t * 0.5, x=0.03 * t) for t in range(21)]
    assert first_immobilisation(progress) is None
    interrupted = [motion(t * 0.5, command=0.0 if t == 10 else 0.1)
                   for t in range(32)]
    assert first_immobilisation(interrupted) == 15.5


def test_unsafe_perception_requires_all_three_conditions():
    samples = [
        PerceptionSafetySample(1.0, 0.40, True, False),
        PerceptionSafetySample(2.0, 0.43, True, True),
        PerceptionSafetySample(3.0, 0.42, True, True),
    ]
    assert first_unsafe_perception(samples) == 3.0


def test_primary_event_uses_time_then_frozen_precedence():
    assert first_primary_event(
        {"mission_timeout": 20.0, "collision": 20.0, "navigation_abort": 18.0},
        ["collision", "navigation_abort", "mission_timeout"],
    ) == ("navigation_abort", 18.0)
    assert first_primary_event(
        {"mission_timeout": 20.0, "collision": 20.0},
        ["collision", "mission_timeout"],
    ) == ("collision", 20.0)


def test_temporal_detectors_reject_non_monotonic_samples():
    with pytest.raises(ValueError, match="strictly increasing"):
        first_localisation_loss([pose(1.0), pose(1.0)])
