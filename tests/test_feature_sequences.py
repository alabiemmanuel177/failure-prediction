from pathlib import Path

from src.features import (
    FeatureSpec, LeakagePolicy, ScalarSample, assemble_causal_sequences,
)


ROOT = Path(__file__).resolve().parents[1]


def policy():
    return LeakagePolicy.from_yaml(ROOT / "configs/leakage_denylist.yaml")


def inputs(future_value=99.0):
    specs = [
        FeatureSpec("command_linear", "/cmd_vel", 0.5),
        FeatureSpec("measured_linear", "/odom", 0.5),
        FeatureSpec("odom_x", "/odom", 0.5),
        FeatureSpec("odom_y", "/odom", 0.5),
    ]
    samples = {
        "command_linear": [ScalarSample(t, 0.2) for t in (0.5, 1.0, 1.5, 2.0)],
        "measured_linear": [ScalarSample(t, t) for t in (0.5, 1.0, 1.5, 2.0)]
                           + [ScalarSample(2.5, future_value)],
        "odom_x": [ScalarSample(t, t) for t in (0.5, 1.0, 1.5, 2.0)],
        "odom_y": [ScalarSample(t, 0.0) for t in (0.5, 1.0, 1.5, 2.0)],
    }
    labels = [{
        "decision_index": 0, "decision_time": 2.0,
        "label": 1, "eligibility": "eligible_positive",
    }]
    primary = ["command_linear", "measured_linear", "tracking_error", "goal_distance"]
    return specs, samples, labels, primary


def test_sequence_has_complete_fixed_rate_history_and_companion_channels():
    specs, samples, labels, primary = inputs()
    examples = assemble_causal_sequences(
        run_id="run", episode_start=0.0, episode_end=2.5,
        labels=labels, specs=specs, samples_by_feature=samples,
        leakage_policy=policy(), primary_features=primary,
        goal_x=4.0, goal_y=0.0, history_seconds=2.0, stride_seconds=0.5,
    )
    assert len(examples) == 1
    assert len(examples[0].values) == 4
    assert len(examples[0].feature_names) == 12
    assert examples[0].decision_time == 2.0


def test_future_sample_cannot_change_an_earlier_sequence():
    specs, samples, labels, primary = inputs(future_value=99.0)
    first = assemble_causal_sequences(
        run_id="run", episode_start=0.0, episode_end=2.5,
        labels=labels, specs=specs, samples_by_feature=samples,
        leakage_policy=policy(), primary_features=primary,
        goal_x=4.0, goal_y=0.0, history_seconds=2.0, stride_seconds=0.5,
    )[0]
    specs, samples, labels, primary = inputs(future_value=-99.0)
    second = assemble_causal_sequences(
        run_id="run", episode_start=0.0, episode_end=2.5,
        labels=labels, specs=specs, samples_by_feature=samples,
        leakage_policy=policy(), primary_features=primary,
        goal_x=4.0, goal_y=0.0, history_seconds=2.0, stride_seconds=0.5,
    )[0]
    assert first.values == second.values


def test_ineligible_label_rows_are_not_emitted():
    specs, samples, labels, primary = inputs()
    labels[0]["label"] = None
    assert assemble_causal_sequences(
        run_id="run", episode_start=0.0, episode_end=2.5,
        labels=labels, specs=specs, samples_by_feature=samples,
        leakage_policy=policy(), primary_features=primary,
        goal_x=4.0, goal_y=0.0, history_seconds=2.0, stride_seconds=0.5,
    ) == []
