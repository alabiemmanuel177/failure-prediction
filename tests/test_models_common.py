import json
from pathlib import Path

import numpy as np
import pytest

from src.features import NormalizationBundle
from src.models import common
from src.models.common import (
    EpisodeArrays, EpisodeRecord, apply_feature_mask, apply_normalization, auroc,
    average_precision, build_training_set, enforce_split, fit_normalization,
    frozen_feature_names, load_ablations, load_dataset_manifest, load_episode_decisions,
    masked_age_by_feature, resolve_feature_mask, subsample_negatives,
)
from models_synthetic import build_dataset


ROOT = Path(__file__).resolve().parents[1]
FEATURES = frozen_feature_names()


def _record(**overrides):
    base = dict(run_id="r", split="development", map_id="m", route_id="m_r0", fault_family="none",
                severity="none", seed=1, protected_test_used=False, primary_event_class=None,
                primary_event_time=None, sequences_sha256=None, dataset_id="d", dataset_root=Path("."))
    base.update(overrides)
    return EpisodeRecord(**base)


def test_frozen_feature_order_is_28_by_3_and_matches_schema_companions():
    assert len(FEATURES) == 84
    assert FEATURES[:3] == ("command_linear", "command_linear__age_seconds", "command_linear__missing")


def test_ablation_groups_cover_every_primary_feature_exactly_once():
    document = load_ablations()
    members = [f for group in document["feature_groups"].values() for f in group]
    assert sorted(members) == sorted(common.primary_value_features())


def test_feature_ablation_zeroes_value_sets_missing_and_max_age_without_reordering():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 10, 84)).astype(np.float32)
    mask = resolve_feature_mask("no_localisation", primary_features=common.primary_value_features())
    assert mask.masked_features == ("pose_covariance_trace", "pose_jump")
    out = apply_feature_mask(X, FEATURES, mask)
    ages = masked_age_by_feature()
    index = {n: i for i, n in enumerate(FEATURES)}
    assert out.shape == X.shape
    assert (out[..., index["pose_covariance_trace"]] == 0).all()
    assert (out[..., index["pose_covariance_trace__missing"]] == 1).all()
    assert ages["pose_covariance_trace"] == 2.0
    assert (out[..., index["pose_covariance_trace__age_seconds"]] == 2.0).all()
    assert ages["pose_jump"] == 0.0     # derived features carry age 0 in the extractor
    untouched = [i for n, i in index.items() if not n.startswith(("pose_covariance_trace", "pose_jump"))]
    assert np.array_equal(out[..., untouched], X[..., untouched])
    assert np.array_equal(X, X)  # input not modified in place
    assert apply_feature_mask(X, FEATURES, resolve_feature_mask(None, primary_features=FEATURES)) is X


def test_single_timestamp_ablation_hides_all_but_the_last_step():
    X = np.ones((2, 10, 84), dtype=np.float32)
    mask = resolve_feature_mask("single_timestamp", primary_features=common.primary_value_features())
    out = apply_feature_mask(X, FEATURES, mask)
    index = {n: i for i, n in enumerate(FEATURES)}
    assert (out[:, :9, index["command_linear"]] == 0).all()
    assert (out[:, :9, index["command_linear__missing"]] == 1).all()
    assert (out[:, :9, index["command_linear__age_seconds"]] == 0.5).all()
    assert np.array_equal(out[:, 9], X[:, 9])


def test_policy_ablations_leave_the_matrix_untouched_and_unknown_names_fail():
    assert common.ablation_kind("no_persistence") == "policy"
    mask = resolve_feature_mask("no_persistence", primary_features=common.primary_value_features())
    assert mask.is_identity or mask.masked_features == ()
    with pytest.raises(ValueError):
        resolve_feature_mask("no_such_thing", primary_features=FEATURES)


def test_array_normalisation_matches_the_row_based_bundle_and_zeroes_missing():
    rng = np.random.default_rng(1)
    names = ("a", "a__age_seconds", "a__missing", "b", "b__age_seconds", "b__missing")
    X = rng.standard_normal((6, 3, 6)).astype(np.float32)
    X[..., 2] = (rng.random((6, 3)) < 0.3)
    X[..., 5] = 0.0
    bundle = fit_normalization(X, names)
    rows = [dict(zip(names, map(float, row))) for row in X.reshape(-1, 6)]
    reference = NormalizationBundle.fit(rows, ["a", "b", "a__age_seconds", "b__age_seconds"],
                                        split="development")
    assert bundle.means == pytest.approx(reference.means)
    assert bundle.scales == pytest.approx(reference.scales)
    out = apply_normalization(X, names, bundle)
    expected = np.asarray([[reference.transform(r)[n] for n in names] for r in rows]).reshape(X.shape)
    assert np.allclose(out, expected, atol=1e-5)
    assert (out[..., 0][X[..., 2] == 1] == 0).all()
    assert (out[..., 2] == X[..., 2]).all()      # missing flags pass through
    assert json.loads(bundle.canonical_json())["fit_split"] == "development"


def test_split_enforcement_refuses_wrong_split_and_unmarked_protection():
    enforce_split([_record()], "development", purpose="fitting")
    with pytest.raises(ValueError, match="fitting"):
        enforce_split([_record(split="validation")], "development", purpose="fitting")
    with pytest.raises(ValueError, match="selection"):
        enforce_split([_record(split="development")], "validation", purpose="selection")
    with pytest.raises(ValueError, match="protected_test_used"):
        enforce_split([_record(protected_test_used=True)], "development", purpose="fitting")
    with pytest.raises(ValueError, match="protected_test_used"):
        enforce_split([_record(protected_test_used=None)], "development", purpose="fitting")


def test_manifest_loader_and_decision_loader_use_contract_layout(tmp_path):
    manifest = build_dataset(tmp_path, "syn-development-4", "development", episodes=4, decisions=50)
    loaded = load_dataset_manifest(manifest)
    assert loaded.dataset_id == "syn-development-4" and len(loaded.records) == 4
    episode = load_episode_decisions(loaded.records[1], FEATURES)
    assert set(np.unique(episode.y)) <= {-1, 0, 1} and episode.eligibility is not None
    assert (episode.y[episode.eligibility == "eligible_positive"] == 1).all()
    with pytest.raises(FileNotFoundError, match="not published"):
        load_dataset_manifest("missing-dataset", tmp_path)
    no_decisions = build_dataset(tmp_path, "nodec-development-2", "development", episodes=2,
                                 write_decisions=False)
    with pytest.raises(FileNotFoundError, match="assemble_episode_decisions"):
        load_episode_decisions(load_dataset_manifest(no_decisions).records[0], FEATURES)


def test_negative_subsampling_caps_per_episode_keeps_all_positives_and_spreads_phases():
    y = np.array([0] * 90 + [1] * 10, dtype=np.int8)
    episode = EpisodeArrays(_record(), np.zeros((100, 10, 84), np.float32), y,
                            np.arange(100), 10.0 + 0.5 * np.arange(100))
    kept = subsample_negatives(episode, cap_per_episode=20, phase_bins=4, rng=np.random.default_rng(0))
    assert (y[kept] == 1).sum() == 10 and (y[kept] == 0).sum() == 20
    negatives = kept[y[kept] == 0]
    assert all(((negatives >= lo) & (negatives < lo + 23)).sum() >= 4 for lo in (0, 23, 46, 68))
    assert kept.size == 100 or subsample_negatives(
        episode, cap_per_episode=None, phase_bins=4, rng=np.random.default_rng(0)).size == 100


def test_training_set_reports_natural_and_rebalanced_prevalence_from_training_only():
    episodes = []
    for number in range(3):
        y = np.array([0] * 80 + [1] * 5, dtype=np.int8) if number else np.zeros(85, np.int8)
        episodes.append(EpisodeArrays(
            _record(run_id=f"e{number}", fault_family="none" if not number else "wheel_slip"),
            np.zeros((85, 10, 84), np.float32), y, np.arange(85), 5.0 + 0.5 * np.arange(85)))
    train = build_training_set(episodes, cap_per_episode=20, phase_bins=4,
                               positive_weighting="class", seed=3)
    summary = train.summary
    assert summary["natural_prevalence"] == pytest.approx(10 / 255)
    assert summary["fitting_negative_windows"] == 60 and summary["fitting_positive_windows"] == 10
    assert summary["pos_weight"] == pytest.approx(6.0)
    assert summary["rebalanced_prevalence"] == pytest.approx(0.5)
    assert set(train.run_ids) == {"e0", "e1", "e2"}
    clean = build_training_set(episodes, cap_per_episode=20, phase_bins=4, positive_weighting="class",
                               seed=3, negatives_only=True, clean_only=True)
    assert set(clean.run_ids) == {"e0"} and (clean.y == 0).all()


def test_average_precision_and_auroc_match_scikit_learn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(5)
    y = (rng.random(300) < 0.1).astype(int)
    scores = np.round(rng.random(300) + 0.4 * y, 2)   # ties on purpose
    assert average_precision(y, scores) == pytest.approx(sklearn_metrics.average_precision_score(y, scores))
    assert auroc(y, scores) == pytest.approx(sklearn_metrics.roc_auc_score(y, scores))
    assert np.isnan(average_precision(np.zeros(5, int), np.ones(5)))


def test_exclude_family_no_op_guard_applies_to_the_pooled_training_set(monkeypatch):
    """A fold family may be absent from one manifest of the pool (the targeted campaign
    has no planner_oscillation episodes) while present in another: the no-op refusal
    must fire only when the family is absent from the whole pool."""
    with_family = common.DatasetManifest(
        dataset_id="a", path=Path("a"), sha256="0",
        records=(_record(run_id="a1", fault_family="planner_oscillation"), _record(run_id="a2")))
    without_family = common.DatasetManifest(
        dataset_id="b", path=Path("b"), sha256="0", records=(_record(run_id="b1", fault_family="wheel_slip"),))
    # per-manifest strict guard still refuses a silent no-op
    with pytest.raises(ValueError, match="matches no episode"):
        common.exclude_family(without_family.records, "planner_oscillation")
    assert len(common.exclude_family(without_family.records, "planner_oscillation", strict=False)) == 1
    monkeypatch.setattr(common, "load_episode_sequences", lambda record, names: record.run_id)
    kept = common.load_split_sequences([with_family, without_family], "development", purpose="fitting",
                                       feature_names=FEATURES, exclude="planner_oscillation")
    assert kept == ["a2", "b1"]
    with pytest.raises(ValueError, match="across the fitting pool"):
        common.load_split_sequences([with_family, without_family], "development", purpose="fitting",
                                    feature_names=FEATURES, exclude="camera_occlusion")
