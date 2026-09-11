from pathlib import Path

import yaml

from scripts.run_live_integrity_campaign import expand


ROOT = Path(__file__).resolve().parents[1]


def test_live_integrity_manifest_is_development_only_and_complete():
    document = yaml.safe_load(
        (ROOT / "data/manifests/live_integrity_v1.yaml").read_text(encoding="utf-8")
    )
    episodes = expand(document)
    assert len(episodes) == 25
    assert len({item["episode_key"] for item in episodes}) == 25
    assert all(item["map"].startswith("dev_") for item in episodes)
    faulted = [item for item in episodes if item["family"] != "none"]
    assert len(faulted) == 21
    assert {(item["family"], item["severity"]) for item in faulted} == {
        (family, severity)
        for family in document["fault_matrix"]["families"]
        for severity in ("low", "medium", "high")
    }
    audit = document["audit_sampling"]
    assert len(audit["primary_episode_keys"]) == 20
    assert len(audit["infrastructure_invalid_reserve_keys"]) == 5
    assert set(audit["primary_episode_keys"]).isdisjoint(
        audit["infrastructure_invalid_reserve_keys"]
    )
    assert set(audit["primary_episode_keys"] + audit["infrastructure_invalid_reserve_keys"]) == {
        item["episode_key"] for item in episodes
    }


def test_replacement_manifest_has_fresh_keys_and_same_frozen_shape():
    first = yaml.safe_load(
        (ROOT / "data/manifests/live_integrity_v1.yaml").read_text(encoding="utf-8")
    )
    replacement = yaml.safe_load(
        (ROOT / "data/manifests/live_integrity_v2.yaml").read_text(encoding="utf-8")
    )
    first_episodes = expand(first)
    replacement_episodes = expand(replacement)
    assert len(replacement_episodes) == 25
    assert all(item["map"].startswith("dev_") for item in replacement_episodes)
    assert {item["episode_key"] for item in first_episodes}.isdisjoint(
        item["episode_key"] for item in replacement_episodes
    )
    audit = replacement["audit_sampling"]
    assert len(audit["primary_episode_keys"]) == 20
    assert len(audit["infrastructure_invalid_reserve_keys"]) == 5
    assert set(audit["primary_episode_keys"] + audit["infrastructure_invalid_reserve_keys"]) == {
        item["episode_key"] for item in replacement_episodes
    }


def test_definitive_v3_manifest_has_early_causal_onset_and_fresh_keys():
    prior = yaml.safe_load(
        (ROOT / "data/manifests/live_integrity_v2.yaml").read_text(encoding="utf-8")
    )
    definitive = yaml.safe_load(
        (ROOT / "data/manifests/live_integrity_v3.yaml").read_text(encoding="utf-8")
    )
    prior_keys = {item["episode_key"] for item in expand(prior)}
    episodes = expand(definitive)
    assert len(episodes) == 25
    assert prior_keys.isdisjoint(item["episode_key"] for item in episodes)
    faulted = [item for item in episodes if item["family"] != "none"]
    assert all(item["clean_prefix_seconds"] == 5.0 for item in faulted)
    assert all(item["planned_onset_seconds"] == 5.0 for item in faulted)


def test_development_supplement_manifest_validates_and_is_disjoint():
    from src.experiments import (
        expand_balanced_pilot, targeted_execution_order, validate_development_supplement,
    )
    manifest = ROOT / "data/manifests/development_supplement_v1.yaml"
    if not manifest.exists():
        import pytest
        pytest.skip("supplement not yet preregistered")
    document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    splits = yaml.safe_load(
        (ROOT / "data/manifests/splits.template.yaml").read_text(encoding="utf-8")
    )
    assert validate_development_supplement(document, splits) == []
    episodes = targeted_execution_order(expand_balanced_pilot(document))
    assert len(episodes) == document["expected_episode_count"] >= 324
    keys = {item["episode_key"] for item in episodes}
    seeds = {item["seed"] for item in episodes}
    assert len(keys) == len(seeds) == len(episodes)
    for other in ("balanced_pilot_v1.yaml", "targeted_development_v1.yaml"):
        prior = expand_balanced_pilot(
            yaml.safe_load((ROOT / "data/manifests" / other).read_text(encoding="utf-8"))
        )
        assert keys.isdisjoint(item["episode_key"] for item in prior)
        assert seeds.isdisjoint(item["seed"] for item in prior)


def test_development_supplement_validator_rejects_uneven_or_wrong_kind():
    from src.experiments import validate_development_supplement
    manifest = ROOT / "data/manifests/development_supplement_v1.yaml"
    if not manifest.exists():
        import pytest
        pytest.skip("supplement not yet preregistered")
    document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    splits = yaml.safe_load(
        (ROOT / "data/manifests/splits.template.yaml").read_text(encoding="utf-8")
    )
    wrong_kind = dict(document, campaign_kind="targeted_event_floor")
    assert any("campaign_kind" in f for f in validate_development_supplement(wrong_kind, splits))
    uneven = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    uneven["design"]["conditions"][0]["replicates"] = uneven["design"]["conditions"][0]["replicates"][:1]
    assert any("at most one" in f or "expected_episode_count" in f
               for f in validate_development_supplement(uneven, splits))
