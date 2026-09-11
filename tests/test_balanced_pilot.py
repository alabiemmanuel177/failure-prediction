from collections import Counter
from pathlib import Path

import yaml

from copy import deepcopy

from src.experiments import (
    balanced_execution_order, expand_balanced_pilot, targeted_execution_order,
    validate_balanced_pilot, validate_targeted_campaign,
)
from scripts.run_balanced_pilot import targeted_prerequisite_findings


ROOT = Path(__file__).resolve().parents[1]


def test_balanced_pilot_is_frozen_unique_and_development_only():
    document = yaml.safe_load(
        (ROOT / "data/manifests/balanced_pilot_v1.yaml").read_text()
    )
    episodes = expand_balanced_pilot(document)
    assert len(episodes) == document["expected_episode_count"] == 648
    assert len({episode["episode_key"] for episode in episodes}) == 648
    assert len({episode["seed"] for episode in episodes}) == 648
    assert all(episode["map"].startswith("dev_") for episode in episodes)
    assert Counter(episode["family"] for episode in episodes) == {
        "none": 144,
        "camera_occlusion": 72,
        "lidar_dropout": 72,
        "wheel_slip": 72,
        "localisation_perturbation": 72,
        "dynamic_blockage": 72,
        "planner_oscillation": 72,
        "semantic_corruption": 72,
    }
    environment = [
        episode for episode in episodes
        if episode["family"] in {"dynamic_blockage", "planner_oscillation"}
    ]
    assert all(episode["placement_mode"] == "path_fraction" for episode in environment)
    assert all(episode["route_fraction"] == 0.55 for episode in environment)
    splits = yaml.safe_load(
        (ROOT / "data/manifests/splits.template.yaml").read_text()
    )
    assert validate_balanced_pilot(document, splits) == []
    transition = document["recording_profile_transition"]
    assert transition["compact_v2_begins_at_attempt_number"] == 98
    assert document["episode_defaults"]["recording_profile"] == "compact_v2"
    replacements = document["infrastructure_replacements"]
    assert {item["original_attempt_number"] for item in replacements} == {97, 141}
    assert len({item["original_episode_key"] for item in replacements}) == 2
    assert len({item["replacement_episode_key"] for item in replacements}) == 2
    assert all(item["original_bag_mcap_count"] == 0 for item in replacements)
    assert all(
        item["scientific_episode_count_contribution"] == 1
        for item in replacements
    )


def test_balanced_pilot_rejects_protected_route_and_bad_timing():
    document = yaml.safe_load(
        (ROOT / "data/manifests/balanced_pilot_v1.yaml").read_text()
    )
    splits = yaml.safe_load(
        (ROOT / "data/manifests/splits.template.yaml").read_text()
    )
    invalid = deepcopy(document)
    invalid["design"]["map_routes"] = {"test_00": ["test_00_r0"]}
    invalid["episode_defaults"]["planned_onset_seconds"] = 1.0
    invalid["expected_episode_count"] = 54
    findings = validate_balanced_pilot(invalid, splits)
    assert "map is outside development split: test_00" in findings
    assert "route is outside development split: test_00_r0" in findings
    assert "planned onset must follow the non-negative clean prefix" in findings


def test_first_staged_wave_balances_all_conditions_across_two_maps():
    document = yaml.safe_load(
        (ROOT / "data/manifests/balanced_pilot_v1.yaml").read_text()
    )
    wave = balanced_execution_order(expand_balanced_pilot(document))[:18]
    condition = Counter(item["episode_key"].rsplit("-", 2)[-2] for item in wave)
    assert set(condition.values()) == {2}
    assert len(condition) == 9
    assert len({item["map"] for item in wave}) == 2
    assert len({item["route"] for item in wave}) == 2


def test_balanced_validation_manifest_is_independent_and_preprotected():
    document = yaml.safe_load(
        (ROOT / "data/manifests/balanced_validation_v1.yaml").read_text(encoding="utf-8")
    )
    splits = yaml.safe_load(
        (ROOT / "data/manifests/splits.template.yaml").read_text(encoding="utf-8")
    )
    episodes = expand_balanced_pilot(document)
    assert validate_balanced_pilot(document, splits) == []
    assert len(episodes) == 324
    assert all(item["map"].startswith("val_") for item in episodes)
    assert all(item["recording_profile"] == "compact_v2" for item in episodes)
    replacements = document["infrastructure_replacements"]
    assert {item["original_attempt_number"] for item in replacements} == {84, 257, 258, 284}
    assert all(item["scientific_episode_count_contribution"] == 1 for item in replacements)
    development = yaml.safe_load(
        (ROOT / "data/manifests/balanced_pilot_v1.yaml").read_text(encoding="utf-8")
    )
    assert {item["seed"] for item in episodes}.isdisjoint(
        item["seed"] for item in expand_balanced_pilot(development)
    )


def test_targeted_development_manifest_matches_frozen_post_pilot_plan():
    document = yaml.safe_load(
        (ROOT / "data/manifests/targeted_development_v1.yaml").read_text(encoding="utf-8")
    )
    splits = yaml.safe_load(
        (ROOT / "data/manifests/splits.template.yaml").read_text(encoding="utf-8")
    )
    assert validate_targeted_campaign(document, splits) == []
    assert document["planning_source"] == "reports/pilot/post_pilot_campaign_size_v2.yaml"
    sizing = yaml.safe_load(
        (ROOT / document["planning_source"]).read_text(encoding="utf-8")
    )
    assert sizing["recommended_targeted_addition_episodes"] == 1212
    assert sizing[
        "maximum_fitting_episodes_before_supplement_if_all_research1_development_admitted"
    ] == 2685
    assert sizing["minimum_route_balanced_development_supplement"] == 324
    ordered = targeted_execution_order(expand_balanced_pilot(document))
    assert len(ordered) == 1212
    assert len({item["episode_key"] for item in ordered}) == 1212
    assert len({item["seed"] for item in ordered}) == 1212
    for start in range(0, len(ordered), 12):
        wave = ordered[start:start + 12]
        assert len({item["family"] for item in wave}) == 1
        assert len({(item["map"], item["route"]) for item in wave}) == 12


def test_targeted_collection_prerequisite_requires_final_validation_inventory(tmp_path):
    findings = targeted_prerequisite_findings(tmp_path)
    assert len(findings) == 3

    report = tmp_path / "reports/validation/balanced_validation_v1.cumulative324.yaml"
    report.parent.mkdir(parents=True)
    report.write_text(yaml.safe_dump({
        "complete_and_artifact_valid": True,
        "protected_test_used": False,
        "counts": {"expected": 324, "observed": 324, "usable": 324},
    }))
    inventory = tmp_path / "data/manifests/balanced_validation_v1.episodes.jsonl"
    inventory.parent.mkdir(parents=True)
    inventory.write_text("{}\n")
    dataset = tmp_path / "data/manifests/balanced_validation_v1.dataset.yaml"
    dataset.write_text(yaml.safe_dump({
        "dataset_id": "balanced_validation_v1-validation-324",
        "protected_test_used": False,
        "episode_inventory": {"rows": 324},
    }))
    assert targeted_prerequisite_findings(tmp_path) == []
