from collections import Counter
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from confirmatory_fixtures import ROOT, assigned_splits, fake_repository, unassigned_splits, unassigned_splits_text
from scripts.build_confirmatory_manifests import (
    held_out_manifest, held_out_map_routes, main, severity_stress_manifest,
)
from src.experiments import (
    expand_balanced_pilot, targeted_execution_order, validate_confirmatory_campaign,
)
from src.protected_data import held_out_campaign_gate


def existing_seeds() -> set[int]:
    seeds: set[int] = set()
    for name in ("balanced_pilot_v1", "balanced_validation_v1", "targeted_development_v1"):
        document = yaml.safe_load((ROOT / f"data/manifests/{name}.yaml").read_text(encoding="utf-8"))
        seeds |= {int(item["seed"]) for item in expand_balanced_pilot(document)}
    return seeds


def test_held_out_and_severity_designs_expand_exactly_without_collisions():
    splits = assigned_splits()
    map_routes = held_out_map_routes(splits)
    held_out = held_out_manifest(map_routes, {})
    stress = severity_stress_manifest(map_routes, {})
    assert validate_confirmatory_campaign(held_out, splits) == []
    assert validate_confirmatory_campaign(stress, splits) == []
    held_out_episodes = expand_balanced_pilot(held_out)
    stress_episodes = expand_balanced_pilot(stress)
    assert len(held_out_episodes) == held_out["expected_episode_count"] == 960
    assert len(stress_episodes) == stress["expected_episode_count"] == 1512
    for episodes in (held_out_episodes, stress_episodes):
        assert len({item["episode_key"] for item in episodes}) == len(episodes)
        assert len({item["seed"] for item in episodes}) == len(episodes)
        assert all(item["map"].startswith("test_") for item in episodes)
    held_out_seeds = {item["seed"] for item in held_out_episodes}
    stress_seeds = {item["seed"] for item in stress_episodes}
    assert held_out_seeds.isdisjoint(stress_seeds)
    assert (held_out_seeds | stress_seeds).isdisjoint(existing_seeds())
    assert Counter(item["family"] for item in held_out_episodes) == {
        "none": 120, **{family: 120 for family in (
            "camera_occlusion", "lidar_dropout", "wheel_slip", "localisation_perturbation",
            "dynamic_blockage", "planner_oscillation", "semantic_corruption")},
    }
    assert Counter(item["system"] for item in held_out_episodes if item["family"] == "none") == {
        "s0": 72, "s3": 48,
    }
    assert Counter(item["severity"] for item in stress_episodes) == {
        "low": 504, "medium": 504, "high": 504,
    }
    assert all(item["severity"] == "medium" for item in held_out_episodes if item["family"] != "none")
    assert held_out["allowed_splits"] == stress["allowed_splits"] == ["held_out_map_test"]
    assert held_out["protected_test_used"] is stress["protected_test_used"] is True
    assert held_out["execution_policy"]["exact_once_per_episode_key"] is True
    assert held_out["execution_policy"]["minimum_free_space_gib_before_episode"] == 100


def test_waves_cover_one_condition_and_replicate_across_all_routes():
    splits = assigned_splits()
    held_out = held_out_manifest(held_out_map_routes(splits), {})
    ordered = targeted_execution_order(expand_balanced_pilot(held_out))
    wave = held_out["execution_policy"]["maximum_episodes_per_invocation"]
    assert wave == 24 and len(ordered) % wave == 0
    first = ordered[:wave]
    assert len({(item["map"], item["route"]) for item in first}) == 24
    assert len({item["episode_key"].rsplit("-", 2)[-2] for item in first}) == 1


def test_validator_rejects_unassigned_split_and_pre_protected_flags():
    splits = assigned_splits()
    held_out = held_out_manifest(held_out_map_routes(splits), {})
    findings = validate_confirmatory_campaign(held_out, unassigned_splits())
    assert "held_out_map_test split is not assigned after the model freeze" in findings
    assert any("outside held_out_map_test split" in item for item in findings)
    wrong = dict(held_out, protected_test_used=False, allowed_splits=["development"])
    findings = validate_confirmatory_campaign(wrong, splits)
    assert "confirmatory campaigns must declare protected_test_used: true" in findings
    assert "confirmatory campaigns must declare allowed_splits [held_out_map_test]" in findings


def test_build_script_dry_run_checks_real_campaigns_and_writes_nothing(tmp_path, capsys):
    splits_path = tmp_path / "splits.yaml"
    splits_path.write_text(yaml.safe_dump(assigned_splits()), encoding="utf-8")
    # Seed/key collisions are checked against copies of every real campaign manifest;
    # the two confirmatory manifests the builder itself writes are left out so the
    # dry run is not refused as an overwrite once they exist in the workspace.
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    real = sorted((ROOT / "data/manifests").glob("*.yaml"))
    copied = 0
    for path in real:
        if path.name not in {"held_out_map_v1.yaml", "severity_stress_v1.yaml"}:
            shutil.copy(path, manifest_dir / path.name)
            copied += 1
    assert copied >= 3
    before = sorted(path.name for path in (ROOT / "data/manifests").glob("*.yaml"))
    assert main(["--splits", str(splits_path), "--manifest-dir", str(manifest_dir), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert '"expanded": 960' in output and '"expanded": 1512' in output
    assert '"findings": []' in output
    assert sorted(path.name for path in (ROOT / "data/manifests").glob("*.yaml")) == before
    assert sorted(path.name for path in manifest_dir.glob("*.yaml")) == sorted(
        path.name for path in real if path.name not in {"held_out_map_v1.yaml", "severity_stress_v1.yaml"}
    )
    unassigned = tmp_path / "unassigned.yaml"
    unassigned.write_text(unassigned_splits_text(), encoding="utf-8")
    with pytest.raises(SystemExit, match="assigned held-out split"):
        main(["--splits", str(unassigned), "--manifest-dir", str(manifest_dir), "--dry-run"])


def test_build_script_writes_immutable_manifests(tmp_path):
    splits_path = tmp_path / "splits.yaml"
    splits_path.write_text(yaml.safe_dump(assigned_splits()), encoding="utf-8")
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    assert main(["--splits", str(splits_path), "--manifest-dir", str(manifest_dir)]) == 0
    held_out = yaml.safe_load((manifest_dir / "held_out_map_v1.yaml").read_text(encoding="utf-8"))
    assert held_out["expected_episode_count"] == 960
    assert held_out["provenance"]["split_manifest_sha256"]
    assert main(["--splits", str(splits_path), "--manifest-dir", str(manifest_dir)]) == 1


def test_gate_requires_freeze_and_assignment(tmp_path):
    ready = fake_repository(tmp_path / "ready")
    assert held_out_campaign_gate(ready, ["test_00"], run_readiness=False) == []
    assert held_out_campaign_gate(ready, ["test_09"], run_readiness=False) == [
        "map test_09 is not an assigned held-out map"
    ]
    unfrozen = fake_repository(tmp_path / "unfrozen", frozen=False)
    assert "model freeze missing: configs/model_freeze.yaml" in held_out_campaign_gate(
        unfrozen, ["test_00"], run_readiness=False
    )
    unassigned = fake_repository(tmp_path / "unassigned", assigned=False)
    assert any("status" in item for item in held_out_campaign_gate(unassigned, run_readiness=False))


def test_runners_refuse_held_out_manifest_before_freeze(monkeypatch, tmp_path):
    """Both runners fail closed on the freeze/assignment gate, independent of the workspace.

    The gate is exercised against a fake pre-freeze repository (unassigned split, no
    freeze record) so the test neither depends on nor touches the real campaign state.
    """
    from scripts import run_balanced_pilot as child_runner
    from scripts import run_balanced_pilot_continuous as continuous_runner

    splits = assigned_splits()
    manifest = tmp_path / "held_out_map_v1.yaml"
    manifest.write_text(yaml.safe_dump(held_out_manifest(held_out_map_routes(splits), {})), encoding="utf-8")
    unfrozen_root = fake_repository(tmp_path / "unfrozen", frozen=False, assigned=False)
    (unfrozen_root / "logs/campaigns").mkdir(parents=True)

    seen: list[tuple] = []

    def gate(root, map_ids=(), *, run_readiness=True):
        seen.append((Path(root), tuple(map_ids), run_readiness))
        return held_out_campaign_gate(unfrozen_root, map_ids, run_readiness=False)

    monkeypatch.setattr(continuous_runner, "held_out_campaign_gate", gate)
    monkeypatch.setattr(sys, "argv", ["run_balanced_pilot_continuous.py", "--manifest", str(manifest), "--dry-run"])
    with pytest.raises(SystemExit, match="refused before the model freeze"):
        continuous_runner.main()
    assert seen and seen[0][0] == ROOT and set(seen[0][1]) == set(held_out_map_routes(splits))

    monkeypatch.setattr(child_runner, "ROOT", unfrozen_root)
    monkeypatch.setattr(sys, "argv", ["run_balanced_pilot.py", "--manifest", str(manifest), "--dry-run"])
    with pytest.raises(SystemExit, match="not assigned after the model freeze"):
        child_runner.main()


def test_episode_runner_gates_protected_maps(monkeypatch, tmp_path):
    from scripts import run_research2_episode as runner

    lock = yaml.safe_load((ROOT / "integration/research1.lock.yaml").read_text(encoding="utf-8"))
    monkeypatch.setattr(runner, "validate_platform", lambda _lock: (Path("/r1"), "sha"))
    monkeypatch.setattr(runner, "held_out_campaign_gate", lambda root, maps: ["model freeze missing"])
    with pytest.raises(SystemExit, match="protected until the model freeze"):
        runner.validate_boundary(lock, "test_00")
    monkeypatch.setattr(runner, "held_out_campaign_gate", lambda root, maps: [])
    assert runner.validate_boundary(lock, "test_00") == (Path("/r1"), "test", "sha")
    assert runner.split_record("test") == ("held_out_map_test", True)
    assert runner.split_record("development") == ("development", False)
    with pytest.raises(SystemExit, match="forbidden or unknown"):
        runner.validate_boundary(lock, "other_00")


def test_six_route_amendment_keeps_protocol_minimums():
    from scripts.build_confirmatory_manifests import (
        FAMILIES, SEVERITIES, held_out_manifest, seeds_for_minimum, severity_stress_manifest,
    )
    map_routes = {f"test_0{m}": [f"test_0{m}_r{r}" for r in range(6)] for m in range(3)}
    routes = 18
    held_seeds = seeds_for_minimum(routes, len(FAMILIES) + 1, 960)
    stress_seeds = seeds_for_minimum(routes, len(FAMILIES) * len(SEVERITIES), 1512)
    assert (held_seeds, stress_seeds) == (7, 4)
    held = held_out_manifest(map_routes, {}, seeds=held_seeds)
    stress = severity_stress_manifest(map_routes, {}, seeds=stress_seeds)
    assert held["expected_episode_count"] == 1008 >= 960
    assert stress["expected_episode_count"] == 1512
    # eight routes reproduce the original protocol matrix exactly
    assert seeds_for_minimum(24, 8, 960) == 5 and seeds_for_minimum(24, 21, 1512) == 3
