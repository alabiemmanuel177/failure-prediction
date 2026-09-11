from collections import Counter
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from scripts.build_recovery_campaign_manifest import (
    RECOVERY_POLICIES, build_manifest, expand_paired_recovery, fake_split, held_out_map_routes,
    validate_paired_recovery,
)


ROOT = Path(__file__).resolve().parents[1]


def manifest():
    splits = fake_split()
    return build_manifest(held_out_map_routes(splits, 8), seeds_per_cell=3, seed_base=3_000_000,
                          gate={"frozen": False}, selector_model="models/x.json"), splits


def test_design_is_paired_across_policies_and_meets_the_1008_floor():
    document, splits = manifest()
    assert document["expected_base_episode_count"] == 504
    assert document["expected_episode_count"] == 1512  # R0, R2, R3 under PA-2026-09-03-04
    assert document["allowed_splits"] == ["held_out_map_test"]
    assert document["protected_test_used"] is True
    assert validate_paired_recovery(document, splits) == []
    episodes = expand_paired_recovery(document)
    assert Counter(item["recovery_policy_id"] for item in episodes) == {p: 504 for p in RECOVERY_POLICIES}
    pairs = {}
    for item in episodes:
        pairs.setdefault(item["pair_key"], []).append((item["seed"], item["family"], item["recovery_policy_id"]))
    assert len(pairs) == 504
    assert all(len({seed for seed, _f, _p in members}) == 1 for members in pairs.values())
    assert len({item["episode_key"] for item in episodes}) == 1512
    assert episodes[0]["episode_key"].endswith("-R0") and episodes[2]["episode_key"].endswith("-R3")


def test_validation_rejects_maps_outside_the_held_out_split():
    document, splits = manifest()
    splits["held_out_map_test"]["maps"] = ["other"]
    findings = validate_paired_recovery(document, splits)
    assert any("outside held_out_map_test" in finding for finding in findings)


def test_split_must_assign_three_maps_with_enough_routes():
    with pytest.raises(ValueError, match="three maps"):
        held_out_map_routes(fake_split(maps=2), 8)
    with pytest.raises(ValueError, match="routes"):
        held_out_map_routes(fake_split(routes_per_map=6), 8)


def test_cli_fails_closed_without_freeze_and_allows_fake_dry_run(tmp_path):
    script = ROOT / "scripts/build_recovery_campaign_manifest.py"
    # An unfrozen freeze record (the template) blocks the manifest whatever the workspace holds.
    unfrozen = tmp_path / "model_freeze.yaml"
    shutil.copy(ROOT / "configs/model_freeze.template.yaml", unfrozen)
    assert yaml.safe_load(unfrozen.read_text())["frozen"] is False
    blocked = subprocess.run([sys.executable, str(script), "--dry-run", "--freeze", str(unfrozen)],
                             capture_output=True, text=True)
    assert blocked.returncode != 0 and "not frozen" in blocked.stderr
    missing = subprocess.run([sys.executable, str(script), "--dry-run", "--freeze", str(tmp_path / "absent.yaml")],
                             capture_output=True, text=True)
    assert missing.returncode != 0 and "not frozen" in missing.stderr
    fake = subprocess.run([sys.executable, str(script), "--dry-run", "--fake-split"], capture_output=True, text=True)
    assert fake.returncode == 0, fake.stderr
    assert '"episodes": 1512' in fake.stdout
    # Post-freeze the real manifest exists; it must validate and carry the phased amendment.
    real = ROOT / "data/manifests/paired_recovery_v1.yaml"
    if real.exists():
        document = yaml.safe_load(real.read_text(encoding="utf-8"))
        assert document["protected_test_used"] is True
        assert document["parallel_execution_admitted_by"] == "PA-2026-09-04-02"
        assert document["recovery_policies"] == ["R0", "R2", "R3"]
    output = tmp_path / "paired.yaml"
    written = subprocess.run([sys.executable, str(script), "--dry-run", "--fake-split", "--output", str(output)],
                             capture_output=True, text=True)
    assert written.returncode == 0, written.stderr
    document = yaml.safe_load(output.read_text())
    assert document["confirmatory_gate"]["dry_run_fake_split"] is True
    refused = subprocess.run([sys.executable, str(script), "--fake-split"], capture_output=True, text=True)
    assert refused.returncode != 0
