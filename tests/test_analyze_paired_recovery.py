import csv
from pathlib import Path
import random
import subprocess
import sys

import pytest
import yaml

from scripts.analyze_paired_recovery import analyse, read_outcomes
from scripts.build_recovery_campaign_manifest import (
    build_manifest, expand_paired_recovery, fake_split, held_out_map_routes,
)


ROOT = Path(__file__).resolve().parents[1]


def small_manifest():
    splits = fake_split(maps=3, routes_per_map=2)
    return build_manifest(held_out_map_routes(splits, 2), seeds_per_cell=1, seed_base=10,
                          gate={"frozen": False}, selector_model="m.json")


def outcome_rows(expected, seed=5, proposed_boost=0.7, violation=False):
    rng = random.Random(seed)
    rows = []
    for episode in expected:
        cell = rng.random()
        policy = episode["recovery_policy_id"]
        complete = cell < 0.3 or (policy == "R3" and rng.random() < proposed_boost)
        rows.append({
            "map_id": episode["map"], "route_id": episode["route"], "seed": str(episode["seed"]),
            "fault_family": episode["family"], "severity": episode["severity"], "policy_id": policy,
            "run_id": episode["episode_key"], "mission_complete": str(complete).lower(),
            "collision": "false", "guard_violation": str(violation and policy == "R3").lower(),
            "guard_rejected": str(rng.random() < 0.1).lower(),
            "added_time_seconds": f"{rng.uniform(0, 8):.2f}", "added_path_length_m": f"{rng.uniform(0, 2):.2f}",
            "intervention_count": "1", "recovery_action": rng.choice(["backup", "wait", "controlled_stop"]),
            "action_regret_vs_oracle": "" if policy == "R0" else f"{rng.uniform(0, 4):.2f}",
            "oracle_action": "backup",
        })
    return rows


def test_analysis_is_complete_only_with_every_pair_and_a_clean_safety_gate():
    expected = expand_paired_recovery(small_manifest())
    rows = outcome_rows(expected)
    report = analyse(rows, expected, replicates=50, seed=1, collision_margin=0.0)
    assert report["complete"] is True and report["incomplete_reasons"] == []
    assert report["comparisons"]["R3_vs_R0"]["pair_count"] == 42
    assert report["h6"]["supported"] in {True, False}
    assert report["comparisons"]["R3_vs_R0"]["mixed_effects_model"]["method"] in {
        "mixed_effects_logistic_variational_bayes_statsmodels", "fallback_hierarchical_bootstrap",
    }
    assert "R3" in report["action_confusion_vs_oracle"]
    assert report["regret_vs_oracle"]["R0"]["episodes_with_regret"] == 0
    missing = analyse(rows[:-1], expected, replicates=50, seed=1, collision_margin=0.0)
    assert missing["complete"] is False and missing["completeness"]["missing_cell_count"] == 1
    unsafe = analyse(outcome_rows(expected, violation=True), expected, replicates=50, seed=1, collision_margin=0.0)
    assert unsafe["safety_gate_passed"] is False and unsafe["complete"] is False


def test_fallback_is_recorded_when_mixed_model_cannot_fit():
    expected = [item for item in expand_paired_recovery(small_manifest()) if item["map"] == "fake_00"]
    rows = outcome_rows(expected, proposed_boost=1.0)
    for row in rows:
        row["mission_complete"] = "true"
    report = analyse(rows, None, replicates=20, seed=1, collision_margin=0.0)
    assert report["comparisons"]["R3_vs_R0"]["mixed_effects_model"]["method"] == "fallback_hierarchical_bootstrap"
    assert report["complete"] is False


def test_cli_requires_freeze_and_explicit_protected_approval_then_writes_once(tmp_path):
    expected = expand_paired_recovery(small_manifest())
    rows = outcome_rows(expected)
    table = tmp_path / "outcomes.csv"
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(small_manifest()))
    freeze = tmp_path / "freeze.yaml"
    freeze.write_text("frozen: false\n")
    output = tmp_path / "paired_recovery.yaml"
    command = [sys.executable, str(ROOT / "scripts/analyze_paired_recovery.py"), str(table),
               "--manifest", str(manifest_path), "--output", str(output), "--freeze", str(freeze),
               "--replicates", "30"]
    blocked = subprocess.run(command + ["--allow-protected-after-freeze"], capture_output=True, text=True)
    assert blocked.returncode != 0 and "confirmatory freeze" in blocked.stderr
    freeze.write_text("frozen: true\n")
    unapproved = subprocess.run(command, capture_output=True, text=True)
    assert unapproved.returncode != 0 and "explicit approval" in unapproved.stderr
    done = subprocess.run(command + ["--allow-protected-after-freeze"], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    report = yaml.safe_load(output.read_text())
    assert report["complete"] is True and report["protected_test_used"] is True
    assert report["inputs"]["outcomes_sha256"] and report["inputs"]["manifest_sha256"]
    again = subprocess.run(command + ["--allow-protected-after-freeze"], capture_output=True, text=True)
    assert again.returncode != 0 and "refusing to overwrite" in again.stderr


def test_read_outcomes_rejects_unknown_policy(tmp_path):
    table = tmp_path / "bad.csv"
    row = outcome_rows(expand_paired_recovery(small_manifest())[:1])[0]
    row["policy_id"] = "R9"
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row.keys()))
        writer.writeheader(); writer.writerow(row)
    with pytest.raises(ValueError, match="unknown recovery policies"):
        read_outcomes(table)
