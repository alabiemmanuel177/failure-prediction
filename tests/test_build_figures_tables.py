import csv
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest
import yaml

from scripts.analyze_paired_recovery import analyse
from scripts.build_figures import FIGURES, build_all as build_figures
from scripts.build_recovery_campaign_manifest import (
    build_manifest, expand_paired_recovery, fake_split, held_out_map_routes,
)
from scripts.build_tables import TABLES, build_all as build_tables
from src.evaluation import AlarmPolicy, apply_alarm_policy
from src.reporting import ArtifactRegistry


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("lidar_dropout", "wheel_slip", "camera_occlusion")


def test_held_out_tables_are_discovered_from_the_confirmatory_report(tmp_path):
    root = tmp_path / "repo"
    table = root / "elsewhere" / "p3_held_out_alarmed.csv"
    write_csv(table, synthetic_predictions("held_out_map_test", models=("p3_causal_tcn",)))
    (root / "reports/confirmatory").mkdir(parents=True)
    (root / "reports/confirmatory/held_out_map.yaml").write_text(yaml.safe_dump({
        "complete": True, "inputs_sha256": {str(table): "0" * 64}}))
    registry = ArtifactRegistry(root)
    assert registry.available("predictions_held_out") is True
    assert registry.available("predictions_validation") is False
    rows = registry.prediction_rows("predictions_held_out")
    assert {row["model_id"] for row in rows} == {"p3_causal_tcn"}
    assert any(key.startswith("predictions_held_out") for key in registry.used)


def synthetic_predictions(split, models=("p3_causal_tcn", "p1_threshold_rules"), seed=3,
                          extra=None, maps=("ho_00", "ho_01")):
    """Alarmed prediction tables in the frozen contract, one episode set per model."""
    rng = random.Random(seed)
    rows = []
    for model in models:
        skill = 0.9 if model == "p3_causal_tcn" else 0.5
        for map_index, map_id in enumerate(maps):
            for route in range(2):
                for family in ("none", *FAMILIES):
                    for replicate in range(2):
                        run_id = f"{map_id}-r{route}-{family}-{replicate}"
                        event = 26.0 if family != "none" and (replicate == 0 or rng.random() < 0.7) else None
                        decisions = []
                        for step in range(60):
                            t = step * 0.5
                            eligibility = "eligible_negative"
                            if event is not None:
                                if event - 10 <= t <= event - 1:
                                    eligibility = "eligible_positive"
                                elif t > event - 1:
                                    eligibility = "excluded_too_late"
                                elif t > event - 20:
                                    eligibility = "excluded_near_event"
                            base = 0.05 + 0.1 * rng.random()
                            missed = family == "camera_occlusion" and replicate == 0 and map_index == 0
                            if eligibility == "eligible_positive" and rng.random() < skill and not missed:
                                base = 0.6 + 0.4 * rng.random()
                            if family == "none" and rng.random() < (0.01 if skill > 0.6 else 0.05):
                                base = 0.95
                            if family == "none" and replicate == 1 and 10 <= step <= 12:
                                base = 0.97  # guaranteed false alarm for every model
                            decisions.append({
                                "run_id": run_id, "decision_index": step, "decision_time": t, "split": split,
                                "map_id": map_id, "route_id": f"{map_id}_r{route}", "fault_family": family,
                                "severity": "medium" if family != "none" else "none", "seed": replicate,
                                "protected_test_used": str(split == "held_out_map_test").lower(),
                                "eligibility": eligibility, "label": int(eligibility == "eligible_positive"),
                                "primary_event_class": "collision" if event else "",
                                "primary_event_time": "" if event is None else event, "model_id": model,
                                "raw_score": round(min(1.0, base * 1.1), 4), "risk_score": round(base, 4),
                                **(extra or {}),
                            })
                        rows.extend(apply_alarm_policy(decisions, AlarmPolicy(0.5)))
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def paired_recovery_report(seed=2):
    splits = fake_split(maps=3, routes_per_map=2)
    manifest = build_manifest(held_out_map_routes(splits, 2), seeds_per_cell=1, seed_base=1,
                              gate={"frozen": False}, selector_model="m.json")
    expected = expand_paired_recovery(manifest)
    rng = random.Random(seed)
    rows = []
    for episode in expected:
        policy = episode["recovery_policy_id"]
        rows.append({"map_id": episode["map"], "route_id": episode["route"], "seed": str(episode["seed"]),
                     "fault_family": episode["family"], "severity": "medium", "policy_id": policy,
                     "mission_complete": str(rng.random() < (0.8 if policy == "R3" else 0.5)).lower(),
                     "collision": "false", "guard_violation": "false", "guard_rejected": "false",
                     "added_time_seconds": "2.0", "added_path_length_m": "0.5", "intervention_count": "1",
                     "recovery_action": rng.choice(["backup", "wait"]), "action_regret_vs_oracle": "1.0",
                     "oracle_action": "backup"})
    return {"complete": True, "protected_test_used": True,
            **analyse(rows, expected, replicates=20, seed=1, collision_margin=0.0)}


@pytest.fixture
def artifact_root(tmp_path):
    root = tmp_path / "repo"
    for name in ("failure_events", "alarm_policy", "recovery_guards", "recovery_costs"):
        target = root / "configs" / f"{name}.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((ROOT / "configs" / f"{name}.yaml").read_text())
    alarm = yaml.safe_load((root / "configs/alarm_policy.yaml").read_text())
    alarm["threshold"] = 0.5  # a frozen numeric threshold, as after scripts/freeze_model.py
    (root / "configs/alarm_policy.yaml").write_text(yaml.safe_dump(alarm))
    # one alarm-applied table per model, as the parallel confirmatory tooling produces
    for split, directory, maps in (("validation", "validation", ("val_00", "val_01")),
                                   ("held_out_map_test", "held_out", ("ho_00", "ho_01"))):
        for model in ("p3_causal_tcn", "p1_threshold_rules"):
            write_csv(root / "reports/predictions" / directory / f"{model}_alarmed.csv",
                      synthetic_predictions(split, models=(model,), maps=maps))
    # unseen-family folds: <work_root>/<family>/held_out_p3_alarmed.csv
    unseen = synthetic_predictions("held_out_map_test", models=("p3_causal_tcn",), seed=9)
    for family in FAMILIES:
        write_csv(root / "reports/unseen_family" / family / "held_out_p3_alarmed.csv",
                  [row for row in unseen if row["fault_family"] in {family, "none"}])
    # ablations: calibrated tables without alarm columns; the frozen policy is applied on load
    for name in ("full", "no_localisation", "single_timestamp"):
        rows = synthetic_predictions("validation", models=("p3_causal_tcn",), seed=11, maps=("val_00", "val_01"))
        for row in rows:
            row.pop("alarm"); row.pop("persistent")
        write_csv(root / "reports/predictions/ablations" / f"{name}.calibrated.csv", rows)
    report_path = root / "reports/recovery/paired_recovery.yaml"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(yaml.safe_dump(paired_recovery_report(), sort_keys=False))
    latency = root / "reports/latency/p3_causal_tcn.json"
    latency.parent.mkdir(parents=True)
    latency.write_text(json.dumps({"model_id": "p3_causal_tcn",
                                   "end_to_end": {"median_ms": 1.2, "p95_ms": 2.5, "max_ms": 4.0}}))
    return root


def test_all_minimum_figures_are_produced_with_sha256_sidecars(artifact_root):
    out = artifact_root / "reports/figures"
    index = build_figures(ArtifactRegistry(artifact_root), out)
    assert index["pending"] == {}
    assert set(index["produced"]) == set(FIGURES)
    trace_files = index["produced"]["fig02_risk_traces"]
    assert len(trace_files) == 3
    for name, files in index["produced"].items():
        for file in files:
            sidecar = json.loads((out / (file + ".json")).read_text())
            assert sidecar["fabricated_inputs"] is False and sidecar["sources"]
            assert all(len(source["sha256"]) == 64 for source in sidecar["sources"].values())
    assert (out / "fig01_architecture_causal_timeline.svg").read_text().startswith("<svg")


def test_figures_never_fabricate_missing_inputs_and_never_overwrite(tmp_path):
    root = tmp_path / "empty"
    (root / "configs").mkdir(parents=True)
    for name in ("failure_events", "alarm_policy", "recovery_guards"):
        (root / "configs" / f"{name}.yaml").write_text((ROOT / "configs" / f"{name}.yaml").read_text())
    out = root / "figures"
    index = build_figures(ArtifactRegistry(root), out)
    assert set(index["produced"]) == {"fig01_architecture_causal_timeline"}
    assert set(index["pending"]) == set(FIGURES) - {"fig01_architecture_causal_timeline"}
    assert all("PENDING" not in reason and "predictions" in reason or "paired_recovery" in reason
               for reason in index["pending"].values())
    with pytest.raises(FileExistsError):
        build_figures(ArtifactRegistry(root), out, ["fig01_architecture_causal_timeline"])


def test_all_tables_are_produced_as_csv_and_markdown(artifact_root):
    out = artifact_root / "reports/tables"
    index = build_tables(ArtifactRegistry(artifact_root), out)
    assert index["pending"] == {} and set(index["produced"]) == set(TABLES)
    with (out / "tab01_predictor_summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (out / "tab04_ablations.csv").open(newline="") as stream:
        ablations = {row["ablation"] for row in csv.DictReader(stream)}
    assert ablations == {"full", "no_localisation", "single_timestamp"}
    with (out / "tab07_latency.csv").open(newline="") as stream:
        assert next(csv.DictReader(stream))["median_ms"] == "1.2"
    with (out / "tab03_unseen_family.csv").open(newline="") as stream:
        assert {row["excluded_family"] for row in csv.DictReader(stream)} <= set(FAMILIES)
    assert {row["model_id"] for row in rows} == {"p3_causal_tcn", "p1_threshold_rules"}
    assert {row["split"] for row in rows} == {"validation", "held_out_map_test"}
    p3 = next(row for row in rows if row["model_id"] == "p3_causal_tcn" and row["split"] == "held_out_map_test")
    assert float(p3["event_recall"]) > 0.5
    markdown = (out / "tab05_recovery_outcomes.md").read_text()
    assert "R3_vs_R0" in markdown and "| comparison |" in markdown
    sidecar = json.loads((out / "tab05_recovery_outcomes.csv.json").read_text())
    assert "paired_recovery" in sidecar["sources"]


def test_table_cli_never_writes_into_the_repository_when_given_another_output_dir(tmp_path):
    """Running against the real repository with a temporary output directory must leave
    reports/tables untouched (whether or not the released tables already exist)."""
    released = ROOT / "reports/tables"
    before = {p: p.stat().st_mtime_ns for p in released.rglob("*")} if released.exists() else {}
    result = subprocess.run([sys.executable, str(ROOT / "scripts/build_tables.py"), "--output-dir", str(tmp_path / "t")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    after = {p: p.stat().st_mtime_ns for p in released.rglob("*")} if released.exists() else {}
    assert after == before
    assert (tmp_path / "t" / "table_index.json").exists()
