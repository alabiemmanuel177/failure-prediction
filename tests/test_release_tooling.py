import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest
import yaml

from scripts.build_demo_storyboard import build_storyboard, load_recovery_log
from scripts.build_release_report import build_release_document
from scripts.fill_manuscript_results import MARKERS, apply, generate
from scripts.finalize_cards import collect, dataset_card_section, finalize_text, model_card_section
from src.release import PACKAGE_FINDINGS, blocking_findings
from src.reporting import ArtifactRegistry
from test_build_figures_tables import paired_recovery_report, synthetic_predictions, write_csv


ROOT = Path(__file__).resolve().parents[1]


def test_blocking_findings_tolerate_only_package_self_references():
    findings = [PACKAGE_FINDINGS["release"], PACKAGE_FINDINGS["model_card"], "model freeze evidence missing"]
    assert blocking_findings(findings, ["release", "model_card"]) == ["model freeze evidence missing"]
    assert blocking_findings(findings, ["release"]) == [PACKAGE_FINDINGS["model_card"], "model freeze evidence missing"]


def test_release_document_passes_only_with_clean_audit_and_passed_rerun(tmp_path):
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    (root / "configs/a.yaml").write_text("x: 1\n")
    (root / "reports/reproduction").mkdir(parents=True)
    output = root / "reports/reproduction/release.yaml"
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    blocked = build_release_document(root, output, audit=lambda _root: [PACKAGE_FINDINGS["release"]])
    assert blocked["passed"] is False and "independent rerun" in blocked["blocking_findings"][0]
    (root / "reports/reproduction/independent_rerun.yaml").write_text("passed: true\n")
    passed = build_release_document(root, output, audit=lambda _root: [PACKAGE_FINDINGS["release"]])
    assert passed["passed"] is True and passed["checksums"]["configs"] == {"configs/a.yaml": passed["checksums"]["configs"]["configs/a.yaml"]}
    assert passed["checksum_counts"]["reports"] == 1  # the rerun record, never the release file itself
    failed = build_release_document(root, output, audit=lambda _root: ["model freeze evidence missing"])
    assert failed["passed"] is False
    dry = build_release_document(root, output, audit=lambda _root: [], dry_run=True)
    assert dry["passed"] is False and dry["completion_audit"]["dry_run_skipped_audit"] is True


def test_release_report_cli_refuses_to_overwrite_and_dry_run_writes_nothing(tmp_path):
    output = tmp_path / "release.yaml"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/build_release_report.py"), "--dry-run",
                             "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "checksum_counts" in result.stdout and not output.exists()
    output.write_text("passed: false\n")
    refused = subprocess.run([sys.executable, str(ROOT / "scripts/build_release_report.py"), "--output", str(output)],
                             capture_output=True, text=True)
    assert refused.returncode != 0 and "refusing" in refused.stderr


def test_reproduce_release_dry_run_prints_the_clean_room_plan():
    result = subprocess.run([sys.executable, str(ROOT / "scripts/reproduce_release.py"), "--dry-run"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    ids = [step["id"] for step in plan["steps"]]
    assert ids[:5] == ["tests", "development_inventory", "validation_inventory", "targeted_inventory", "raw_payload_audit"]
    assert "predict_held_out_map" in ids and ids[-1] == "tables"
    assert plan["shared_inputs_symlinked"] == ["data/raw", "data/derived", "models"]
    assert not (ROOT / "reports/reproduction").exists()


def test_storyboard_produces_frames_shots_and_documented_assembly(tmp_path):
    rows = [row for row in synthetic_predictions("held_out_map_test", models=("p3_causal_tcn",))
            if row["run_id"] == "ho_00-r0-lidar_dropout-0"]
    log = tmp_path / "recovery.csv"
    with log.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["time_seconds", "event_type", "action", "detail"])
        writer.writeheader()
        writer.writerow({"time_seconds": 18.0, "event_type": "recovery_action_executed", "action": "backup", "detail": "guarded"})
    out = tmp_path / "demo"
    shot_list = build_storyboard(rows, load_recovery_log(log), out, demo_seconds=90.0, fps=2.0, sources={})
    assert shot_list["frame_count"] == 161 and shot_list["video_assembled"] is False
    assert (out / "frames/frame_0000.svg").exists() and (out / "timeline.svg").exists()
    shots = [shot["shot"] for shot in shot_list["shots"]]
    assert shots[0] == "title" and "recovery_action" in shots and shots[-1] == "closing"
    assert shot_list["event_time_s"] == 26.0
    assert "ffmpeg" in (out / "ASSEMBLY.md").read_text()
    assert json.loads((out / "shot_list.json").read_text())["run_id"] == "ho_00-r0-lidar_dropout-0"
    with pytest.raises(FileExistsError):
        build_storyboard(rows, [], out, demo_seconds=90.0, fps=2.0, sources={})


def test_storyboard_cli_rejects_unknown_run(tmp_path):
    table = tmp_path / "p.csv"
    write_csv(table, synthetic_predictions("held_out_map_test", models=("p3_causal_tcn",)))
    result = subprocess.run([sys.executable, str(ROOT / "scripts/build_demo_storyboard.py"), str(table), "nope",
                             "--output-dir", str(tmp_path / "demo")], capture_output=True, text=True)
    assert result.returncode != 0 and "run_id not found" in result.stderr


def _draft_card(text: str) -> str:
    """Reset a (possibly already finalised) card to its pre-training contract state."""
    text = re.sub(r"^Status: .*$", "Status: pre-training contract", text, count=1, flags=re.M)
    marker = "\n## Final record"
    return text.split(marker)[0].rstrip() + "\n" if marker in text else text


def test_finalize_cards_flips_status_once_and_appends_generated_record(tmp_path):
    facts = collect(ROOT)
    text = _draft_card((ROOT / "docs/model-card.md").read_text())
    final = finalize_text(text, model_card_section(facts), facts)
    assert "Status: final" in final and "## Final record" in final and "pending" in final
    assert "Status: pre-training contract" not in final
    with pytest.raises(ValueError, match="already final"):
        finalize_text(final, model_card_section(facts), facts)
    dataset = finalize_text(_draft_card((ROOT / "docs/dataset-card.md").read_text()), dataset_card_section(facts), facts)
    assert dataset.count("Status:") == 1 and "Status: final" in dataset


def test_finalize_cards_real_run_is_blocked_by_the_completion_audit(tmp_path):
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "docs/model-card.md").write_text(_draft_card((ROOT / "docs/model-card.md").read_text()))
    (root / "docs/dataset-card.md").write_text(_draft_card((ROOT / "docs/dataset-card.md").read_text()))
    shutil.copy(ROOT / "scripts/audit_project_completion.py", root / "scripts/audit_project_completion.py")
    shutil.copytree(ROOT / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
    before = (root / "docs/model-card.md").read_text()
    result = subprocess.run([sys.executable, str(ROOT / "scripts/finalize_cards.py"), "--root", str(root)],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "stay non-final" in result.stderr
    assert (root / "docs/model-card.md").read_text() == before
    dry = subprocess.run([sys.executable, str(ROOT / "scripts/finalize_cards.py"), "--root", str(root), "--dry-run"],
                         capture_output=True, text=True)
    assert dry.returncode == 0 and "Status: final" in dry.stdout
    assert (root / "docs/model-card.md").read_text() == before


def test_manuscript_filler_replaces_every_marker_from_reports(tmp_path):
    root = tmp_path / "repo"
    for model in ("p3_causal_tcn", "p1_threshold_rules"):
        write_csv(root / "reports/predictions/held_out" / f"{model}_alarmed.csv",
                  synthetic_predictions("held_out_map_test", models=(model,)))
    (root / "reports/recovery").mkdir(parents=True)
    (root / "reports/recovery/paired_recovery.yaml").write_text(yaml.safe_dump(paired_recovery_report(), sort_keys=False))
    (root / "reports/reproduction").mkdir()
    (root / "reports/reproduction/independent_rerun.yaml").write_text(yaml.safe_dump({
        "passed": True, "git_commit": "abc123def456", "finished_utc": "2026-10-01T00:00:00Z",
        "diffs": {"tab01_predictor_summary": {"status": "identical_bytes"}}}))
    (root / "reports/tables").mkdir()
    (root / "reports/tables/tab03_unseen_family.csv").write_text(
        "model_id,excluded_family,episodes,events,detected,event_recall,false_alerts_per_mission\n"
        "p3_causal_tcn,lidar_dropout,4,3,2,0.6667,0.0\n")
    generated = generate(root, ArtifactRegistry(root))
    assert not any(marker in value for value in generated.values() for marker in MARKERS)
    assert "percentage points" in generated["abstract"] and "H6" in generated["abstract"]
    filled = apply((ROOT / "manuscript/main.md").read_text(), generated)
    assert not any(marker in filled for marker in MARKERS)
    assert "## 4. Results" in filled and "## 5. Discussion" in filled and "## 6. Limitations" in filled
    assert "independent clean rerun" in filled


def test_manuscript_filler_keeps_markers_when_reports_are_missing_and_never_edits_without_audit(tmp_path):
    root = tmp_path / "repo"
    (root / "manuscript").mkdir(parents=True)
    shutil.copy(ROOT / "manuscript/main.md", root / "manuscript/main.md")
    shutil.copytree(ROOT / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (root / "scripts").mkdir()
    shutil.copy(ROOT / "scripts/audit_project_completion.py", root / "scripts/audit_project_completion.py")
    before = (root / "manuscript/main.md").read_text()
    generated = generate(root, ArtifactRegistry(root))
    assert generated["abstract"].startswith("RESULT_PENDING") and generated["release"] == "RELEASE_PENDING"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/fill_manuscript_results.py"), "--root", str(root)],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "pending markers" in result.stderr
    assert (root / "manuscript/main.md").read_text() == before
    dry = subprocess.run([sys.executable, str(ROOT / "scripts/fill_manuscript_results.py"), "--root", str(root), "--dry-run"],
                         capture_output=True, text=True)
    assert dry.returncode == 0 and "RESULT_PENDING" in dry.stdout
