#!/usr/bin/env python3
"""Advance the balanced pilot in exact-once waves until complete or safely stopped.

Each child invocation remains limited by the manifest wave size. After every complete
wave this controller creates immutable wave and cumulative reports and appends one
hash-chained research-log record. The child runner retains authority over extraction,
ROS-domain, Research 1 activity, artifact-validation, and free-space gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.protected_data import held_out_campaign_gate


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def logged_waves(path: Path, campaign_id: str) -> set[int]:
    result: set[int] = set()
    for row in jsonl(path):
        metadata = row.get("metadata", {})
        if metadata.get("campaign") == campaign_id and isinstance(metadata.get("wave"), int):
            result.add(int(metadata["wave"]))
    return result


def finalize_wave(
    *, campaign_id: str, wave: int, completed: int, manifest: Path,
    report_root: Path, log: Path, split_name: str,
) -> None:
    wave_report = report_root / f"{campaign_id}.wave{wave}.yaml"
    if not wave_report.exists():
        subprocess.run([
            sys.executable, str(ROOT / "scripts/report_balanced_pilot_wave.py"),
            "--wave", str(wave), "--manifest", str(manifest),
            "--output", str(wave_report),
        ], check=True)
    cumulative = report_root / f"{campaign_id}.cumulative{completed}.yaml"
    if not cumulative.exists():
        subprocess.run([
            sys.executable, str(ROOT / "scripts/summarize_balanced_pilot.py"),
            "--manifest", str(manifest), "--allow-incomplete", "--output", str(cumulative),
        ], check=True)
    if wave not in logged_waves(log, campaign_id):
        report = yaml.safe_load(wave_report.read_text(encoding="utf-8"))
        free_gib = shutil.disk_usage(ROOT).free / 2**30
        metadata = {
            "campaign": campaign_id,
            "wave": wave,
            "cumulative_episodes": completed,
            "episodes": report["counts"]["episodes"],
            "usable": report["counts"]["usable"],
            "invalid": report["counts"]["invalid"],
            "terminal_events": report["counts"]["terminal_events"],
            "successes": report["counts"]["successes"],
            "collisions": report["counts"]["collisions"],
            "event_prevalence": report["event_prevalence"],
            "bag_gib": report["bag_gib"],
            "recording_profiles": report["recording_profiles"],
            "perception_metric_summaries": report["perception_metric_summaries"],
            "free_space_gib_after": round(free_gib, 3),
            "protected_test_used": False,
            "training_admission": "human_gate_passed_dataset_campaign_in_progress",
            "report": str(wave_report.relative_to(ROOT)),
            "cumulative_report": str(cumulative.relative_to(ROOT)),
        }
        subprocess.run([
            sys.executable, str(ROOT / "scripts/research_log.py"), "add",
            "--kind", "experiment", "--actor", "Codex",
            "--message",
            f"Completed {campaign_id} wave {wave}: all {report['counts']['episodes']} "
            f"{split_name} episodes passed exact-once artifact validation; protected data "
            "remained untouched and training admission remained machine/data-gated.",
            "--metadata", json.dumps(metadata, sort_keys=True),
        ], check=True)
        subprocess.run([
            sys.executable, str(ROOT / "scripts/research_log.py"), "verify",
        ], check=True)


def finalize_confirmatory_wave(
    *, campaign_id: str, wave: int, completed: int, ledger: Path, log: Path,
) -> None:
    """Log a protected wave from the exact-once ledger only; no outcome summaries."""
    if wave in logged_waves(log, campaign_id):
        return
    rows = jsonl(ledger)[:completed]
    invalid = sum(1 for row in rows if row.get("returncode"))
    metadata = {
        "campaign": campaign_id,
        "wave": wave,
        "cumulative_episodes": completed,
        "invalid_attempts": invalid,
        "free_space_gib_after": round(shutil.disk_usage(ROOT).free / 2**30, 3),
        "protected_test_used": True,
        "protected_outcomes_consulted": False,
        "outcome_summaries_inspected": False,
    }
    subprocess.run([
        sys.executable, str(ROOT / "scripts/research_log.py"), "add",
        "--kind", "experiment", "--actor", "Codex",
        "--message",
        f"Completed {campaign_id} wave {wave}: {completed} protected held-out episodes "
        "attempted exact-once after the model freeze; outcomes were not inspected.",
        "--metadata", json.dumps(metadata, sort_keys=True),
    ], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/research_log.py"), "verify"], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/manifests/balanced_pilot_v1.yaml",
    )
    parser.add_argument("--max-waves", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_waves is not None and args.max_waves < 1:
        raise SystemExit("max-waves must be positive")
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    campaign_id = str(manifest["campaign_id"])
    allowed_splits = manifest.get("allowed_splits", [])
    confirmatory = (
        allowed_splits == ["held_out_map_test"]
        and manifest.get("campaign_kind") == "held_out_confirmatory"
    )
    if confirmatory:
        gate_findings = held_out_campaign_gate(
            ROOT, list(manifest.get("design", {}).get("map_routes", {})),
            run_readiness=not args.dry_run,
        )
        if gate_findings:
            raise SystemExit(
                "held_out_map_test campaign refused before the model freeze and split "
                "assignment:\n- " + "\n- ".join(gate_findings)
            )
    elif allowed_splits not in (["development"], ["validation"]):
        raise SystemExit("continuous campaign must contain exactly one pre-protected split")
    split_name = str(allowed_splits[0])
    protected = bool(manifest.get("protected_test_used"))
    expected = int(manifest["expected_episode_count"])
    wave_size = int(manifest["execution_policy"]["maximum_episodes_per_invocation"])
    if expected % wave_size:
        raise SystemExit("expected episode count must be divisible by wave size")
    ledger = ROOT / "logs/campaigns" / f"{campaign_id}.jsonl"
    report_root = ROOT / (
        "reports/pilot" if split_name == "development"
        else "reports/confirmatory" if confirmatory else "reports/validation"
    )
    research_log = ROOT / "logs/research-log.jsonl"
    completed = len(jsonl(ledger))
    remainder = completed % wave_size
    state = {
        "campaign_id": campaign_id,
        "completed": completed,
        "remaining": expected - completed,
        "completed_waves": completed // wave_size,
        "total_waves": expected // wave_size,
        "protected_test_used": protected,
    }
    print(json.dumps(state, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return 0

    def finalize(wave: int, completed_count: int) -> None:
        if confirmatory:
            finalize_confirmatory_wave(
                campaign_id=campaign_id, wave=wave, completed=completed_count,
                ledger=ledger, log=research_log,
            )
        else:
            finalize_wave(
                campaign_id=campaign_id, wave=wave, completed=completed_count,
                manifest=args.manifest, report_root=report_root, log=research_log,
                split_name=split_name,
            )

    last_complete = completed - remainder
    if last_complete:
        # Recover idempotently when a prior exact-once child invocation finished but
        # reporting was interrupted before the controller started.
        finalize(last_complete // wave_size, last_complete)
    if remainder:
        expected_addition = wave_size - remainder
        subprocess.run([
            sys.executable, str(ROOT / "scripts/run_balanced_pilot.py"),
            "--manifest", str(args.manifest),
        ], check=True)
        new_completed = len(jsonl(ledger))
        if new_completed != completed + expected_addition:
            raise SystemExit(
                f"partial-wave recovery added {new_completed - completed} rows, "
                f"expected {expected_addition}"
            )
        completed = new_completed
        finalize(completed // wave_size, completed)
    waves_run = 0
    while completed < expected and (
        args.max_waves is None or waves_run < args.max_waves
    ):
        subprocess.run([
            sys.executable, str(ROOT / "scripts/run_balanced_pilot.py"),
            "--manifest", str(args.manifest),
        ], check=True)
        new_completed = len(jsonl(ledger))
        if new_completed != completed + wave_size:
            raise SystemExit(
                f"child completed {new_completed - completed} rows, expected {wave_size}"
            )
        completed = new_completed
        wave = completed // wave_size
        finalize(wave, completed)
        waves_run += 1
        print(json.dumps({
            "completed_wave": wave,
            "completed_episodes": completed,
            "remaining_episodes": expected - completed,
            "protected_test_used": protected,
        }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
