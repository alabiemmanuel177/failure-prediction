#!/usr/bin/env python3
"""Collect a preregistered development or validation campaign after extraction readiness.

Human threshold and annotation review remain mandatory for feature extraction used in
model fitting or selection, but do not prohibit immutable raw pre-protected collection.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_live_integrity_campaign import (
    append_ledger,
    attempted_keys,
    episode_command,
    research1_campaign_active,
    validate_completed_artifact,
)
from src.experiments import (
    balanced_execution_order, expand_balanced_pilot, targeted_execution_order,
    validate_balanced_pilot, validate_confirmatory_campaign, validate_development_supplement,
    validate_targeted_campaign,
)
from src.protected_data import held_out_campaign_gate
# Closed-loop recovery campaigns (work package Q): only manifests that declare
# ``recovery_policies`` expand per policy and pass --recovery-policy to the runner.
from src.experiments.recovery_campaigns import (
    PAIRED_RECOVERY_KIND, RECOVERY_PILOT_KIND, declares_recovery_policies,
    expand_recovery_policies, validate_recovery_pilot,
)
from src.recovery.plumbing import recovery_cli_arguments


def targeted_prerequisite_findings(root: Path) -> list[str]:
    """Require complete, preprotected validation evidence before targeted collection."""
    findings: list[str] = []
    report_path = root / "reports/validation/balanced_validation_v1.cumulative324.yaml"
    dataset_path = root / "data/manifests/balanced_validation_v1.dataset.yaml"
    inventory_path = root / "data/manifests/balanced_validation_v1.episodes.jsonl"
    if not report_path.is_file():
        findings.append("validation cumulative-324 report is missing")
    else:
        report = yaml.safe_load(report_path.read_text(encoding="utf-8")) or {}
        counts = report.get("counts", {})
        if not (
            report.get("complete_and_artifact_valid") is True
            and report.get("protected_test_used") is False
            and counts.get("expected") == counts.get("observed")
            == counts.get("usable") == 324
        ):
            findings.append("validation cumulative-324 report is not complete and valid")
    if not inventory_path.is_file():
        findings.append("validation episode inventory is missing")
    if not dataset_path.is_file():
        findings.append("validation dataset manifest is missing")
    else:
        dataset = yaml.safe_load(dataset_path.read_text(encoding="utf-8")) or {}
        if not (
            dataset.get("dataset_id") == "balanced_validation_v1-validation-324"
            and dataset.get("protected_test_used") is False
            and dataset.get("episode_inventory", {}).get("rows") == 324
        ):
            findings.append("validation dataset manifest is not the frozen 324-episode inventory")
    return findings


def supplement_prerequisite_findings(root: Path) -> list[str]:
    """Require the finalised targeted-development inventory before the supplement."""
    findings = targeted_prerequisite_findings(root)
    dataset_path = root / "data/manifests/targeted_development_v1.dataset.yaml"
    if not dataset_path.is_file():
        findings.append("targeted development dataset manifest is missing; finalise it first")
    else:
        dataset = yaml.safe_load(dataset_path.read_text(encoding="utf-8")) or {}
        if not (
            dataset.get("dataset_id") == "targeted_development_v1-development-1212"
            and dataset.get("protected_test_used") is False
            and dataset.get("episode_inventory", {}).get("rows") == 1212
        ):
            findings.append("targeted development dataset manifest is not the frozen 1,212-episode inventory")
    return findings


def recovery_prerequisite_findings(root: Path) -> list[str]:
    """A recovery campaign needs the frozen predictor, its threshold and live evidence."""
    findings: list[str] = []
    freeze_path = root / "configs/model_freeze.yaml"
    freeze = yaml.safe_load(freeze_path.read_text(encoding="utf-8")) if freeze_path.is_file() else None
    if not (isinstance(freeze, dict) and freeze.get("frozen") is True):
        findings.append("configs/model_freeze.yaml is absent or not frozen")
    alarm_path = root / "configs/alarm_policy.yaml"
    alarm = yaml.safe_load(alarm_path.read_text(encoding="utf-8")) if alarm_path.is_file() else {}
    if not isinstance(alarm, dict) or alarm.get("threshold") is None:
        findings.append("configs/alarm_policy.yaml has no frozen threshold")
    evidence_path = root / "configs/recovery_live_evidence.yaml"
    evidence = yaml.safe_load(evidence_path.read_text(encoding="utf-8")) if evidence_path.is_file() else None
    if not (isinstance(evidence, dict) and evidence.get("frozen") is True):
        findings.append("configs/recovery_live_evidence.yaml is absent or not frozen")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/manifests/balanced_pilot_v1.yaml",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args()
    document = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    splits = yaml.safe_load(
        (ROOT / "data/manifests/splits.template.yaml").read_text(encoding="utf-8")
    )
    supplement = document.get("campaign_kind") == "development_supplement"
    targeted = document.get("campaign_kind") == "targeted_event_floor" or supplement
    confirmatory = document.get("campaign_kind") == "held_out_confirmatory"
    recovery_pilot = document.get("campaign_kind") == RECOVERY_PILOT_KIND
    paired_recovery = document.get("campaign_kind") == PAIRED_RECOVERY_KIND
    recovery = declares_recovery_policies(document)
    if recovery_pilot:
        findings = validate_recovery_pilot(document, splits)
    elif paired_recovery:
        from scripts.build_recovery_campaign_manifest import validate_paired_recovery
        findings = validate_paired_recovery(document, splits)
        confirmatory = True
    elif confirmatory:
        findings = validate_confirmatory_campaign(document, splits)
    elif supplement:
        findings = validate_development_supplement(document, splits)
    elif targeted:
        findings = validate_targeted_campaign(document, splits)
    else:
        findings = validate_balanced_pilot(document, splits)
    if findings:
        raise SystemExit("invalid balanced pilot:\n- " + "\n- ".join(findings))
    expanded = expand_balanced_pilot(document)
    episodes = (
        targeted_execution_order(expanded) if (targeted or confirmatory or recovery)
        else balanced_execution_order(expanded)
    )
    if recovery:
        # Every policy of one pairing cell runs consecutively; policies share the seed.
        episodes = expand_recovery_policies(document, episodes)
    expected = int(document["expected_episode_count"])
    if len(episodes) != expected:
        raise SystemExit(f"expanded {len(episodes)} episodes, expected {expected}")
    if len({item["episode_key"] for item in episodes}) != len(episodes):
        raise SystemExit("pilot episode keys are not unique")
    if not recovery and len({item["seed"] for item in episodes}) != len(episodes):
        raise SystemExit("pilot seeds are not unique")
    if recovery and len({(item["seed"], item["recovery_policy_id"]) for item in episodes}) != len(episodes):
        raise SystemExit("recovery seed/policy pairs are not unique")
    campaign_id = document["campaign_id"]
    ledger = ROOT / "logs/campaigns" / f"{campaign_id}.jsonl"
    attempted = attempted_keys(ledger)
    pending = [episode for episode in episodes if episode["episode_key"] not in attempted]
    wave_size = int(document["execution_policy"]["maximum_episodes_per_invocation"])
    wave_remainder = len(attempted) % wave_size
    boundary_remaining = wave_size - wave_remainder if wave_remainder else wave_size
    requested_limit = args.max_episodes or wave_size
    limit = min(requested_limit, boundary_remaining)
    if limit < 1:
        raise SystemExit("max episodes must be positive")
    runnable = pending[:limit]
    counts = {}
    for episode in episodes:
        counts[episode["family"]] = counts.get(episode["family"], 0) + 1
    print(json.dumps({
        "campaign_id": campaign_id,
        "total": len(episodes),
        "attempted": len(attempted),
        "pending": len(pending),
        "scheduled_this_invocation": len(runnable),
        "family_counts": counts,
        "protected_test_used": bool(document.get("protected_test_used")),
    }, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    validator_args: tuple[str, ...] = ()
    if confirmatory:
        gate_findings = held_out_campaign_gate(ROOT, list(document["design"]["map_routes"]))
        if gate_findings:
            raise SystemExit(
                "confirmatory campaign is blocked until the model freeze and protected "
                "split assignment pass the confirmatory readiness gate:\n- "
                + "\n- ".join(gate_findings)
            )
        validator_args = ("--allow-protected-after-freeze",)
    if recovery:
        recovery_findings = recovery_prerequisite_findings(ROOT)
        if recovery_findings:
            raise SystemExit(
                "recovery campaign is blocked until the predictor and live evidence are frozen:\n- "
                + "\n- ".join(recovery_findings)
            )
    if targeted:
        prerequisite_findings = (
            supplement_prerequisite_findings(ROOT) if supplement
            else targeted_prerequisite_findings(ROOT)
        )
        if prerequisite_findings:
            raise SystemExit(
                "targeted development is blocked until validation is finalized:\n- "
                + "\n- ".join(prerequisite_findings)
            )
        inventory_gate = subprocess.run([
            sys.executable,
            str(ROOT / "scripts/validate_validation_dataset_inventory.py"),
        ], check=False)
        if inventory_gate.returncode:
            raise SystemExit(
                "targeted development is blocked: validation inventory validation failed"
            )
    pause_path = ROOT / "logs/campaigns" / f"{campaign_id}.pause"
    if pause_path.exists():
        raise SystemExit(
            f"campaign pause marker present: {pause_path}; inspect its reason before resuming"
        )
    lock_path = ROOT / "logs/campaigns" / f"{campaign_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another {campaign_id} runner holds {lock_path}")
    gate = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_readiness.py"), "--stage",
         "confirmatory" if confirmatory else "extraction"],
        check=False,
    )
    if gate.returncode:
        raise SystemExit("balanced pilot is blocked until the readiness gate passes")
    if os.environ.get("ROS_DISTRO") != "jazzy" or os.environ.get("ROS_DOMAIN_ID") != "52":
        raise SystemExit("source scripts/env_research2.sh; pilot requires ROS 2 Jazzy on domain 52")
    if research1_campaign_active():
        raise SystemExit("Research 1 confirmatory service is active")
    minimum_free = float(
        document["execution_policy"]["minimum_free_space_gib_before_episode"]
    ) * 2**30
    for episode in runnable:
        free = shutil.disk_usage(args.output_root.resolve()).free
        if free < minimum_free:
            raise SystemExit(
                f"free-space reserve reached before {episode['episode_key']}: "
                f"{free / 2**30:.1f} GiB available, {minimum_free / 2**30:.1f} GiB required"
            )
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result = subprocess.run(
            episode_command(campaign_id, episode, args.output_root.resolve())
            + recovery_cli_arguments(episode),
            check=False,
        )
        artifact_rc = (
            validate_completed_artifact(
                campaign_id, episode["episode_key"], args.output_root.resolve(),
                validator_args,
            ) if result.returncode == 0 else None
        )
        effective = result.returncode or artifact_rc or 0
        append_ledger(ledger, {
            "campaign_id": campaign_id,
            "episode_key": episode["episode_key"],
            "started_utc": started,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "returncode": effective,
            "episode_returncode": result.returncode,
            "artifact_validation_returncode": artifact_rc,
        })
        if effective:
            raise SystemExit(
                f"pilot stopped on invalid episode {episode['episode_key']}; retained for audit"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
