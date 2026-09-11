#!/usr/bin/env python3
"""Run preregistered replacements for declared preprotected infrastructure invalids."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_live_integrity_campaign import (
    append_ledger, attempted_keys, episode_command, research1_campaign_active,
    validate_completed_artifact,
)


def summary_for_episode(campaign_id: str, episode_key: str, root: Path) -> dict | None:
    matches = []
    for path in (root / "summaries").glob("*.yaml"):
        summary = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            continue
        identity = summary.get("identity", {})
        if identity.get("campaign_id") == campaign_id \
                and identity.get("episode_key") == episode_key:
            matches.append(summary)
    if len(matches) > 1:
        raise ValueError(f"multiple summaries for {campaign_id}/{episode_key}")
    return matches[0] if matches else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/manifests/balanced_pilot_replacements_v1.yaml",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    document = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    allowed_splits = document.get("allowed_splits")
    protected = document.get("protected_test_used")
    if protected is True and allowed_splits == ["held_out_map_test"]:
        # Post-freeze replacements for a confirmatory campaign: the same gate that
        # admits the parent campaign must pass (freeze, assigned split, readiness).
        from src.protected_data import held_out_campaign_gate
        maps = sorted({str(item["map"]) for item in document.get("episodes", [])})
        findings = held_out_campaign_gate(ROOT, maps)
        if findings:
            raise SystemExit("protected replacement blocked:\n- " + "\n- ".join(findings))
    elif protected is not False or allowed_splits not in (["development"], ["validation"]):
        raise SystemExit("replacement manifest must contain one preprotected split")
    campaign_id = str(document["campaign_id"])
    parent_campaign_id = str(document.get("parent_campaign_id", "balanced_pilot_v1"))
    ledger = ROOT / "logs/campaigns" / f"{campaign_id}.jsonl"
    attempted = attempted_keys(ledger)
    pending = [episode for episode in document["episodes"]
               if episode["episode_key"] not in attempted]
    print(f"campaign={campaign_id} total={len(document['episodes'])} "
          f"attempted={len(attempted)} pending={len(pending)}")
    if args.dry_run:
        return 0
    lock_path = ROOT / "logs/campaigns" / f"{parent_campaign_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another pilot runner holds {lock_path}")
    gate = subprocess.run([
        sys.executable, str(ROOT / "scripts/check_readiness.py"), "--stage", "extraction",
    ], check=False)
    if gate.returncode:
        raise SystemExit("extraction readiness gate failed")
    if research1_campaign_active():
        raise SystemExit("Research 1 confirmatory service is active")
    if shutil.disk_usage(args.output_root.resolve()).free < 100 * 2**30:
        raise SystemExit("100 GiB pilot free-space reserve reached")
    for episode in pending:
        policy = str(document.get("replacement_policy", {}).get("original_invalid_kind"))
        original = summary_for_episode(
            parent_campaign_id, episode["replaces_episode_key"], args.output_root
        )
        if policy == "pre_goal_refusal_before_launch":
            # The episode runner refused the attempt before creating a run id (platform
            # boundary refusal, 8 September 2026): the ledger holds a non-zero episode
            # return code, no artifact validation, and nothing was materialised.
            parent_ledger = ROOT / "logs/campaigns" / f"{parent_campaign_id}.jsonl"
            rows = [json.loads(line) for line in parent_ledger.read_text(encoding="utf-8").splitlines()]
            matches = [row for row in rows if row.get("episode_key") == episode["replaces_episode_key"]]
            eligible = (
                original is None
                and episode.get("replaces_run_id") in (None, "", "none")
                and len(matches) == 1
                and matches[0].get("episode_returncode") not in (None, 0)
                and matches[0].get("artifact_validation_returncode") is None
            )
        elif policy == "pre_goal_unmaterialized_attempt":
            parent_ledger = ROOT / "logs/campaigns" / f"{parent_campaign_id}.jsonl"
            rows = [json.loads(line) for line in parent_ledger.read_text(encoding="utf-8").splitlines()]
            matches = [row for row in rows if row.get("episode_key") == episode["replaces_episode_key"]]
            run_id = episode["replaces_run_id"]
            # Two retained shapes of a pre-goal startup failure: the sequential runner
            # left no summary at all; the parallel runner retains an invalid summary
            # with missing_mandatory_topic_before_goal and zero bag files.
            retained_invalid = (
                original is not None
                and original.get("identity", {}).get("run_id") == run_id
                and original.get("outcome", {}).get("terminal_state") == "invalid"
                and original.get("outcome", {}).get("invalid_reason") == "missing_mandatory_topic_before_goal"
                and original.get("provenance", {}).get("bag_mcap_count") == 0
            )
            eligible = (
                (original is None or retained_invalid)
                and len(matches) == 1
                and matches[0].get("episode_returncode") != 0
                and matches[0].get("artifact_validation_returncode") is None
                and (args.output_root / "logs" / run_id).is_dir()
                and not any((args.output_root / "bags" / run_id).glob("*.mcap"))
            )
        elif policy == "post_validation_payload_loss":
            parent_ledger = ROOT / "logs/campaigns" / f"{parent_campaign_id}.jsonl"
            rows = [json.loads(line) for line in parent_ledger.read_text(encoding="utf-8").splitlines()]
            matches = [row for row in rows if row.get("episode_key") == episode["replaces_episode_key"]]
            run_id = episode["replaces_run_id"]
            summary_path = args.output_root / "summaries" / f"{run_id}.yaml"
            bag_dir = args.output_root / "bags" / run_id
            mcap_files = list(bag_dir.glob("*.mcap"))
            metadata_path = bag_dir / "metadata.yaml"
            eligible = (
                original is None
                and len(matches) == 1
                and matches[0].get("episode_returncode") == 0
                and matches[0].get("artifact_validation_returncode") == 0
                and summary_path.is_file() and summary_path.stat().st_size == 0
                and len(mcap_files) == 1 and mcap_files[0].stat().st_size == 0
                and metadata_path.is_file() and metadata_path.stat().st_size == 0
                and (args.output_root / "logs" / run_id).is_dir()
            )
        elif policy == "teardown_before_summary_publication":
            parent_ledger = ROOT / "logs/campaigns" / f"{parent_campaign_id}.jsonl"
            rows = [json.loads(line) for line in parent_ledger.read_text(encoding="utf-8").splitlines()]
            matches = [row for row in rows if row.get("episode_key") == episode["replaces_episode_key"]]
            run_id = episode["replaces_run_id"]
            bag_dir = args.output_root / "bags" / run_id
            mcap_files = list(bag_dir.glob("*.mcap"))
            metadata_path = bag_dir / "metadata.yaml"
            eligible = (
                original is None
                and len(matches) == 1
                and matches[0].get("episode_returncode") != 0
                and matches[0].get("artifact_validation_returncode") is None
                and (args.output_root / "logs" / run_id).is_dir()
                and len(mcap_files) == 1 and mcap_files[0].stat().st_size > 0
                and metadata_path.is_file() and metadata_path.stat().st_size > 0
            )
        elif original is None or original["identity"]["run_id"] != episode["replaces_run_id"]:
            raise SystemExit("replacement does not resolve the declared original summary")
        elif policy == "pre_goal_zero_bag":
            eligible = (
                original["outcome"].get("terminal_state") == "invalid"
                and original["outcome"].get("invalid_reason")
                == "missing_mandatory_topic_before_goal"
                and original["provenance"].get("bag_mcap_count") == 0
            )
        elif policy == "treatment_delivery_artifact_invalid":
            parent_ledger = ROOT / "logs/campaigns" / f"{parent_campaign_id}.jsonl"
            rows = [json.loads(line) for line in parent_ledger.read_text(encoding="utf-8").splitlines()]
            matches = [row for row in rows if row.get("episode_key") == episode["replaces_episode_key"]]
            eligible = (
                len(matches) == 1
                and matches[0].get("episode_returncode") == 0
                and matches[0].get("artifact_validation_returncode") != 0
                and original["provenance"].get("bag_mcap_count") == 1
                and original.get("label_only", {}).get("fault_family") == episode.get("family")
                and original.get("environment", {}).get("split") == allowed_splits[0]
            )
        elif policy == "post_goal_artifact_validation_failure":
            # The episode ran to a terminal state but the artifact validator rejected the
            # recording (e.g. a required topic with no recorded messages): the ledger holds
            # episode_returncode 0 with a non-zero artifact validation code.
            parent_ledger = ROOT / "logs/campaigns" / f"{parent_campaign_id}.jsonl"
            rows = [json.loads(line) for line in parent_ledger.read_text(encoding="utf-8").splitlines()]
            matches = [row for row in rows if row.get("episode_key") == episode["replaces_episode_key"]]
            eligible = (
                len(matches) == 1
                and matches[0].get("episode_returncode") == 0
                and matches[0].get("artifact_validation_returncode") not in (None, 0)
                and original["provenance"].get("bag_mcap_count") == 1
                and original.get("environment", {}).get("split") == allowed_splits[0]
            )
        else:
            raise SystemExit(f"unknown replacement eligibility policy: {policy}")
        if not eligible:
            raise SystemExit("original attempt is not eligible for the declared replacement policy")
        command = episode_command(campaign_id, episode, args.output_root.resolve())
        command.extend([
            "--replacement-for-episode-key", episode["replaces_episode_key"],
            "--replacement-for-run-id", episode["replaces_run_id"],
        ])
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result = subprocess.run(command, check=False)
        # Post-freeze (protected) replacements are validated with the same explicit
        # admission the parallel dispatcher uses; the validator otherwise flags the
        # held-out split itself as a finding.
        validator_args = ("--allow-protected-after-freeze",) if protected is True else ()
        artifact_rc = validate_completed_artifact(
            campaign_id, episode["episode_key"], args.output_root.resolve(), validator_args
        ) if result.returncode == 0 else None
        effective = result.returncode or artifact_rc or 0
        append_ledger(ledger, {
            "campaign_id": campaign_id,
            "episode_key": episode["episode_key"],
            "replaces_episode_key": episode["replaces_episode_key"],
            "replaces_run_id": episode["replaces_run_id"],
            "started_utc": started,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "returncode": effective,
            "episode_returncode": result.returncode,
            "artifact_validation_returncode": artifact_rc,
        })
        if effective:
            raise SystemExit("replacement attempt failed; retained and no automatic retry allowed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
