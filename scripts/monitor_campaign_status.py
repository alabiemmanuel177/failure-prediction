#!/usr/bin/env python3
"""Write an atomic, read-only campaign-health snapshot for periodic monitoring."""

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments import campaign_episodes  # noqa: E402


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def lock_is_held(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return False


def append_history(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def snapshot(manifest: dict, *, root: Path = ROOT) -> dict:
    campaign_id = str(manifest["campaign_id"])
    expected = int(manifest["expected_episode_count"])
    ledger = jsonl(root / "logs/campaigns" / f"{campaign_id}.jsonl")
    ledger_keys = [row["episode_key"] for row in ledger]
    attempted = set(ledger_keys)
    duplicate_parent_keys = sorted(
        key for key, count in Counter(ledger_keys).items() if count > 1
    )
    expected_keys: set[str] | None = None
    if isinstance(manifest.get("design"), dict):
        expected_keys = {
            str(item["episode_key"]) for item in campaign_episodes(manifest)
        }
        if len(expected_keys) != expected:
            raise ValueError(
                "expanded campaign keys do not match expected_episode_count"
            )
    unexpected_parent_keys = sorted(
        attempted - expected_keys if expected_keys is not None else set()
    )
    successful = {row["episode_key"] for row in ledger if row.get("returncode") == 0}
    failed = {row["episode_key"] for row in ledger if row.get("returncode") != 0}

    resolved_replacements = []
    declared_invalid = {
        item["original_episode_key"]
        for item in manifest.get("infrastructure_replacements", [])
        if item["original_episode_key"] in attempted
    }
    successful_without_declared_invalid = successful - declared_invalid
    unresolved_invalid = set(failed) | declared_invalid
    for replacement in manifest.get("infrastructure_replacements", []):
        original = replacement["original_episode_key"]
        replacement_rows = jsonl(
            root / "logs/campaigns"
            / f"{replacement['replacement_campaign_id']}.jsonl"
        )
        matches = [
            row for row in replacement_rows
            if row.get("episode_key") == replacement["replacement_episode_key"]
            and row.get("returncode") == 0
        ]
        if original in unresolved_invalid and len(matches) == 1:
            resolved_replacements.append(original)
            unresolved_invalid.discard(original)

    scientifically_resolved = len(successful_without_declared_invalid) + len(resolved_replacements)
    lock_path = root / "logs/campaigns" / f"{campaign_id}.lock"
    held = lock_is_held(lock_path)
    integrity_failed = bool(duplicate_parent_keys or unexpected_parent_keys)
    if integrity_failed:
        state = "fail_stopped_ledger_integrity"
    elif scientifically_resolved == expected and not unresolved_invalid:
        state = "complete"
    elif held:
        state = "running"
    elif unresolved_invalid:
        state = "fail_stopped_unresolved_invalid"
    else:
        state = "idle_incomplete"

    last = ledger[-1] if ledger else None
    return {
        "schema_version": 1,
        "observed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "campaign_id": campaign_id,
        "state": state,
        "runner_lock_held": held,
        "counts": {
            "expected_design_keys": expected,
            "attempted_design_keys": len(attempted),
            "successful_parent_artifacts": len(successful_without_declared_invalid),
            "failed_parent_attempts": len(failed),
            "valid_declared_replacements": len(resolved_replacements),
            "scientifically_resolved_design_keys": scientifically_resolved,
            "remaining_design_keys": (
                len(expected_keys - attempted)
                if expected_keys is not None
                else max(0, expected - len(attempted))
            ),
            "unresolved_invalid_attempts": len(unresolved_invalid),
        },
        "integrity": {
            "duplicate_parent_episode_keys": duplicate_parent_keys,
            "unexpected_parent_episode_keys": unexpected_parent_keys,
        },
        "resolved_replacement_keys": sorted(resolved_replacements),
        "unresolved_invalid_keys": sorted(unresolved_invalid),
        "last_ledger_record": last,
        "free_space_gib": round(shutil.disk_usage(root).free / 2**30, 3),
        "protected_test_used": False,
        "monitor_policy": "observe_only_no_restart_no_retry",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/status/campaign_monitor.yaml",
    )
    parser.add_argument("--history", type=Path)
    args = parser.parse_args()
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    report = snapshot(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    temporary.replace(args.output)
    history = args.history or ROOT / "logs/monitoring" / f"{manifest['campaign_id']}.jsonl"
    append_history(history, report)
    print(yaml.safe_dump(report, sort_keys=False), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
