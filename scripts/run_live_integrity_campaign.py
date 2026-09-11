#!/usr/bin/env python3
"""Run the development-only, exact-once Research 2 integrity campaign."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ROS_PACKAGE = ROOT / "ros_ws" / "src" / "failure_experiment"
sys.path.insert(0, str(ROS_PACKAGE))

from failure_experiment.parameters import load_fault_parameters
from src.platform_boundary import validate_platform
from src.recovery.plumbing import recovery_cli_arguments


SEVERITY_INDEX = {"low": 1, "medium": 2, "high": 3}


def research1_campaign_active() -> bool:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", "rcn-confirmatory.service"],
        check=False, capture_output=True, text=True,
    )
    return result.stdout.strip() in {"active", "activating", "reloading"}


def expand(document: dict) -> list[dict]:
    episodes = []
    for control in document["controls"]:
        episodes.append({
            **control, "family": "none", "severity": "none",
            "clean_prefix_seconds": 10.0, "planned_onset_seconds": 15.0,
            "maximum_duration_seconds": 20.0, "maximum_wait_seconds": 10.0,
        })
    matrix = document["fault_matrix"]
    episode_suffix = str(matrix.get("episode_suffix", "001"))
    for family, settings in matrix["families"].items():
        for severity in matrix["severities"]:
            episode = {**matrix["defaults"], **settings}
            episode.update({
                "episode_key": f"{family}-{severity}-{episode_suffix}",
                "family": family,
                "severity": severity,
                "seed": int(settings["seed_base"]) + SEVERITY_INDEX[severity],
            })
            episode.pop("seed_base", None)
            episode.pop("placement_status", None)
            episodes.append(episode)
    return episodes


def episode_command(campaign_id: str, episode: dict, output_root: Path) -> list[str]:
    command = [
        sys.executable, str(ROOT / "scripts" / "run_research2_episode.py"),
        "--campaign-id", campaign_id,
        "--episode-key", episode["episode_key"],
        "--map", episode["map"], "--route", episode["route"],
        "--system", episode["system"], "--family", episode["family"],
        "--severity", episode["severity"], "--seed", str(episode["seed"]),
        "--clean-prefix-seconds", str(episode["clean_prefix_seconds"]),
        "--planned-onset-seconds", str(episode["planned_onset_seconds"]),
        "--maximum-duration-seconds", str(episode["maximum_duration_seconds"]),
        "--maximum-wait-seconds", str(episode["maximum_wait_seconds"]),
        "--output-root", str(output_root),
    ]
    for name in (
        "injection_x", "injection_y", "injection_yaw", "placement_mode", "route_fraction",
        "recording_profile",
    ):
        if name in episode:
            command.extend(["--" + name.replace("_", "-"), str(episode[name])])
    # Empty for every campaign without recovery_policies (byte-identical command line).
    command.extend(recovery_cli_arguments(episode))
    return command


def attempted_keys(ledger: Path) -> set[str]:
    if not ledger.exists():
        return set()
    keys = set()
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if line.strip():
            keys.add(json.loads(line)["episode_key"])
    return keys


def append_ledger(ledger: Path, record: dict) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_completed_artifact(
    campaign_id: str, episode_key: str, output_root: Path, extra_args: tuple[str, ...] = (),
) -> int:
    matches = []
    for path in sorted((output_root / "summaries").glob("*.yaml")):
        summary = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            continue
        identity = summary.get("identity", {})
        if identity.get("campaign_id") == campaign_id and identity.get("episode_key") == episode_key:
            matches.append(path)
    if len(matches) != 1:
        print(
            f"expected one summary for {campaign_id}/{episode_key}, found {len(matches)}",
            file=sys.stderr,
        )
        return 1
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_episode_artifacts.py"),
         str(matches[0]), *extra_args],
        check=False,
    ).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data" / "manifests" / "live_integrity_v1.yaml",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    document = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    campaign_id = document["campaign_id"]
    episodes = expand(document)
    if len({item["episode_key"] for item in episodes}) != len(episodes):
        raise SystemExit("manifest episode_key values are not unique")
    if any(not item["map"].startswith("dev_") for item in episodes):
        raise SystemExit("live integrity campaign must use development maps only")
    lock = yaml.safe_load((ROOT / "integration" / "research1.lock.yaml").read_text())
    validate_platform(lock)
    for episode in episodes:
        load_fault_parameters(ROOT, episode["family"], episode["severity"])

    ledger = ROOT / "logs" / "campaigns" / f"{campaign_id}.jsonl"
    attempted = attempted_keys(ledger)
    pending = [item for item in episodes if item["episode_key"] not in attempted]
    print(f"campaign={campaign_id} total={len(episodes)} attempted={len(attempted)} pending={len(pending)}")
    if args.dry_run:
        for episode in pending:
            print(" ".join(episode_command(campaign_id, episode, args.output_root.resolve())))
        return 0
    if os.environ.get("ROS_DISTRO") != "jazzy":
        raise SystemExit("source scripts/env_research2.sh before campaign execution")
    if research1_campaign_active():
        raise SystemExit("Research 1 protected campaign is active; refusing resource contention")

    for episode in pending:
        if research1_campaign_active():
            raise SystemExit("Research 1 protected campaign became active; stopping before next episode")
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result = subprocess.run(
            episode_command(campaign_id, episode, args.output_root.resolve()), check=False
        )
        artifact_returncode = (
            validate_completed_artifact(campaign_id, episode["episode_key"], args.output_root.resolve())
            if result.returncode == 0 else None
        )
        effective_returncode = result.returncode or artifact_returncode or 0
        append_ledger(ledger, {
            "campaign_id": campaign_id,
            "episode_key": episode["episode_key"],
            "started_utc": started,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "returncode": effective_returncode,
            "episode_returncode": result.returncode,
            "artifact_validation_returncode": artifact_returncode,
        })
        if effective_returncode != 0:
            raise SystemExit(
                f"episode {episode['episode_key']} failed with {effective_returncode}; "
                "preserved and stopping exact-once campaign"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
