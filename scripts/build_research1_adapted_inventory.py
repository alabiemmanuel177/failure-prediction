#!/usr/bin/env python3
"""Publish the hash-addressed inventory of adapted Research 1 development episodes.

Reads the admitted adapter manifest and the adapted summaries produced by
``scripts/adapt_research1_episode.py`` and writes
``data/manifests/research1_development_v1.episodes.jsonl`` in the same row schema as
``data/manifests/balanced_pilot_v1.episodes.jsonl`` so that
``scripts/extract_dataset_sequences.py --summary-root <adapted>/summaries`` can consume
it unchanged. Rows are emitted only for run_ids whose four adapted artifacts exist and
whose summary still agrees with the admitted manifest. Never overwrites.

    python3 scripts/build_research1_adapted_inventory.py [--adapted-root DIR] [--output PATH]
        [--require-complete]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import jsonl_bytes, publish_new_bytes, sha256_file  # noqa: E402
from src.research1_adapter import CAMPAIGN_ID, RECORDING_PROFILE, episode_key  # noqa: E402


MANIFEST = ROOT / "data/manifests/research1_development_adapter_v1.jsonl"
DEFAULT_ADAPTED_ROOT = ROOT / "data/adapted/research1_development_v1"
DEFAULT_OUTPUT = ROOT / "data/manifests/research1_development_v1.episodes.jsonl"


def inventory_row(row: dict, adapted_root: Path) -> dict | None:
    run_id = str(row["run_id"])
    summary_path = adapted_root / "summaries" / f"{run_id}.yaml"
    events_path = adapted_root / "events" / f"{run_id}.json"
    health_path = adapted_root / "topic_health" / f"{run_id}.json"
    clock_path = adapted_root / "clock_maps" / f"{run_id}.json"
    if not all(path.is_file() for path in (summary_path, events_path, health_path, clock_path)):
        return None
    summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
    identity, environment = summary["identity"], summary["environment"]
    label, outcome, provenance = summary["label_only"], summary["outcome"], summary["provenance"]
    if identity["run_id"] != run_id or identity["campaign_id"] != CAMPAIGN_ID:
        raise ValueError(f"{run_id}: adapted summary identity differs")
    if environment.get("protected_test_used") is not False or environment.get("split") != "development":
        raise ValueError(f"{run_id}: adapted summary is not a development episode")
    if provenance.get("event_sidecar") != str(events_path) or provenance.get("receive_clock_map") != str(clock_path):
        raise ValueError(f"{run_id}: adapted summary references unexpected sidecars")
    if sha256_file(clock_path) != provenance.get("receive_clock_map_sha256"):
        raise ValueError(f"{run_id}: clock map checksum differs from summary")
    for field in ("fault_family", "severity", "primary_event_class", "terminal_state"):
        source = label if field in label else outcome
        if source.get(field) != row.get(field):
            raise ValueError(f"{run_id}: {field} differs between manifest and adapted summary")
    bag_path = Path(provenance["bag_path"])
    bag_bytes = sum(item.stat().st_size for item in bag_path.rglob("*") if item.is_file())
    return {
        "dataset_episode_key": episode_key(run_id),
        "run_id": run_id,
        "observed_campaign_id": identity["campaign_id"],
        "observed_episode_key": identity["episode_key"],
        "infrastructure_replacement": False,
        "replacement_for_run_id": None,
        "map_id": environment["map_id"],
        "route_id": environment["route_id"],
        "system_id": environment["system_id"],
        "seed": environment["seed"],
        "split": environment["split"],
        "protected_test_used": False,
        "fault_family": label["fault_family"],
        "severity": label.get("severity"),
        "primary_event_class": label.get("primary_event_class"),
        "terminal_state": outcome["terminal_state"],
        "success": outcome["success"],
        "collision": outcome["collision"],
        "timeout": outcome["timeout"],
        "duration_s": outcome["duration_s"],
        "recording_profile": provenance.get("recording_profile", RECORDING_PROFILE),
        "bag_bytes": bag_bytes,
        "bag_checksum_sha256": provenance["bag_checksum_sha256"],
        "summary_sha256": sha256_file(summary_path),
        "topic_health_sha256": sha256_file(health_path),
        "event_sidecar_sha256": sha256_file(events_path),
        "research1_platform_commit": identity["research1_platform_commit"],
        "research2_config_hash": identity["research2_config_hash"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--adapted-root", type=Path, default=DEFAULT_ADAPTED_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-complete", action="store_true",
                        help="refuse unless every admitted row has been adapted")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    records = []
    missing = []
    for row in rows:
        record = inventory_row(row, args.adapted_root)
        if record is None:
            missing.append(row["run_id"])
        else:
            records.append(record)
    if args.require_complete and missing:
        raise SystemExit(f"{len(missing)} admitted rows are not adapted yet; refusing")
    if not records:
        raise SystemExit("no adapted episodes found; nothing to publish")
    if len({item["run_id"] for item in records}) != len(records):
        raise SystemExit("duplicate run_id in adapted inventory")
    records.sort(key=lambda item: (item["map_id"], item["route_id"], item["run_id"]))
    payload = jsonl_bytes(records)
    publish_new_bytes(args.output, payload)
    print(json.dumps({
        "inventory": str(args.output), "rows": len(records), "not_yet_adapted": len(missing),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "events": sum(item["primary_event_class"] is not None for item in records),
        "protected_test_used": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
