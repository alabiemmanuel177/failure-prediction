#!/usr/bin/env python3
"""Append hash-chained JSON records to the research decision log."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "logs" / "research-log.jsonl"
KINDS = ("decision", "protocol_change", "exclusion", "data_issue", "experiment", "review")


def canonical(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
    return records


def verify(records: list[dict]) -> None:
    previous = "GENESIS"
    for index, record in enumerate(records, 1):
        stored_hash = record.get("record_hash")
        unsigned = {key: value for key, value in record.items() if key != "record_hash"}
        if unsigned.get("previous_hash") != previous:
            raise ValueError(f"record {index}: previous_hash does not match chain")
        computed = hashlib.sha256(canonical(unsigned)).hexdigest()
        if stored_hash != computed:
            raise ValueError(f"record {index}: record_hash is invalid")
        previous = stored_hash


def append_record(path: Path, *, kind: str, actor: str, message: str, metadata: dict) -> dict:
    records = read_records(path)
    verify(records)
    previous = records[-1]["record_hash"] if records else "GENESIS"
    unsigned = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "actor": actor,
        "message": message,
        "metadata": metadata,
        "previous_hash": previous,
    }
    record = {**unsigned, "record_hash": hashlib.sha256(canonical(unsigned)).hexdigest()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.set_defaults(command="verify")

    add_parser = subparsers.add_parser("add")
    add_parser.add_argument("--kind", choices=KINDS, required=True)
    add_parser.add_argument("--actor", required=True)
    add_parser.add_argument("--message", required=True)
    add_parser.add_argument("--metadata", default="{}", help="JSON object")

    args = parser.parse_args()
    records = read_records(args.log)
    verify(records)
    if args.command == "verify":
        print(f"verified {len(records)} chained record(s) in {args.log}")
        return 0

    metadata = json.loads(args.metadata)
    if not isinstance(metadata, dict):
        raise SystemExit("--metadata must decode to a JSON object")
    record = append_record(
        args.log,
        kind=args.kind,
        actor=args.actor,
        message=args.message,
        metadata=metadata,
    )
    print(record["record_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

