#!/usr/bin/env python3
"""Assign the protected held-out maps and routes strictly after the model freeze.

The Research 1 test maps stay uninspected until ``configs/model_freeze.yaml`` is
frozen. Only then does this script list the test map directories, read each map's
frozen route manifest, freeze exactly ``--maps-required`` maps times
``--routes-per-map`` routes into ``data/manifests/splits.template.yaml``
(``held_out_map_test``), publish an immutable assignment record with SHA-256 of every
map and route file, and append a ``protocol_change`` research-log record.

``--dry-run`` never reads the test directory: it only reports the gate state and what
the assignment would do.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research_log import append_record
from src.dataset_inventory import publish_new_bytes
from src.protected_data import (
    HELD_OUT_ASSIGNED_STATUS, HELD_OUT_SPLIT, model_freeze_findings,
)

ACTOR = "Claude Fable 5.1 (AI assistant, directed by the researcher)"
TEST_SPLIT_DIRECTORY = "test"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def gate_findings(
    *, freeze_path: Path, alarm_path: Path, splits: dict, record_path: Path,
    run_readiness_subprocess: bool,
) -> list[str]:
    """Reuse the confirmatory readiness checks minus the split requirement."""
    findings: list[str] = []
    if not freeze_path.exists():
        findings.append(f"model freeze missing: {freeze_path}")
    else:
        try:
            findings.extend(
                model_freeze_findings(yaml.safe_load(freeze_path.read_text(encoding="utf-8")))
            )
        except yaml.YAMLError as error:
            findings.append(f"model freeze unreadable: {error}")
    if not alarm_path.exists():
        findings.append(f"alarm policy missing: {alarm_path}")
    else:
        alarm = yaml.safe_load(alarm_path.read_text(encoding="utf-8")) or {}
        if alarm.get("threshold") is None:
            findings.append("alarm_policy:threshold is not frozen")
    split = splits.get(HELD_OUT_SPLIT) or {}
    if split.get("maps") or split.get("routes"):
        findings.append("held_out_map_test is already assigned; refusing to reassign")
    if record_path.exists():
        findings.append(f"assignment record already exists: {record_path}")
    if run_readiness_subprocess:
        # Training-stage readiness covers every confirmatory prerequisite except the
        # split assignment (which this script creates) and the freeze checked above.
        gate = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "check_readiness.py"),
             "--stage", "training"],
            check=False, capture_output=True, text=True,
        )
        if gate.returncode:
            findings.append("check_readiness --stage training failed:\n" + gate.stdout.strip())
    return findings


def active_routes(spec: dict) -> list[dict]:
    """Return the balanced route set: originals with v1.1 replacements substituted."""
    routes = list(spec.get("routes") or [])
    replacements = {
        item["replaces_route_id"]: item
        for item in (spec.get("replacement_routes") or [])
        if item.get("replaces_route_id")
    }
    result = []
    for route in routes:
        replacement = replacements.get(route["route_id"])
        result.append(replacement if replacement is not None else route)
    return result


def read_test_maps(
    research1_root: Path, *, maps_required: int, routes_per_map: int,
) -> tuple[list[dict], list[str]]:
    """Read protected map directories and route manifests; fail closed on shortfalls."""
    findings: list[str] = []
    test_root = research1_root / "data" / TEST_SPLIT_DIRECTORY
    if not test_root.is_dir():
        return [], [f"Research 1 test directory missing: {test_root}"]
    map_ids = sorted(
        path.name for path in test_root.iterdir()
        if path.is_dir() and path.name.startswith("test_") and (path / "map.yaml").is_file()
    )
    if len(map_ids) != maps_required:
        findings.append(
            f"expected exactly {maps_required} protected maps, found {len(map_ids)}: {map_ids}"
        )
    maps = []
    for map_id in map_ids:
        map_dir = test_root / map_id
        route_path = research1_root / "configs" / "routes" / f"{map_id}.yaml"
        if not route_path.is_file():
            findings.append(f"{map_id}: route manifest missing: {route_path}")
            continue
        spec = yaml.safe_load(route_path.read_text(encoding="utf-8")) or {}
        if spec.get("map_id") != map_id:
            findings.append(f"{map_id}: route manifest declares map_id {spec.get('map_id')!r}")
        routes = active_routes(spec)
        route_ids = [route["route_id"] for route in routes]
        if len(route_ids) != len(set(route_ids)):
            findings.append(f"{map_id}: duplicate route ids in route manifest")
        if len(routes) < routes_per_map:
            findings.append(
                f"{map_id}: route manifest has {len(routes)} active routes, "
                f"{routes_per_map} required; routes are never invented"
            )
        bad_prefix = [rid for rid in route_ids if not rid.startswith(f"{map_id}_")]
        if bad_prefix:
            findings.append(f"{map_id}: route ids without map prefix: {bad_prefix}")
        unverified = [
            route["route_id"] for route in routes[:routes_per_map]
            if route.get("s0_clean_verified") is not True
        ]
        files = {
            "map.yaml": map_dir / "map.yaml",
            "map.pgm": map_dir / "map.pgm",
            "routes.yaml": route_path,
        }
        missing = [name for name, path in files.items() if not path.is_file()]
        if missing:
            findings.append(f"{map_id}: missing files {missing}")
        maps.append({
            "map_id": map_id,
            "map_directory": str(map_dir),
            "route_manifest": str(route_path),
            "route_manifest_frozen": spec.get("frozen"),
            "map_hash_declared": spec.get("map_hash"),
            "episode_timeout_s": spec.get("episode_timeout_s"),
            "routes": route_ids[:routes_per_map],
            "routes_not_selected": route_ids[routes_per_map:],
            "replacement_routes_substituted": sorted(
                item["replaces_route_id"] for item in (spec.get("replacement_routes") or [])
                if item.get("replaces_route_id") in route_ids
            ),
            "routes_without_s0_clean_verification": unverified,
            "file_sha256": {
                name: sha256_file(path) for name, path in files.items() if path.is_file()
            },
        })
    return maps, findings


def render_held_out_block(maps: list[dict], *, assigned_utc: str, freeze_sha256: str) -> str:
    lines = [f"{HELD_OUT_SPLIT}:"]
    lines.append("  maps: [" + ", ".join(item["map_id"] for item in maps) + "]")
    lines.append("  routes:")
    for item in maps:
        for route_id in item["routes"]:
            lines.append(f"    - {route_id}")
    lines.append("  inspect_only_after_policy_freeze: true")
    lines.append(f"  status: {HELD_OUT_ASSIGNED_STATUS}")
    lines.append(f'  assigned_utc: "{assigned_utc}"')
    lines.append(f"  model_freeze_sha256: {freeze_sha256}")
    return "\n".join(lines) + "\n"


def replace_held_out_block(text: str, block: str) -> str:
    """Replace only the top-level held_out_map_test block, byte-for-byte elsewhere."""
    lines = text.splitlines(keepends=True)
    start = end = None
    for index, line in enumerate(lines):
        if start is None:
            if line.startswith(f"{HELD_OUT_SPLIT}:"):
                start = index
            continue
        if line.strip() and not line.startswith((" ", "\t", "#", "-")):
            end = index
            break
    if start is None:
        raise ValueError(f"{HELD_OUT_SPLIT} block not found in split manifest")
    if end is None:
        end = len(lines)
    return "".join(lines[:start]) + block + "".join(lines[end:])


def verify_replacement(original_text: str, updated_text: str, maps: list[dict]) -> None:
    before = yaml.safe_load(original_text)
    after = yaml.safe_load(updated_text)
    for key in before:
        if key == HELD_OUT_SPLIT:
            continue
        if before[key] != after.get(key):
            raise ValueError(f"split manifest key {key} changed during assignment")
    if set(after) != set(before):
        raise ValueError("split manifest top-level keys changed during assignment")
    split = after[HELD_OUT_SPLIT]
    expected_routes = [route for item in maps for route in item["routes"]]
    if split["maps"] != [item["map_id"] for item in maps] or split["routes"] != expected_routes:
        raise ValueError("rendered held-out block does not match the assignment")
    assigned = {split_name: set(after[split_name]["maps"]) for split_name in
                ("development", "validation", HELD_OUT_SPLIT)}
    if assigned[HELD_OUT_SPLIT] & (assigned["development"] | assigned["validation"]):
        raise ValueError("held-out maps overlap a pre-protected split")


def durable_replace(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research1-root", type=Path, default=None,
                        help="defaults to integration/research1.lock.yaml research1_root")
    parser.add_argument("--splits", type=Path, default=ROOT / "data/manifests/splits.template.yaml")
    parser.add_argument("--model-freeze", type=Path, default=ROOT / "configs/model_freeze.yaml")
    parser.add_argument("--alarm", type=Path, default=ROOT / "configs/alarm_policy.yaml")
    parser.add_argument(
        "--record", type=Path,
        default=ROOT / "reports/confirmatory/protected_split_assignment.yaml",
    )
    parser.add_argument("--research-log", type=Path, default=ROOT / "logs/research-log.jsonl")
    parser.add_argument("--maps-required", type=int, default=3)
    parser.add_argument("--routes-per-map", type=int, default=6)
    parser.add_argument("--actor", default=ACTOR)
    parser.add_argument("--dry-run", action="store_true",
                        help="report the gate state without reading the test directory")
    parser.add_argument("--i-confirm-model-freeze", action="store_true",
                        help="explicit human confirmation that the model freeze is final")
    parser.add_argument(
        "--no-readiness-subprocess", action="store_true",
        help="test-only: skip check_readiness --stage training (fake repositories)",
    )
    args = parser.parse_args(argv)

    if args.research1_root is None:
        lock = yaml.safe_load((ROOT / "integration/research1.lock.yaml").read_text(encoding="utf-8"))
        args.research1_root = Path(lock["research1_root"])
    if args.no_readiness_subprocess:
        print("WARNING: repository readiness subprocess skipped (test-only mode)")
    splits_text = args.splits.read_text(encoding="utf-8")
    splits = yaml.safe_load(splits_text) or {}
    findings = gate_findings(
        freeze_path=args.model_freeze, alarm_path=args.alarm, splits=splits,
        record_path=args.record,
        run_readiness_subprocess=not args.no_readiness_subprocess,
    )
    plan = {
        "dry_run": args.dry_run,
        "research1_root": str(args.research1_root),
        "test_directory": str(args.research1_root / "data" / TEST_SPLIT_DIRECTORY),
        "test_directory_read": False,
        "maps_required": args.maps_required,
        "routes_per_map": args.routes_per_map,
        "split_manifest": str(args.splits),
        "assignment_record": str(args.record),
        "gate_findings": findings,
        "would_assign": not findings,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 1 if findings else 0
    if findings:
        raise SystemExit("protected split assignment refused:\n- " + "\n- ".join(findings))
    if not args.i_confirm_model_freeze:
        raise SystemExit(
            "protected split assignment requires --i-confirm-model-freeze after the "
            "model, calibration and alarm policy are frozen"
        )

    freeze_bytes = args.model_freeze.read_bytes()
    freeze_sha256 = sha256_bytes(freeze_bytes)
    maps, map_findings = read_test_maps(
        args.research1_root, maps_required=args.maps_required,
        routes_per_map=args.routes_per_map,
    )
    if map_findings:
        raise SystemExit(
            "protected split assignment failed closed (nothing written):\n- "
            + "\n- ".join(map_findings)
        )
    assigned_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    block = render_held_out_block(maps, assigned_utc=assigned_utc, freeze_sha256=freeze_sha256)
    updated = replace_held_out_block(splits_text, block)
    verify_replacement(splits_text, updated, maps)

    record = {
        "schema_version": 1,
        "kind": "protected_split_assignment",
        "status": HELD_OUT_ASSIGNED_STATUS,
        "assigned_utc": assigned_utc,
        "assigned_by": args.actor,
        "protected_test_used": True,
        "protected_outcomes_consulted": False,
        "research1_root": str(args.research1_root),
        "model_freeze": str(args.model_freeze),
        "model_freeze_sha256": freeze_sha256,
        "alarm_policy_sha256": sha256_file(args.alarm),
        "split_manifest": str(args.splits),
        "split_manifest_sha256_before": sha256_bytes(splits_text.encode("utf-8")),
        "split_manifest_sha256_after": sha256_bytes(updated.encode("utf-8")),
        "maps_required": args.maps_required,
        "routes_per_map": args.routes_per_map,
        "maps": maps,
        "held_out_map_test": {
            "maps": [item["map_id"] for item in maps],
            "routes": [route for item in maps for route in item["routes"]],
        },
        "route_selection_rule": (
            "first routes_per_map active routes in route-manifest order; Protocol 1.1 "
            "replacement routes substitute their originals; no route is invented"
        ),
    }
    payload = yaml.safe_dump(record, sort_keys=False).encode("utf-8")
    publish_new_bytes(args.record, payload)
    durable_replace(args.splits, updated)
    log_record = append_record(
        args.research_log,
        kind="protocol_change",
        actor=args.actor,
        message=(
            f"Assigned the protected held_out_map_test split after the model freeze: "
            f"{len(maps)} maps x {args.routes_per_map} routes frozen from the Research 1 "
            "test route manifests. Protected outcomes remain unconsulted."
        ),
        metadata={
            "maps": record["held_out_map_test"]["maps"],
            "route_count": len(record["held_out_map_test"]["routes"]),
            "model_freeze_sha256": freeze_sha256,
            "assignment_record": str(args.record),
            "assignment_record_sha256": sha256_bytes(payload),
            "protected_test_used": True,
            "protected_outcomes_consulted": False,
        },
    )
    print(json.dumps({
        **plan, "test_directory_read": True, "assigned": record["held_out_map_test"],
        "record": str(args.record), "research_log_record_hash": log_record["record_hash"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
