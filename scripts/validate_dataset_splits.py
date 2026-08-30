#!/usr/bin/env python3
"""Validate episode independence and frozen map-route assignments before extraction."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def assignments(document: dict) -> tuple[dict[str, str], dict[str, str]]:
    maps, routes = {}, {}
    for split in ("development", "validation", "held_out_map_test"):
        for map_id in document[split]["maps"]:
            if map_id in maps:
                raise ValueError(f"map assigned to multiple splits: {map_id}")
            maps[map_id] = split
        for route_id in document[split]["routes"]:
            if route_id in routes:
                raise ValueError(f"route assigned to multiple splits: {route_id}")
            routes[route_id] = split
    return maps, routes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_root", type=Path)
    parser.add_argument(
        "--splits", type=Path, default=ROOT / "data/manifests/splits.template.yaml"
    )
    parser.add_argument("--allow-protected", action="store_true")
    args = parser.parse_args()
    split_doc = yaml.safe_load(args.splits.read_text(encoding="utf-8"))
    try:
        map_split, route_split = assignments(split_doc)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    findings, run_ids, episode_keys, bag_paths = [], set(), set(), set()
    count = 0
    for path in sorted(args.summary_root.glob("*.yaml")):
        summary = yaml.safe_load(path.read_text(encoding="utf-8"))
        identity, environment = summary["identity"], summary["environment"]
        run_id = identity["run_id"]
        if run_id in run_ids:
            findings.append(f"duplicate run_id: {run_id}")
        run_ids.add(run_id)
        episode_key = identity.get("episode_key")
        if episode_key:
            campaign_key = (identity.get("campaign_id"), episode_key)
            if campaign_key in episode_keys:
                findings.append(f"duplicate campaign episode key: {campaign_key}")
            episode_keys.add(campaign_key)
        bag_path = summary.get("provenance", {}).get("bag_path")
        if bag_path in bag_paths:
            findings.append(f"bag reused across episodes: {bag_path}")
        bag_paths.add(bag_path)
        map_id, route_id = environment["map_id"], environment["route_id"]
        expected_map = map_split.get(map_id)
        expected_route = route_split.get(route_id)
        declared = environment.get("split")
        if expected_map is None or expected_route is None:
            findings.append(f"unassigned map or route: {map_id}/{route_id}")
        elif expected_map != expected_route or declared != expected_map:
            findings.append(
                f"split mismatch {run_id}: map={expected_map}, route={expected_route}, declared={declared}"
            )
        if expected_map == "held_out_map_test" and not args.allow_protected:
            findings.append(f"protected episode present before authorization: {run_id}")
        count += 1
    if findings:
        print(f"SPLIT INVALID: {len(findings)} finding(s)")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print(f"SPLIT VALID: {count} independent episode summaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
