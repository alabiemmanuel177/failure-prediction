#!/usr/bin/env python3
"""Derive the phased-execution admission record from the two concurrency checks.

Both pre-declared checks (six unrestricted workers; six workers with one S3 episode
at a time) failed on GPU-perception (S3) episodes. Phased execution runs S0 episodes
six-wide and S3 episodes strictly alone. This script admits that plan only if, in
EVERY completed check, the S0 conditions satisfied the unchanged policy bounds:
clean S0 false alerts per mission within the 0.10 budget, and no deployable feature
beyond the standardised-mean-difference bound in any S0 condition. S3-alone is the
sequential reference condition and needs no new evidence. The record is immutable and
names both source reports by hash; it introduces no new measurement.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import publish_new_bytes, sha256_file  # noqa: E402


def s0_findings(report: dict, smd_bound: float, budget: float,
                criterion: str = "absolute", tolerance: float = 0.05) -> tuple[list[str], dict]:
    findings: list[str] = []
    fa = report["false_alerts"]
    by_condition = fa.get("by_condition", {})
    clean_s0 = by_condition.get("clean_s0")
    if clean_s0 is None:
        # v1 predates the per-condition breakdown: recompute from per-mission counts.
        rows = {}
        for line in (ROOT / "data/derived" / report["check_dataset_id"] / "extraction_manifest.jsonl").read_text().splitlines():
            import json
            item = json.loads(line)
            rows[item["run_id"]] = item["dataset_episode_key"].rsplit("-", 2)[-2]
        counts = [n for run, n in fa["check"]["per_mission"].items() if rows.get(run) == "clean_s0"]
        clean_s0 = {"check_false_alerts_per_mission": sum(counts) / len(counts), "clean_missions": len(counts)}
    rate = clean_s0.get("check_false_alerts_per_mission")
    if rate is None and isinstance(clean_s0.get("check"), dict):
        rate = clean_s0["check"].get("false_alerts_per_clean_mission")
    reference = None
    if isinstance(clean_s0.get("sequential_reference"), dict):
        reference = clean_s0["sequential_reference"].get("false_alerts_per_clean_mission")
    if reference is None:
        rows = {}
        import json
        ref_manifest = ROOT / "data/derived/balanced_pilot_v1-development-648/extraction_manifest.jsonl"
        for line in ref_manifest.read_text().splitlines():
            item = json.loads(line)
            rows[item["run_id"]] = item["dataset_episode_key"].rsplit("-", 2)[-2]
        counts = [n for run, n in fa["sequential_reference"]["per_mission"].items() if rows.get(run) == "clean_s0"]
        reference = sum(counts) / len(counts) if counts else None
    if criterion == "absolute":
        if rate is None or rate > budget:
            findings.append(f"clean_s0 false alerts per mission {rate} exceed the {budget} budget")
    else:
        if rate is None or reference is None or rate > reference + tolerance:
            findings.append(f"clean_s0 false alerts per mission {rate} exceed the sequential reference "
                            f"{reference} by more than {tolerance}")
    worst = {}
    for condition in ("clean_s0", "oscillation"):
        block = report["feature_shift"].get(condition, {})
        columns = block.get("features", block.get("columns", {})) if isinstance(block, dict) else {}
        for name in block.get("features_beyond_maximum", []) if isinstance(block, dict) else []:
            findings.append(f"{condition}: {name} beyond the SMD bound")
        for name, item in columns.items():
            if not isinstance(item, dict):
                continue
            smd = item.get("pooled_smd", item.get("smd"))
            if isinstance(smd, (int, float)):
                worst[f"{condition}:{name}"] = smd
                if abs(smd) > smd_bound:
                    findings.append(f"{condition}: {name} shifted {smd:+.3f} beyond {smd_bound}")
        for finding in report.get("findings", []):
            if finding.startswith(f"{condition}:"):
                findings.append(finding)
    top = sorted(worst.items(), key=lambda kv: -abs(kv[1]))[:5]
    return sorted(set(findings)), {"clean_s0_false_alerts_per_mission": rate,
                                   "clean_s0_sequential_reference": reference,
                                   "largest_s0_smd": dict(top)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", default=[
        "reports/integrity/concurrency_shift_check_v1.yaml",
        "reports/integrity/concurrency_shift_check_v2.yaml",
    ])
    parser.add_argument("--policy", type=Path, default=ROOT / "configs/concurrency_shift_policy.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/integrity/concurrency_phased_admission_v1.yaml")
    parser.add_argument("--criterion", choices=("absolute", "non_inferiority"), default="absolute",
                        help="absolute: check rate within the 0.10 budget; non_inferiority: check rate within "
                             "tolerance of the sequential reference rate on the same maps and conditions")
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args()
    policy = yaml.safe_load(args.policy.read_text(encoding="utf-8"))

    def find(document, key):
        if isinstance(document, dict):
            if key in document:
                return document[key]
            for value in document.values():
                found = find(value, key)
                if found is not None:
                    return found
        return None

    smd_bound = float(find(policy, "maximum_absolute_smd") or 0.5)
    budget = float(find(policy, "budget_per_clean_mission") or 0.10)
    evidence = {}
    all_findings: list[str] = []
    for relative in args.reports:
        path = ROOT / relative
        report = yaml.safe_load(path.read_text(encoding="utf-8"))
        findings, summary = s0_findings(report, smd_bound, budget, args.criterion, args.tolerance)
        evidence[relative] = {"sha256": sha256_file(path), "check_passed_overall": report["passed"],
                              "s0_findings": findings, **summary}
        all_findings.extend(f"{relative}: {f}" for f in findings)
    record = {
        "schema_version": 1,
        "purpose": "phased execution admission: S0 episodes six-wide, S3 episodes strictly alone",
        "derived_from": list(args.reports),
        "policy": str(args.policy.relative_to(ROOT)),
        "policy_sha256": sha256_file(args.policy),
        "rule": ("admit only if every completed concurrency check shows clean S0 false alerts per "
                 "mission within budget and no S0-condition deployable feature beyond the SMD bound; "
                 "S3-alone is the sequential reference condition"),
        "smd_bound": smd_bound,
        "false_alert_budget_per_clean_mission": budget,
        "false_alert_criterion": args.criterion,
        "non_inferiority_tolerance": args.tolerance if args.criterion == "non_inferiority" else None,
        "evidence": evidence,
        "findings": all_findings,
        "passed": not all_findings,
        "phased_execution": {"s0": 6, "s3": 1},
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protected_test_used": False,
        "new_measurement_performed": False,
    }
    publish_new_bytes(args.output, yaml.safe_dump(record, sort_keys=False).encode("utf-8"))
    print(f"phased admission passed={record['passed']} -> {args.output}")
    for finding in all_findings:
        print(" -", finding)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
