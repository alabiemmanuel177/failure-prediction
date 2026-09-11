#!/usr/bin/env python3
"""Validate the minimized Research 1 catalog against structural metadata only."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import sha256_file  # noqa: E402
from src.research1_catalog import validate_structural_catalog  # noqa: E402


RESEARCH1 = Path("/home/eao/risk-calibrated-nav")
REQUIRED_TOPICS = {"/cmd_vel", "/odom", "/amcl_pose", "/scan", "/plan"}


def main() -> int:
    catalog_path = ROOT / "data/manifests/research1_temporal_catalog_v2.jsonl"
    report_path = ROOT / "reports/integrity/research1_temporal_catalog_v2.yaml"
    payload = catalog_path.read_bytes()
    records = [json.loads(line) for line in payload.splitlines() if line.strip()]
    report = yaml.safe_load(report_path.read_text(encoding="utf-8"))
    findings: list[str] = []
    split_counts = Counter(item.get("split") for item in records)
    if report.get("schema_version") != 2:
        findings.append("catalog report schema is not v2")
    if report.get("catalog", {}).get("sha256") != hashlib.sha256(payload).hexdigest():
        findings.append("catalog payload hash does not match report")
    if report.get("counts", {}).get("catalog_rows") != len(records) or len(records) != 1189:
        findings.append("catalog row count is invalid")
    if dict(split_counts) != {"development": 825, "validation": 364}:
        findings.append(f"catalog split counts are invalid: {dict(split_counts)}")
    source = report.get("source_audit", {})
    source_path = ROOT / str(source.get("path", ""))
    if not source_path.is_file() or source.get("sha256") != sha256_file(source_path):
        findings.append("source compatibility audit is missing or changed")
    controls = report.get("controls", {})
    if not (
        controls.get("protected_test_rows_exported") == 0
        and controls.get("nonrequired_topic_names_exported") is False
        and controls.get("admitted_for_model_use") is False
    ):
        findings.append("catalog minimization or admission controls are invalid")
    findings.extend(validate_structural_catalog(
        records, research1_root=RESEARCH1, required_topics=REQUIRED_TOPICS,
    ))
    if findings:
        print("RESEARCH 1 STRUCTURAL CATALOG INVALID", file=sys.stderr)
        for finding in findings[:50]:
            print(f"- {finding}", file=sys.stderr)
        return 1
    print(
        "RESEARCH 1 STRUCTURAL CATALOG VALID: 1189 rows; development=825, "
        "validation=364; source metadata exact; 0 protected rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
