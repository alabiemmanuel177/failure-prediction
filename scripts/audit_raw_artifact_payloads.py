#!/usr/bin/env python3
"""Fail unless every zero-length raw artifact belongs to a declared excluded run."""

from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.artifact_integrity import audit_raw_payloads  # noqa: E402


def main() -> int:
    output = ROOT / "reports/integrity/raw_payload_integrity.yaml"
    report = audit_raw_payloads(ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    temporary.replace(output)
    if not report["passed"]:
        print(f"RAW PAYLOAD INTEGRITY FAILED: {output}")
        return 1
    counts = report["counts"]
    print(
        "RAW PAYLOAD INTEGRITY PASS: "
        f"{counts['zero_length_runs']} declared excluded loss run(s), "
        f"{counts['unexpected_zero_length_runs']} unexpected; report: {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
