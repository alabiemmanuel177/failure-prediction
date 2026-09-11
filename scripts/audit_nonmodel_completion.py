#!/usr/bin/env python3
"""Write and report the fail-closed pre-model engineering completion audit."""

from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nonmodel_completion import audit_nonmodel  # noqa: E402


def main() -> int:
    report = audit_nonmodel(ROOT)
    output = ROOT / "reports/status/nonmodel_completion.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    if report["status"] != "complete":
        print("NON-MODEL WORK INCOMPLETE")
        for finding in report["incomplete_checks"]:
            print(f"- {finding}")
        print(f"report: {output}")
        return 1
    print(f"NON-MODEL MACHINE WORK COMPLETE: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
