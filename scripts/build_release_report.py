#!/usr/bin/env python3
"""Write the immutable release record ``reports/reproduction/release.yaml``.

The record lists SHA-256 checksums of every manifest, config, model checkpoint,
prediction table, report, table, figure, card and manuscript. ``passed`` is true only
when ``scripts/audit_project_completion.py`` passes in a subprocess (tolerating only
the finding that this very file resolves) and the independent rerun record passed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import publish_new_bytes  # noqa: E402
from src.release import (  # noqa: E402
    AuditRunner, blocking_findings, checksum_tree, git_dirty, git_head, run_completion_audit,
)


CHECKSUM_PATTERNS = {
    "manifests": ["data/manifests/*.yaml", "data/manifests/*.jsonl"],
    "configs": ["configs/*.yaml", "configs/faults/*.yaml"],
    "model_checkpoints": ["models/*/checkpoint.pt", "models/*/*.json", "models/*/*.sha256",
                          "models/recovery_selector/*.json", "models/recovery_selector/*.sha256"],
    "prediction_tables": ["reports/predictions/*.csv"],
    "reports": ["reports/**/*.yaml", "reports/**/*.json", "reports/tables/*.csv", "reports/tables/*.md",
                "reports/figures/*.svg"],
    "documentation": ["docs/*.md", "manuscript/main.md", "README.md"],
}


def build_release_document(
    root: Path, output: Path, *, audit: AuditRunner = run_completion_audit, dry_run: bool = False,
) -> dict[str, Any]:
    checksums = {group: checksum_tree(root, patterns, exclude=[output])
                 for group, patterns in CHECKSUM_PATTERNS.items()}
    rerun_path = root / "reports/reproduction/independent_rerun.yaml"
    rerun = yaml.safe_load(rerun_path.read_text(encoding="utf-8")) if rerun_path.exists() else None
    rerun_passed = bool(isinstance(rerun, dict) and rerun.get("passed") is True)
    findings = [] if dry_run else audit(root)
    blocking = blocking_findings(findings, ["release"])
    if not rerun_passed:
        blocking.append("independent rerun record missing or not passed: reports/reproduction/independent_rerun.yaml")
    return {
        "schema_version": 1,
        "evidence_type": "release_record",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_head(root),
        "working_tree_dirty": git_dirty(root),
        "protected_test_used": True,
        "checksum_counts": {group: len(items) for group, items in checksums.items()},
        "checksums": checksums,
        "completion_audit": {
            "command": "python3 scripts/audit_project_completion.py",
            "tolerated_findings": ["independent reproduction evidence missing or not passed: reports/reproduction/release.yaml"],
            "findings": findings,
            "dry_run_skipped_audit": dry_run,
        },
        "independent_rerun": {"path": str(rerun_path.relative_to(root)), "passed": rerun_passed},
        "blocking_findings": blocking,
        "passed": bool(not dry_run and not blocking),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="print the record without the audit; writes nothing")
    args = parser.parse_args()
    output = args.output or (args.root / "reports/reproduction/release.yaml")
    if output.exists():
        raise SystemExit(f"refusing to overwrite immutable release record: {output}")
    document = build_release_document(args.root, output, dry_run=args.dry_run)
    if args.dry_run:
        preview = {**document, "checksums": {g: f"{len(v)} files" for g, v in document["checksums"].items()}}
        print(yaml.safe_dump(preview, sort_keys=False))
        return 0
    publish_new_bytes(output, yaml.safe_dump(document, sort_keys=False).encode("utf-8"))
    print(f"wrote {output}: passed={document['passed']}")
    if not document["passed"]:
        print("blocking findings:\n- " + "\n- ".join(document["blocking_findings"]))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
