#!/usr/bin/env python3
"""Validate a completed supervisor threshold review without applying it."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


EVENTS = {"localisation_loss", "immobilisation", "unsafe_perception"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review", type=Path)
    args = parser.parse_args()
    document = yaml.safe_load(args.review.read_text(encoding="utf-8"))
    findings = []
    if document.get("protected_outcomes_consulted") is not False:
        findings.append("protected_outcomes_consulted must be false")
    if str(document.get("reviewer_name", "")).startswith("TODO"):
        findings.append("reviewer_name is unresolved")
    if str(document.get("reviewed_utc", "")).startswith("TODO"):
        findings.append("reviewed_utc is unresolved")
    decisions = document.get("decisions", {})
    if set(decisions) != EVENTS:
        findings.append("review must contain exactly the three prespecified event decisions")
    for event in EVENTS:
        values = decisions.get(event, {})
        if values.get("decision") not in {"accept", "revise", "reject"}:
            findings.append(f"{event}: decision is not final")
        if str(values.get("rationale", "")).startswith("TODO"):
            findings.append(f"{event}: rationale is unresolved")
    if document.get("overall_decision") not in {"approved", "rejected"}:
        findings.append("overall_decision must be approved or rejected")
    if findings:
        print(f"REVIEW INVALID: {len(findings)} finding(s)")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("REVIEW VALID: decisions are complete; values have not been applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
