#!/usr/bin/env python3
"""Fail unless every artifact required for the complete Research 2 claim exists."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.completion import audit_completion


def main() -> int:
    findings = audit_completion(ROOT)
    verification_commands = {
        "confirmatory readiness gate does not pass": [
            sys.executable, str(ROOT / "scripts/check_readiness.py"),
            "--stage", "confirmatory",
        ],
        "manual causal-label audit gate does not pass": [
            sys.executable, str(ROOT / "scripts/check_manual_audit_gate.py"),
        ],
        "Research 1 content boundary does not pass": [
            sys.executable, str(ROOT / "scripts/check_research1_boundary.py"),
        ],
        "tamper-evident research log does not verify": [
            sys.executable, str(ROOT / "scripts/research_log.py"), "verify",
        ],
        "development dataset inventory does not validate": [
            sys.executable, str(ROOT / "scripts/validate_development_dataset_inventory.py"),
        ],
        "validation dataset inventory does not validate": [
            sys.executable, str(ROOT / "scripts/validate_validation_dataset_inventory.py"),
        ],
        "targeted-development dataset inventory does not validate": [
            sys.executable, str(ROOT / "scripts/validate_targeted_dataset_inventory.py"),
        ],
        "raw artifact payload integrity does not pass": [
            sys.executable, str(ROOT / "scripts/audit_raw_artifact_payloads.py"),
        ],
        "recovery guard verification does not pass": [
            sys.executable, str(ROOT / "scripts/verify_recovery_guards.py"),
        ],
    }
    for finding, command in verification_commands.items():
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode:
            findings.append(finding)
    if findings:
        print(f"RESEARCH 2 INCOMPLETE: {len(findings)} finding(s)")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("RESEARCH 2 COMPLETE: all required evidence is present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
