"""Release-package helpers: completion-audit tolerance, checksums and git identity.

The completion audit (``src.completion.audit_completion``) demands the finished
release report, final cards and marker-free manuscript. Those four artefacts are
produced *by* the package scripts, so each script tolerates exactly the findings that
its own output resolves and nothing else; every scientific-evidence finding still
blocks.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
from typing import Callable, Iterable


PACKAGE_FINDINGS = {
    "release": "independent reproduction evidence missing or not passed: reports/reproduction/release.yaml",
    "model_card": "final model card is missing",
    "dataset_card": "final dataset card is missing",
    "manuscript": "manuscript still contains pending result or release markers",
}
AuditRunner = Callable[[Path], list[str]]


def run_completion_audit(root: Path) -> list[str]:
    """Run scripts/audit_project_completion.py in a subprocess and parse its findings."""
    result = subprocess.run(
        [sys.executable, str(root / "scripts/audit_project_completion.py")],
        check=False, capture_output=True, text=True, cwd=root,
    )
    if result.returncode == 0:
        return []
    findings = [line[2:].strip() for line in result.stdout.splitlines() if line.startswith("- ")]
    return findings or [f"completion audit failed without findings (rc={result.returncode})"]


def blocking_findings(findings: Iterable[str], tolerated: Iterable[str]) -> list[str]:
    allowed = {PACKAGE_FINDINGS[name] for name in tolerated}
    return [finding for finding in findings if finding not in allowed]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_tree(root: Path, patterns: Iterable[str], exclude: Iterable[Path] = ()) -> dict[str, str]:
    excluded = {path.resolve() for path in exclude}
    checksums: dict[str, str] = {}
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path.is_file() and path.resolve() not in excluded and not path.name.startswith("."):
                checksums[str(path.relative_to(root))] = sha256_file(path)
    return checksums


def git_head(root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def git_dirty(root: Path) -> bool:
    result = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False)
    return bool(result.stdout.strip())
