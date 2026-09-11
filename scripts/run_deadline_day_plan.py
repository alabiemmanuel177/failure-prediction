#!/usr/bin/env python3
"""PA-2026-09-04-01 day plan: second concurrency check, then the final model pass.

Stages (idempotent, resumable):
  1. wait until the supplement is finalised and derived
  2. wait until no campaign runner holds a lock, then run the 36-episode
     concurrency_shift_check_v2 (six workers, at most one S3 episode at a time)
  3. inventory, derive and score it with the preliminary_v1 P3 proxy, evaluate the
     unchanged policy -> reports/integrity/concurrency_shift_check_v2.yaml
  4. only then run the final_v1 model-development pass (GPU work is kept off the
     host while the check measures perception latency)
The model freeze remains a separate researcher-signed step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv/bin/python"
ACTOR = "Claude Fable 5.1 (AI assistant, directed by the researcher)"
CHECK_MANIFEST = ROOT / "data/manifests/concurrency_shift_check_v2.yaml"
CHECK_ID = "concurrency_shift_check_v2-development-36"
CHECK_REPORT = ROOT / "reports/integrity/concurrency_shift_check_v2.yaml"
SUPPLEMENT_ID = "development_supplement_v1-development-504"
PROXY_MODEL = ROOT / "models/preliminary_v1/p3_causal_tcn__seed20260903"
PROXY_CALIBRATOR = ROOT / "reports/calibration/preliminary_v1/p3_causal_tcn.calibrator.json"
PROXY_THRESHOLD = ROOT / "reports/thresholds/preliminary_v1/p3_causal_tcn.yaml"
PRED_DIR = ROOT / "reports/predictions/concurrency_shift_check_v2"
FITTING_POOL = ("balanced_pilot_v1-development-648", "targeted_development_v1-development-1212",
                SUPPLEMENT_ID)


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}", flush=True)


def run(command: list[str], ros: bool = False) -> int:
    if ros:
        joined = " ".join(f"'{part}'" for part in command)
        command = ["bash", "-lc", f"cd {ROOT} && source scripts/env_research2.sh && {joined}"]
    log("$ " + " ".join(command))
    return subprocess.run(command, check=False).returncode


def runner_active() -> bool:
    return bool(subprocess.run(["pgrep", "-f", "run_balanced_pilot_continuous.py|run_campaign_parallel.py|run_research2_episode.py"],
                               capture_output=True, text=True).stdout.strip())


def report_complete(path: Path, expected: int) -> bool:
    if not path.exists():
        return False
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    counts = value.get("counts", {})
    return bool(value.get("complete_and_artifact_valid") is True
                and counts.get("expected") == counts.get("usable") == expected)


def research_log(kind: str, message: str, metadata: dict) -> None:
    run([sys.executable, str(ROOT / "scripts/research_log.py"), "add", "--kind", kind,
         "--actor", ACTOR, "--message", message, "--metadata", json.dumps(metadata, sort_keys=True)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    derived = ROOT / "data/derived"

    while not (derived / SUPPLEMENT_ID / "decisions_report.yaml").exists():
        log("waiting for supplement derivation")
        time.sleep(args.poll_seconds)

    check_cumulative = ROOT / "reports/pilot/concurrency_shift_check_v2.cumulative36.yaml"
    passed = None
    if not report_complete(check_cumulative, 36):
        while runner_active():
            log("waiting for the previous runner to exit")
            time.sleep(60)
        code = run([sys.executable, str(ROOT / "scripts/run_campaign_parallel.py"), "--manifest",
                    str(CHECK_MANIFEST), "--workers", str(args.workers)], ros=True)
        if code or not report_complete(check_cumulative, 36):
            research_log("data_issue", "Second concurrency shift check campaign did not complete cleanly; "
                         "remaining campaigns stay sequential per PA-2026-09-04-01.",
                         {"returncode": code, "protected_test_used": False})
            passed = False
    if passed is None and not CHECK_REPORT.exists():
        if not (ROOT / "data/manifests/concurrency_shift_check_v2.episodes.jsonl").exists():
            run([sys.executable, str(ROOT / "scripts/build_campaign_inventory.py"), "--manifest",
                 str(CHECK_MANIFEST), "--report", str(check_cumulative), "--dataset-id", CHECK_ID])
        if not (derived / CHECK_ID / "extraction_report.yaml").exists():
            run(["nice", "-n", "19", sys.executable, str(ROOT / "scripts/extract_dataset_sequences.py"),
                 "--inventory", str(ROOT / "data/manifests/concurrency_shift_check_v2.episodes.jsonl"),
                 "--dataset-id", CHECK_ID, "--workers", "4"], ros=True)
        if not (derived / CHECK_ID / "decisions_report.yaml").exists():
            run([sys.executable, str(ROOT / "scripts/assemble_dataset_decisions.py"), "--dataset-id", CHECK_ID])
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        raw = PRED_DIR / "p3_proxy.check.raw.csv"
        calibrated = PRED_DIR / "p3_proxy.check.calibrated.csv"
        if not raw.exists():
            run([str(VENV), str(ROOT / "scripts/predict_decisions.py"), "--model-dir", str(PROXY_MODEL),
                 "--dataset", CHECK_ID, "--output", str(raw)])
        if not calibrated.exists():
            run([sys.executable, str(ROOT / "scripts/apply_calibration.py"), str(raw), str(PROXY_CALIBRATOR),
                 str(calibrated)])
        reference = ROOT / "reports/predictions/concurrency_shift_check_v1/p3_proxy.reference.calibrated.csv"
        run([sys.executable, str(ROOT / "scripts/evaluate_concurrency_shift.py"), "--check-dataset-id", CHECK_ID,
             "--predictions", str(calibrated), "--reference-predictions", str(reference),
             "--threshold-record", str(PROXY_THRESHOLD), "--output", str(CHECK_REPORT)])
    if passed is None:
        passed = bool(CHECK_REPORT.exists()
                      and (yaml.safe_load(CHECK_REPORT.read_text(encoding="utf-8")) or {}).get("passed") is True)
    log(f"second concurrency shift check passed={passed}")
    research_log("experiment", f"Second concurrency shift check (six workers, S3 serialised) evaluated: passed={passed}; "
                 f"remaining campaigns run {'with that topology' if passed else 'sequentially'} per PA-2026-09-04-01.",
                 {"report": str(CHECK_REPORT.relative_to(ROOT)), "passed": passed, "protected_test_used": False})

    comparison = ROOT / "reports/model_selection/final_v1/validation_comparison.yaml"
    if not comparison.exists():
        command = [sys.executable, str(ROOT / "scripts/run_model_development.py"), "--tag", "final_v1",
                   "--selection-dataset", "balanced_validation_v1-validation-324"]
        for dataset in FITTING_POOL:
            command += ["--train-dataset", dataset]
        code = run(command)
        if code:
            return code
    log("FINAL PASS COMPLETE: freeze packet ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
