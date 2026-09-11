#!/usr/bin/env python3
"""Execute the PA-2026-09-03-04 overnight plan without human intervention.

Stages (each idempotent; the script resumes at the first incomplete stage):

  1. wait until the targeted campaign is finalised and derived
  2. run the 36-episode concurrency shift check with six workers
  3. inventory, derive and score it with the preliminary_v1 P3 proxy
  4. evaluate the pre-declared shift policy -> reports/integrity/concurrency_shift_check_v1.yaml
  5. run the 504-episode supplement in parallel if the check passed, sequentially otherwise
  6. finalise and derive the supplement (the final model pass waiter then fires)

Nothing here freezes a model or touches protected data.
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
CHECK_MANIFEST = ROOT / "data/manifests/concurrency_shift_check_v1.yaml"
CHECK_ID = "concurrency_shift_check_v1-development-36"
CHECK_REPORT = ROOT / "reports/integrity/concurrency_shift_check_v1.yaml"
SUPPLEMENT_MANIFEST = ROOT / "data/manifests/development_supplement_v1.yaml"
SUPPLEMENT_ID = "development_supplement_v1-development-504"
TARGETED_ID = "targeted_development_v1-development-1212"
PROXY_MODEL = ROOT / "models/preliminary_v1/p3_causal_tcn__seed20260903"
PROXY_CALIBRATOR = ROOT / "reports/calibration/preliminary_v1/p3_causal_tcn.calibrator.json"
PROXY_THRESHOLD = ROOT / "reports/thresholds/preliminary_v1/p3_causal_tcn.yaml"
PRED_DIR = ROOT / "reports/predictions/concurrency_shift_check_v1"


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}", flush=True)


def run(command: list[str], ros: bool = False) -> int:
    if ros:
        joined = " ".join(f"'{part}'" for part in command)
        command = ["bash", "-lc", f"cd {ROOT} && source scripts/env_research2.sh && {joined}"]
    log("$ " + " ".join(command))
    return subprocess.run(command, check=False).returncode


def wait_for(path: Path, poll: int, label: str) -> None:
    while not path.exists():
        log(f"waiting for {label}: {path.relative_to(ROOT)}")
        time.sleep(poll)


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

    # 1. targeted derived
    wait_for(derived / TARGETED_ID / "decisions_report.yaml", args.poll_seconds, "targeted derivation")

    # 2. shift-check campaign (36 episodes, six workers)
    check_cumulative = ROOT / "reports/pilot/concurrency_shift_check_v1.cumulative36.yaml"
    if not report_complete(check_cumulative, 36):
        while subprocess.run(["pgrep", "-f", "run_balanced_pilot_continuous.py|run_campaign_parallel.py"],
                             capture_output=True, text=True).stdout.strip():
            log("waiting for the previous runner to exit")
            time.sleep(60)
        # Build the two post-freeze ROS packages now that no episode is launching; the
        # sequential campaign sources ros_ws/install per episode, so this never runs
        # while it is active.
        run(["bash", "-lc", f"cd {ROOT}/ros_ws && colcon build --symlink-install "
             "--packages-select failure_monitor recovery_manager"], ros=True)
        code = run([sys.executable, str(ROOT / "scripts/run_campaign_parallel.py"), "--manifest",
                    str(CHECK_MANIFEST), "--workers", str(args.workers)], ros=True)
        if code or not report_complete(check_cumulative, 36):
            research_log("data_issue", "Concurrency shift check campaign did not complete cleanly; "
                         "falling back to sequential execution for all remaining campaigns.",
                         {"returncode": code, "protected_test_used": False})
            passed = False
        else:
            passed = None
    else:
        passed = None

    # 3-4. derive, score and evaluate
    if passed is None and not CHECK_REPORT.exists():
        if not (ROOT / "data/manifests/concurrency_shift_check_v1.episodes.jsonl").exists():
            run([sys.executable, str(ROOT / "scripts/build_campaign_inventory.py"), "--manifest",
                 str(CHECK_MANIFEST), "--report", str(check_cumulative), "--dataset-id", CHECK_ID])
        if not (derived / CHECK_ID / "extraction_report.yaml").exists():
            run(["nice", "-n", "19", sys.executable, str(ROOT / "scripts/extract_dataset_sequences.py"),
                 "--inventory", str(ROOT / "data/manifests/concurrency_shift_check_v1.episodes.jsonl"),
                 "--dataset-id", CHECK_ID, "--workers", "4"], ros=True)
        if not (derived / CHECK_ID / "decisions_report.yaml").exists():
            run([sys.executable, str(ROOT / "scripts/assemble_dataset_decisions.py"), "--dataset-id", CHECK_ID])
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        tables = {}
        for label, dataset in (("check", CHECK_ID), ("reference", "balanced_pilot_v1-development-648")):
            raw = PRED_DIR / f"p3_proxy.{label}.raw.csv"
            calibrated = PRED_DIR / f"p3_proxy.{label}.calibrated.csv"
            if not raw.exists():
                run([str(VENV), str(ROOT / "scripts/predict_decisions.py"), "--model-dir", str(PROXY_MODEL),
                     "--dataset", dataset, "--output", str(raw)])
            if not calibrated.exists():
                run([sys.executable, str(ROOT / "scripts/apply_calibration.py"), str(raw),
                     str(PROXY_CALIBRATOR), str(calibrated)])
            tables[label] = calibrated
        run([sys.executable, str(ROOT / "scripts/evaluate_concurrency_shift.py"), "--check-dataset-id", CHECK_ID,
             "--predictions", str(tables["check"]), "--reference-predictions", str(tables["reference"]),
             "--threshold-record", str(PROXY_THRESHOLD), "--output", str(CHECK_REPORT)])
    if passed is None:
        passed = bool(CHECK_REPORT.exists()
                      and (yaml.safe_load(CHECK_REPORT.read_text(encoding="utf-8")) or {}).get("passed") is True)
    log(f"concurrency shift check passed={passed}")
    research_log("experiment", f"Concurrency shift check evaluated: passed={passed}; remaining campaigns run "
                 f"{'with six workers' if passed else 'sequentially'} per PA-2026-09-03-04.",
                 {"report": str(CHECK_REPORT.relative_to(ROOT)), "passed": passed, "protected_test_used": False})

    # 5. supplement
    supplement_cumulative = ROOT / "reports/pilot/development_supplement_v1.cumulative504.yaml"
    if not report_complete(supplement_cumulative, 504):
        while subprocess.run(["pgrep", "-f", "run_balanced_pilot_continuous.py|run_campaign_parallel.py"],
                             capture_output=True, text=True).stdout.strip():
            log("waiting for the previous runner to exit")
            time.sleep(60)
        if passed:
            run([sys.executable, str(ROOT / "scripts/run_campaign_parallel.py"), "--manifest",
                 str(SUPPLEMENT_MANIFEST), "--workers", str(args.workers), "--concurrency-override"], ros=True)
        else:
            run([sys.executable, str(ROOT / "scripts/run_balanced_pilot_continuous.py"), "--manifest",
                 str(SUPPLEMENT_MANIFEST)], ros=True)
    # 6. finalise and derive (idempotent)
    return run([sys.executable, str(ROOT / "scripts/await_campaign_and_derive.py"), "supplement",
                "--dataset-id", SUPPLEMENT_ID, "--poll-seconds", "120"])


if __name__ == "__main__":
    raise SystemExit(main())
