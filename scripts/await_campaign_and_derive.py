#!/usr/bin/env python3
"""Wait for a non-model campaign to complete, then finalise and derive it.

Polls the campaign's cumulative report without touching the runner (it never starts,
stops or retries an episode). When `finalize_nonmodel_collection.py` reports the
campaign complete and artifact-valid, it publishes the immutable inventory, runs the
offline causal extraction and the all-decisions assembly, and appends one research-log
record. Every step is idempotent: existing immutable outputs are left untouched and a
re-run resumes at the first missing artifact.
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
sys.path.insert(0, str(ROOT / "scripts"))

from finalize_nonmodel_collection import collection_spec, report_is_complete  # noqa: E402

ACTOR = "Claude Fable 5.1 (AI assistant, directed by the researcher)"


def run(command: list[str], env_ros: bool = False) -> None:
    if env_ros:
        joined = " ".join(f"'{part}'" for part in command)
        command = ["bash", "-lc", f"source {ROOT}/scripts/env_research2.sh >/dev/null 2>&1; nice -n 19 {joined}"]
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validation", "targeted", "supplement"))
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--poll-seconds", type=int, default=600)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--then-start-manifest", type=Path, default=None,
                        help="after derivation, start the continuous runner on this manifest")
    parser.add_argument("--then-await-stage", default=None,
                        help="stage name to await/finalise/derive after --then-start-manifest")
    parser.add_argument("--then-dataset-id", default=None)
    args = parser.parse_args()
    spec = collection_spec(args.stage)
    while not report_is_complete(spec):
        monitor = ROOT / "reports/status/targeted_development_monitor.yaml"
        progress = ""
        if monitor.exists():
            counts = (yaml.safe_load(monitor.read_text(encoding="utf-8")) or {}).get("counts", {})
            progress = f" resolved={counts.get('scientifically_resolved_design_keys')}/{counts.get('expected_design_keys')}"
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} waiting for {spec.report.name}{progress}", flush=True)
        time.sleep(args.poll_seconds)
    if not spec.dataset_manifest.exists():
        run([sys.executable, str(ROOT / "scripts/finalize_nonmodel_collection.py"), args.stage])
    derived = ROOT / "data/derived" / args.dataset_id
    if not (derived / "extraction_report.yaml").exists():
        run([sys.executable, str(ROOT / "scripts/extract_dataset_sequences.py"),
             "--inventory", str(spec.inventory), "--dataset-id", args.dataset_id,
             "--workers", str(args.workers)], env_ros=True)
    if not (derived / "decisions_report.yaml").exists():
        run([sys.executable, str(ROOT / "scripts/assemble_dataset_decisions.py"),
             "--dataset-id", args.dataset_id])
    report = yaml.safe_load((derived / "extraction_report.yaml").read_text(encoding="utf-8"))
    metadata = {
        "dataset_id": args.dataset_id, "counts": report["counts"],
        "extraction_manifest_sha256": report["extraction_manifest_sha256"],
        "protected_test_used": False, "training_performed": False,
    }
    run([sys.executable, str(ROOT / "scripts/research_log.py"), "add", "--kind", "experiment",
         "--actor", ACTOR, "--message",
         f"Finalised the {args.stage} campaign into its immutable inventory and derived causal "
         f"sequences and all-decision artifacts for {args.dataset_id}; no model fitted, no protected data touched.",
         "--metadata", json.dumps(metadata, sort_keys=True)])
    print(f"DONE {args.dataset_id}", flush=True)
    if args.then_start_manifest:
        lock_busy = True
        while lock_busy:
            result = subprocess.run(["pgrep", "-f", "run_balanced_pilot_continuous.py"], capture_output=True, text=True)
            lock_busy = bool(result.stdout.strip())
            if lock_busy:
                print("waiting for the previous continuous runner to exit", flush=True)
                time.sleep(60)
        log_path = ROOT / "logs" / f"{args.then_start_manifest.stem}.continuous.log"
        command = (f"cd {ROOT} && source scripts/env_research2.sh && nohup python3 "
                   f"scripts/run_balanced_pilot_continuous.py --manifest {args.then_start_manifest} "
                   f"> {log_path} 2>&1 &")
        print("$ " + command, flush=True)
        subprocess.run(["bash", "-lc", command], check=True)
        run([sys.executable, str(ROOT / "scripts/research_log.py"), "add", "--kind", "experiment",
             "--actor", ACTOR, "--message",
             f"Started the preregistered {args.then_start_manifest.name} campaign after the previous "
             "campaign was finalised and derived; sequential execution, protected data untouched.",
             "--metadata", json.dumps({"manifest": str(args.then_start_manifest), "protected_test_used": False})])
        if args.then_await_stage and args.then_dataset_id:
            follow = [sys.executable, str(ROOT / "scripts/await_campaign_and_derive.py"), args.then_await_stage,
                      "--dataset-id", args.then_dataset_id, "--poll-seconds", str(args.poll_seconds),
                      "--workers", str(args.workers)]
            print("$ " + " ".join(follow), flush=True)
            return subprocess.run(follow, check=False).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
