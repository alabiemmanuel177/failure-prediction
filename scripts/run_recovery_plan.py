#!/usr/bin/env python3
"""Recovery work package after the freeze: pilot -> selector -> paired campaign -> H6.

Stages (idempotent; every output is skipped when it already exists):
  0. wait until severity_stress_v1 has its complete cumulative record and no simulator
     runner is active (the two campaigns share the six worker slots)
  1. recovery_pilot_v1 (882 validation-map episodes, R0 + six forced actions) in the
     phased pattern of PA-2026-09-04-02: S0 six-wide, then S3 alone; pre-goal startup
     invalids are auto-declared as same-cell replacements (scripts/declare_startup_replacements.py)
  2. outcome table -> replay cost table -> R3 cost-sensitive selector fitted on the
     validation split under PA-2026-09-03-04 (models/recovery_selector/r3_cost_sensitive_ridge_v1.json)
  3. paired_recovery_v1 (1,512 protected held-out episodes, R0/R2/R3) phased the same way,
     behind the confirmatory readiness gate, with the outcome-free completion record
  4. protected outcome table -> scripts/analyze_paired_recovery.py (H6) -> final hypothesis table

Runs as the transient user unit research2-recovery (see scripts/resume_after_reboot.sh);
a GPU wedge stops the plan with exit 3 (auto-reboot when RESEARCH2_AUTO_REBOOT=1).
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_post_freeze_plan import (  # noqa: E402
    ACTOR, VENV, log, phased_campaign, report_complete, run, run_phase, wait_idle,
)

PILOT = ROOT / "data/manifests/recovery_pilot_v1.yaml"
PAIRED = ROOT / "data/manifests/paired_recovery_v2.yaml"  # v1 ran without live execution (PA-2026-09-10-01)
SEVERITY_DONE = ROOT / "reports/confirmatory/severity_stress_v1.cumulative1512.yaml"
PILOT_DONE = ROOT / "reports/validation/recovery_pilot_v1.cumulative882.yaml"
PAIRED_DONE = ROOT / "reports/confirmatory/paired_recovery_v2.cumulative1512.yaml"
RECOVERY = ROOT / "reports/recovery"
PILOT_OUTCOMES = RECOVERY / "recovery_pilot_v1.outcomes.csv"
PILOT_COSTS = RECOVERY / "recovery_pilot_v1.costs.csv"
SELECTOR = ROOT / "models/recovery_selector/r3_cost_sensitive_ridge_v1.json"
PAIRED_OUTCOMES = RECOVERY / "paired_recovery_v2.outcomes.csv"
H6_REPORT = RECOVERY / "paired_recovery.yaml"
FINAL_DIR = ROOT / "reports/confirmatory/final"


def record(kind: str, message: str, metadata: dict) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts/research_log.py"), "add", "--kind", kind, "--actor", ACTOR,
                    "--message", message, "--metadata", json.dumps(metadata)], check=False, stdout=subprocess.DEVNULL)


def wait_for_severity() -> None:
    waited = 0
    while not report_complete(SEVERITY_DONE, 1512):
        if waited % 1800 == 0:
            log("waiting for severity_stress_v1 to complete before the recovery pilot takes the simulators")
        time.sleep(300)
        waited += 300
    wait_idle()


PILOT_EXCLUSIONS = ROOT / "data/manifests/recovery_pilot_exclusions_v1.yaml"


def documented_exclusions() -> set[str]:
    if not PILOT_EXCLUSIONS.exists():
        return set()
    document = yaml.safe_load(PILOT_EXCLUSIONS.read_text(encoding="utf-8")) or {}
    return {str(item["episode_key"]) for item in document.get("exclusions", [])}


def pilot_complete_with_exclusions() -> bool:
    """Complete when every design cell is resolved or documented as excluded."""
    if report_complete(PILOT_DONE, 882):
        return True
    if not PILOT_DONE.exists():
        return False
    value = yaml.safe_load(PILOT_DONE.read_text(encoding="utf-8")) or {}
    counts, integrity = value.get("counts", {}), value.get("integrity", {})
    excluded = documented_exclusions()
    usable = int(counts.get("usable", -1))
    resolved = usable + len(excluded) == 882 == int(counts.get("expected", -1))
    # The summariser reports integrity as counts; the only permitted gap is the
    # documented exclusions (their cells have no scientific episode).
    missing = int(integrity.get("missing_episode_keys") or 0)
    no_other_gaps = (missing <= len(excluded) and not integrity.get("unexpected_episode_keys")
                     and not integrity.get("duplicate_episode_keys") and not integrity.get("replacement_errors"))
    return resolved and no_other_gaps and value.get("protected_test_used") is False


def pilot_campaign() -> None:
    if pilot_complete_with_exclusions():
        log("recovery_pilot_v1 already complete")
        return
    run_phase(PILOT, 6, "s0")
    run_phase(PILOT, 1, "s3")
    for replacements in sorted((ROOT / "data/manifests").glob("*_replacements_v*.yaml")):
        document = yaml.safe_load(replacements.read_text(encoding="utf-8")) or {}
        if document.get("parent_campaign_id") != PILOT.stem:
            continue
        wait_idle()
        run([sys.executable, str(ROOT / "scripts/run_balanced_pilot_replacements.py"), "--manifest",
             str(replacements)], ros=True, check=False)
    if not pilot_complete_with_exclusions():
        wait_idle()
        if PILOT_DONE.exists():
            # The wave finaliser wrote an --allow-incomplete snapshot at 882 ledger rows
            # (before the replacements); keep it aside and write the complete record.
            stale = PILOT_DONE.with_name(PILOT_DONE.stem + ".before_replacements.yaml")
            if not stale.exists():
                PILOT_DONE.rename(stale)
                log(f"moved incomplete pilot snapshot aside: {stale}")
        run([sys.executable, str(ROOT / "scripts/summarize_balanced_pilot.py"), "--manifest", str(PILOT),
             "--output", str(PILOT_DONE)], check=False)
    if not pilot_complete_with_exclusions():
        raise SystemExit("recovery_pilot_v1 did not reach a complete, artifact-valid cumulative report "
                         "(net of the documented exclusions in recovery_pilot_exclusions_v1.yaml)")


def selector() -> None:
    RECOVERY.mkdir(parents=True, exist_ok=True)
    if not PILOT_OUTCOMES.exists():
        run([VENV_PY, str(ROOT / "scripts/build_recovery_outcome_table.py"), "--manifest", str(PILOT),
             "--output", str(PILOT_OUTCOMES)])
    if not PILOT_COSTS.exists():
        run([VENV_PY, str(ROOT / "scripts/build_recovery_cost_table.py"), "--outcomes", str(PILOT_OUTCOMES),
             "--manifest", str(PILOT), "--output", str(PILOT_COSTS)])
    if not SELECTOR.exists():
        SELECTOR.parent.mkdir(parents=True, exist_ok=True)
        run([VENV_PY, str(ROOT / "scripts/train_recovery_selector.py"), str(PILOT_COSTS), "--output", str(SELECTOR),
             "--fit-split", "validation", "--amendment-id", "PA-2026-09-03-04"])
        record("experiment", "R3 cost-sensitive recovery selector fitted on the recovery pilot's validation-split "
               "replay cost table under PA-2026-09-03-04 (guard-rejected actions never training targets); "
               "frozen before the paired held-out campaign.",
               {"selector": str(SELECTOR.relative_to(ROOT)), "cost_table": str(PILOT_COSTS.relative_to(ROOT)),
                "protected_outcomes_consulted": False})


def paired_campaign() -> None:
    if not SELECTOR.exists():
        raise SystemExit("paired campaign needs the frozen R3 selector")
    amendment = yaml.safe_load((ROOT / "configs/protocol_amendment_1.8.yaml").read_text(encoding="utf-8")) or {}
    if amendment.get("status") != "approved" or not amendment.get("approved_by"):
        raise SystemExit("paired_recovery_v2 requires the researcher's signature on PA-2026-09-10-01 (configs/protocol_amendment_1.8.yaml)")
    run([sys.executable, str(ROOT / "scripts/check_readiness.py"), "--stage", "confirmatory"])
    phased_campaign(PAIRED, 1512, 6)


def analyse_h6() -> None:
    if not PAIRED_OUTCOMES.exists():
        run([VENV_PY, str(ROOT / "scripts/build_recovery_outcome_table.py"), "--manifest", str(PAIRED),
             "--output", str(PAIRED_OUTCOMES), "--allow-protected-after-freeze"])
    if not H6_REPORT.exists():
        run([VENV_PY, str(ROOT / "scripts/analyze_paired_recovery.py"), str(PAIRED_OUTCOMES),
             "--manifest", str(PAIRED), "--output", str(H6_REPORT), "--allow-protected-after-freeze"])
        record("experiment", "H6 (paired closed-loop recovery, R3 versus R0 on protected held-out maps) analysed "
               "by scripts/analyze_paired_recovery.py from the exact-once paired_recovery_v1 outcome table.",
               {"report": str(H6_REPORT.relative_to(ROOT)), "protected_outcomes_consulted": True})
    for name in ("hypotheses.yaml", "hypotheses.md"):
        (FINAL_DIR / name).unlink(missing_ok=True)
    run([VENV_PY, str(ROOT / "scripts/analyze_hypotheses.py"), "--held-out",
         str(ROOT / "reports/confirmatory/held_out_map.final.yaml"), "--recovery", str(H6_REPORT),
         "--output-dir", str(FINAL_DIR)])


VENV_PY = str(VENV)


def status() -> dict:
    return {
        "severity_complete": report_complete(SEVERITY_DONE, 1512),
        "pilot_complete": pilot_complete_with_exclusions(), "documented_exclusions": sorted(documented_exclusions()),
        "pilot_outcomes": PILOT_OUTCOMES.exists(), "pilot_costs": PILOT_COSTS.exists(),
        "selector": SELECTOR.exists(),
        "paired_complete": report_complete(PAIRED_DONE, 1512),
        "paired_outcomes": PAIRED_OUTCOMES.exists(), "h6_report": H6_REPORT.exists(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print stage status and exit")
    parser.add_argument("--skip-wait", action="store_true", help="do not wait for severity (operator override)")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(status(), indent=2))
        return 0
    hold = ROOT / "logs/campaigns/recovery_plan.hold"
    if hold.exists():
        log(f"hold marker present ({hold}); not running any stage until the operator removes it")
        return 0
    if not args.skip_wait:
        wait_for_severity()
    pilot_campaign()
    selector()
    paired_campaign()
    analyse_h6()
    log("RECOVERY PLAN COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
