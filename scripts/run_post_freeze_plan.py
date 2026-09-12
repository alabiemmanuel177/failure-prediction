#!/usr/bin/env python3
"""Post-freeze confirmatory plan under PA-2026-09-04-02 (phased execution).

Preconditions: configs/model_freeze.yaml frozen by the researcher. Stages, each
idempotent and resumable:

  1. assign the protected split (six routes per map) and build the held-out and
     severity-stress manifests
  2. held-out campaign: phase A (S0, six workers) then phase B (S3, one worker)
  3. protected inventory, derivation and all-decisions artifacts (gated)
  4. held-out predictions for P1, P3 (primary), P4, P5, P6 with the frozen calibrators
     and the frozen alarm policy; confirmatory evaluation (H1-H4)
  5. severity-stress campaign, phases A and B, derived and scored
  6. unseen-family folds: fit (development/validation only) then evaluate on held-out
  7. natural-failure audit and the hypothesis table

The paired recovery study is orchestrated separately after the live bring-up.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv/bin/python"
ACTOR = "Claude Fable 5.1 (AI assistant, directed by the researcher)"
FITTING_POOL = ("balanced_pilot_v1-development-648", "targeted_development_v1-development-1212",
                "development_supplement_v1-development-504")
SELECTION = "balanced_validation_v1-validation-324"
HELD_OUT_ID = "held_out_map_v1-held_out_map_test-1008"
SEVERITY_ID = "severity_stress_v1-held_out_map_test-1512"
PRED = ROOT / "reports/predictions/held_out"


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}", flush=True)


def sweep_incomplete_model_dirs() -> None:
    """Move aside model directories a crashed or refused fit left without a checkpoint.

    The trainer refuses to overwrite an existing directory, so a directory holding only a
    FAILED marker (or nothing) would otherwise block every retry of that fold/ablation.
    Mirrors the sweep in scripts/resume_after_reboot.sh for restarts without a reboot.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for pattern in ("models/ablations/*/*", "models/unseen_family/*/*/*"):
        for directory in sorted(ROOT.glob(pattern)):
            if directory.is_dir() and not (directory / "checkpoint.pt").exists():
                target = ROOT / "models/_incomplete_runs" / f"{directory.parent.name}__{directory.name}__{stamp}"
                target.parent.mkdir(parents=True, exist_ok=True)
                directory.rename(target)
                log(f"moved incomplete model dir aside: {directory} -> {target}")


def run(command: list[str], ros: bool = False, check: bool = True) -> int:
    if ros:
        joined = " ".join(f"'{part}'" for part in command)
        command = ["bash", "-lc", f"cd {ROOT} && source scripts/env_research2.sh && {joined}"]
    log("$ " + " ".join(command))
    code = subprocess.run(command, check=False).returncode
    if code and check:
        raise SystemExit(f"stage failed with exit {code}: {' '.join(command[:4])}")
    return code


def gpu_healthy() -> bool:
    """A wedged AMDGPU makes every simulator fail preflight; never resume into it."""
    try:
        probe = subprocess.run([str(VENV), "-c", "import torch; x=torch.randn(512,512,device='cuda'); "
                                "assert float((x@x).sum()) != 0.0"], capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        # A wedged AMDGPU can hang the probe indefinitely; a hang is a wedge.
        return False
    return probe.returncode == 0 and "amdgpu" not in (probe.stderr + probe.stdout)


def handle_gpu_wedge(context: str) -> None:
    log(f"GPU WEDGE detected ({context}); stopping instead of consuming design keys")
    subprocess.run([sys.executable, str(ROOT / "scripts/research_log.py"), "add", "--kind", "data_issue",
                    "--actor", ACTOR, "--message", f"AMDGPU wedge detected during {context}: the plan stopped "
                    "instead of resuming; a host reboot clears it and the resume service restarts the plan.",
                    "--metadata", json.dumps({"context": context, "protected_outcomes_consulted": False})],
                   check=False, stdout=subprocess.DEVNULL)
    if os.environ.get("RESEARCH2_AUTO_REBOOT") == "1":
        log("auto-reboot authorised by the researcher: rebooting the host")
        subprocess.run(["systemctl", "reboot"], check=False)
        # A wedged amdgpu can stall the orderly shutdown for hours (6 September: the
        # 07:17Z reboot request only took effect at 09:10Z). If the host is still up
        # after five minutes, escalate to a forced reboot (units skipped, filesystems
        # still synced); the resume service restarts everything on the next boot.
        time.sleep(300)
        log("orderly reboot has not taken effect after 300 s: forcing")
        subprocess.run(["systemctl", "reboot", "--force"], check=False)
    sys.exit(3)


def runner_active() -> bool:
    return bool(subprocess.run(["pgrep", "-f", "run_balanced_pilot_continuous.py|run_campaign_parallel.py|run_research2_episode.py"],
                               capture_output=True, text=True).stdout.strip())


def wait_idle() -> None:
    while runner_active():
        log("waiting for the previous runner to exit")
        time.sleep(60)


def report_complete(path: Path, expected: int) -> bool:
    if not path.exists():
        return False
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    counts = value.get("counts", {})
    return bool(value.get("complete_and_artifact_valid") is True
                and counts.get("expected") == counts.get("usable") == expected)


def run_phase(manifest: Path, workers: int, systems: str, attempts: int = 400) -> None:
    """Run one phase until its keys drain, auto-declaring pre-goal startup invalids.

    The dispatcher stops on the first invalid attempt (exact-once). When the retained
    summary proves a pre-goal startup failure, the preregistered same-cell replacement
    is declared, logged, and the phase resumes; any other invalid kind stops the plan
    for a human decision.
    """
    for attempt in range(1, attempts + 1):
        wait_idle()
        if not gpu_healthy():
            handle_gpu_wedge(f"{manifest.stem} phase {systems} before run {attempt}")
        run([sys.executable, str(ROOT / "scripts/run_campaign_parallel.py"), "--manifest", str(manifest),
             "--concurrency-override", "--workers", str(workers), "--systems", systems], ros=True, check=False)
        wait_idle()
        drain = ROOT / "logs/campaigns" / f"{manifest.stem}.drain"
        if drain.exists():
            log(f"{manifest.stem} phase {systems}: drain marker present ({drain}); in-flight episodes finished, "
                "not resuming (operator reconfiguration)")
            sys.exit(0)
        if not gpu_healthy():
            handle_gpu_wedge(f"{manifest.stem} phase {systems} after run {attempt}")
        code = run([sys.executable, str(ROOT / "scripts/declare_startup_replacements.py"), "--manifest",
                    str(manifest)], check=False)
        if code:
            log(f"{manifest.stem} phase {systems}: an invalid attempt needs a human decision"); sys.exit(3)
        probe = subprocess.run([sys.executable, str(ROOT / "scripts/run_campaign_parallel.py"), "--manifest",
                                str(manifest), "--concurrency-override", "--workers", str(workers), "--systems",
                                systems, "--dry-run"], capture_output=True, text=True, check=False)
        pending = 0
        try:
            payload = json.loads(probe.stdout[probe.stdout.index("{"):])
            pending = int(payload.get("claimable_in_systems", payload.get("pending", 0)) or 0)
        except (ValueError, TypeError):
            log(f"could not parse dispatcher dry-run: {probe.stdout[-300:]} {probe.stderr[-300:]}")
        if pending == 0:
            log(f"{manifest.stem} phase {systems} drained after {attempt} run(s)")
            return
        log(f"{manifest.stem} phase {systems}: {pending} keys still pending; resuming (run {attempt + 1})")
    raise SystemExit(f"{manifest.stem} phase {systems} did not drain within {attempts} runs")


def phased_campaign(manifest: Path, expected: int, workers: int) -> None:
    cumulative = ROOT / "reports/confirmatory" / f"{manifest.stem}.cumulative{expected}.yaml"
    if report_complete(cumulative, expected):
        log(f"{manifest.stem} already complete")
        return
    run_phase(manifest, workers, "s0")
    run_phase(manifest, 1, "s3")
    # Declared infrastructure replacements run alone, sequentially, after both phases:
    # every child manifest whose parent_campaign_id is this campaign (one manifest per
    # replacement policy).
    for replacements in sorted((ROOT / "data/manifests").glob("*_replacements_v*.yaml")):
        document = yaml.safe_load(replacements.read_text(encoding="utf-8")) or {}
        if document.get("parent_campaign_id") != manifest.stem:
            continue
        wait_idle()
        run([sys.executable, str(ROOT / "scripts/run_balanced_pilot_replacements.py"), "--manifest",
             str(replacements)], ros=True, check=False)
    # Protected campaigns get an outcome-free completion record (the pilot summariser
    # refuses to call a protected campaign complete and would expose outcomes).
    if not report_complete(cumulative, expected):
        wait_idle()
        if cumulative.exists():
            # An earlier, incomplete record (written before later replacements) is
            # immutable; keep it aside so the finaliser can publish the complete one.
            stale = cumulative.with_name(cumulative.stem + f".incomplete_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.yaml")
            cumulative.rename(stale)
            log(f"moved incomplete completion record aside: {stale}")
        run([sys.executable, str(ROOT / "scripts/finalize_confirmatory_campaign.py"), "--manifest", str(manifest),
             "--output", str(cumulative)], check=False)
    if not report_complete(cumulative, expected):
        raise SystemExit(f"{manifest.stem} did not reach a complete, artifact-valid cumulative report")


def derive_protected(manifest: Path, expected: int, dataset_id: str) -> None:
    derived = ROOT / "data/derived" / dataset_id
    inventory = ROOT / "data/manifests" / f"{manifest.stem}.episodes.jsonl"
    cumulative = ROOT / "reports/confirmatory" / f"{manifest.stem}.cumulative{expected}.yaml"
    if not inventory.exists():
        run([sys.executable, str(ROOT / "scripts/build_campaign_inventory.py"), "--manifest", str(manifest),
             "--report", str(cumulative), "--dataset-id", dataset_id, "--allow-protected-after-freeze",
             "--collection-status", "held_out_confirmatory_complete"])
    if not (derived / "extraction_report.yaml").exists():
        run(["nice", "-n", "19", sys.executable, str(ROOT / "scripts/extract_dataset_sequences.py"),
             "--inventory", str(inventory), "--dataset-id", dataset_id, "--workers", "4",
             "--allow-protected-after-freeze"], ros=True)
    if not (derived / "decisions_report.yaml").exists():
        run([sys.executable, str(ROOT / "scripts/assemble_dataset_decisions.py"), "--dataset-id", dataset_id,
             "--allow-protected-after-freeze"])


def score(dataset_id: str, tag: str, freeze: dict) -> dict[str, Path]:
    """Predict, calibrate (frozen calibrators) and apply the frozen alarm policy."""
    out = PRED / tag
    out.mkdir(parents=True, exist_ok=True)
    tables: dict[str, Path] = {}
    primary_dir = ROOT / freeze["predictor"]["checkpoint"]
    primary_dir = primary_dir.parent if primary_dir.is_file() else primary_dir
    secondary_path = ROOT / "configs/model_freeze_secondary.yaml"
    secondary = (yaml.safe_load(secondary_path.read_text(encoding="utf-8")) or {}).get("secondary_predictors", {}) \
        if secondary_path.exists() else {}
    model_dirs = {"p3_causal_tcn": primary_dir}
    for model_id, entry in secondary.items():
        model_dirs[model_id] = ROOT / entry["model_dir"]
    calibrators = {"p3_causal_tcn": ROOT / freeze["calibration"]["artifact"]}
    for model_id, entry in secondary.items():
        if entry.get("calibrator"):
            calibrators[model_id] = ROOT / entry["calibrator"]
    for model_id, model_dir in model_dirs.items():
        raw = out / f"{model_id}.raw.csv"
        cal = out / f"{model_id}.calibrated.csv"
        alarmed = out / f"{model_id}.alarmed.csv"
        if not raw.exists():
            run([str(VENV), str(ROOT / "scripts/predict_decisions.py"), "--model-dir", str(model_dir),
                 "--dataset", dataset_id, "--output", str(raw), "--allow-protected-after-freeze"])
        source = raw
        if model_id in calibrators and not cal.exists():
            run([sys.executable, str(ROOT / "scripts/apply_calibration.py"), str(raw), str(calibrators[model_id]),
                 str(cal), "--allow-protected-after-freeze"])
        if model_id in calibrators:
            source = cal
        if not alarmed.exists():
            command = [sys.executable, str(ROOT / "scripts/apply_alarm_policy.py"), str(source), str(alarmed),
                       "--allow-protected-after-freeze"]
            if model_id in secondary:
                # Secondary predictors carry their own frozen validation thresholds.
                command += ["--secondary-model", model_id]
            run(command)
        tables[model_id] = alarmed
    # P1 rules. The analysis-only oracle P6 never scores protected episodes.
    for model_id, script_args in (
        ("p1_threshold_rules", None),
    ):
        raw = out / f"{model_id}.raw.csv"
        alarmed = out / f"{model_id}.alarmed.csv"
        if not raw.exists():
            if model_id == "p6_oracle":
                run([str(VENV), str(ROOT / "scripts/predict_decisions.py"), "--model-dir",
                     str(ROOT / "models/final_v1/p6_oracle__seed20260903"), "--dataset", dataset_id,
                     "--output", str(raw), "--allow-protected-after-freeze"])
            else:
                run([sys.executable, str(ROOT / "scripts/predict_threshold_rules.py"), "--dataset", dataset_id,
                     "--output", str(raw), "--allow-protected-after-freeze"])
        if not alarmed.exists():
            run([sys.executable, str(ROOT / "scripts/apply_alarm_policy.py"), str(raw), str(alarmed),
                 "--allow-protected-after-freeze"])
        tables[model_id] = alarmed
    return tables


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--skip-severity", action="store_true")
    parser.add_argument("--only-severity", action="store_true",
                        help="run only the severity-stress campaign, derivation and scoring (simulators), "
                             "leaving the GPU-only stages to the main plan")
    args = parser.parse_args()
    if not args.only_severity:
        sweep_incomplete_model_dirs()
    freeze_path = ROOT / "configs/model_freeze.yaml"
    while not freeze_path.exists():
        log("waiting for the researcher-signed model freeze")
        time.sleep(300)
    freeze = yaml.safe_load(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("frozen") is not True:
        raise SystemExit("model freeze is not frozen")

    # 1. protected split and manifests
    splits = yaml.safe_load((ROOT / "data/manifests/splits.template.yaml").read_text(encoding="utf-8"))
    if not splits["held_out_map_test"].get("maps"):
        run([sys.executable, str(ROOT / "scripts/assign_protected_split.py"), "--i-confirm-model-freeze"])
    held_out = ROOT / "data/manifests/held_out_map_v1.yaml"
    severity = ROOT / "data/manifests/severity_stress_v1.yaml"
    if not held_out.exists():
        run([sys.executable, str(ROOT / "scripts/build_confirmatory_manifests.py")])
    run([sys.executable, str(ROOT / "scripts/check_readiness.py"), "--stage", "confirmatory"])

    if args.only_severity:
        phased_campaign(severity, 1512, args.workers)
        derive_protected(severity, 1512, SEVERITY_ID)
        score(SEVERITY_ID, "severity_stress_v1", freeze)
        log("SEVERITY STRESS COMPLETE")
        return 0

    # 2-3. held-out campaign and derivation
    phased_campaign(held_out, 1008, args.workers)
    derive_protected(held_out, 1008, HELD_OUT_ID)

    # 4. scoring and confirmatory evaluation
    tables = score(HELD_OUT_ID, "held_out_map_v1", freeze)
    report = ROOT / "reports/confirmatory/held_out_map.yaml"
    if not report.exists():
        command = [sys.executable, str(ROOT / "scripts/evaluate_confirmatory.py"), "--p3", str(tables["p3_causal_tcn"]),
                   "--p1", str(tables["p1_threshold_rules"]), "--validation-p3",
                   str(ROOT / freeze["analysis"]["immutable_validation_predictions"]),
                   "--allow-protected-after-freeze"]
        for model_id in ("p4_gru", "p5_compact_transformer", "p6_oracle"):
            if model_id in tables:
                command += [f"--{model_id.split('_')[0]}", str(tables[model_id])]
        run(command)

    # 5. severity stress
    if not args.skip_severity:
        phased_campaign(severity, 1512, args.workers)
        derive_protected(severity, 1512, SEVERITY_ID)
        score(SEVERITY_ID, "severity_stress_v1", freeze)

    # 6. unseen-family folds
    unseen = ROOT / "reports/confirmatory/unseen_family.yaml"
    if not unseen.exists():
        base = [sys.executable, str(ROOT / "scripts/run_unseen_family_folds.py"),
                *[a for d in FITTING_POOL for a in ("--train-dataset", d)],
                "--selection-dataset", SELECTION, "--held-out-dataset", HELD_OUT_ID]
        run(base + ["--stage", "fit"])
        run(base + ["--stage", "evaluate", "--allow-protected-after-freeze"])

    # 7. natural failures and hypotheses
    natural = ROOT / "reports/confirmatory/natural_failure_audit.yaml"
    if not natural.exists():
        run([sys.executable, str(ROOT / "scripts/audit_natural_failures.py"),
             *[a for d in (*FITTING_POOL, SELECTION) for a in ("--dataset-id", d)],
             "--predictions", str(ROOT / freeze["analysis"]["immutable_validation_predictions"])], check=False)
    run([sys.executable, str(ROOT / "scripts/analyze_hypotheses.py")], check=False)
    log("POST-FREEZE CONFIRMATORY PLAN COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
