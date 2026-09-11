#!/usr/bin/env python3
"""Orchestrate the seven prespecified leave-one-family-out (unseen-family) folds.

For every family F in ``splits.template.yaml:unseen_family_folds``:

``--stage fit`` (allowed before the model freeze; development/validation data only)
  1. P3: train with ``--exclude-family F`` through the contract training CLI
     (``scripts/train_predictor.py``), predict the validation dataset, drop family-F
     validation episodes, select the calibrator (``scripts/select_calibration.py``),
     apply it (``scripts/apply_calibration.py``) and select the fold's own budget
     threshold (``scripts/select_alarm_threshold.py``);
  2. P1: tune the transparent threshold rules with ``scripts/tune_threshold_rules.py
     --exclude-family F`` (family-F episodes leave both the development quantile grid
     and the validation tuning set) and freeze the fold's rules into the fold work root
     with ``--rules-output``; P1's alarm threshold is the fixed 0.5 on the {0, 1} rule
     score, recorded in ``threshold_p1.yaml`` without any further selection.

``--stage evaluate`` (post-freeze only: ``--allow-protected-after-freeze`` and a
passing ``check_readiness --stage confirmatory``)
  3. predict the held-out dataset with ``--allow-protected-after-freeze`` (P3 through
     ``scripts/predict_decisions.py``, P1 through ``scripts/predict_threshold_rules.py
     --rules <fold rules>``), keep family-F and clean held-out episodes, apply the fold
     calibrator (P3) and the fold threshold (alarm policy applied in-process with the
     frozen persistence and cooldown), and evaluate per-family recall of P3 against P1.
     The fold threshold is applied in-process (``src.evaluation.apply_alarm_policy``)
     because ``scripts/apply_alarm_policy.py`` deliberately forbids explicit thresholds
     on held-out tables; the fold threshold is validation-selected, never test-adapted.
  4. write ``reports/confirmatory/unseen_family.yaml`` (per-family results, H5 = count of
     families where P3 > P1, no pooled claim, ``complete: true``).

``--dry-run`` prints every command without running anything. ``--assemble-only``
skips the commands of the evaluate stage and only reads existing fold tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_confirmatory import (
    ALARM_COLUMN, CONTRACT_COLUMNS, episode_outcomes, hierarchical_bootstrap, load_table,
    paired_recall_difference, summarize, truth,
)
from src.dataset_inventory import publish_new_bytes
from src.evaluation import AlarmPolicy, apply_alarm_policy
from src.protected_data import HELD_OUT_SPLIT, enforce_protected_boundary

H5_MINIMUM_FAMILIES = 5
DEFAULT_TRAIN_DATASETS = ("balanced_pilot_v1-development-648",)
DEFAULT_SELECTION_DATASET = "balanced_validation_v1-validation-324"
DEFAULT_HELD_OUT_DATASET = "held_out_map_v1-held_out_map_test-960"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fold_paths(work_root: Path, family: str) -> dict[str, Path]:
    work = work_root / family
    return {
        "work": work,
        "validation_p3": work / "validation_p3.csv",
        "validation_p3_excluded": work / "validation_p3_excluded.csv",
        "validation_p3_calibrated": work / "validation_p3_calibrated.csv",
        "calibration": work / "calibration_p3.yaml",
        "calibrator": work / "calibrator_p3.json",
        "threshold_p3": work / "threshold_p3.yaml",
        "validation_p1": work / "validation_p1.csv",
        "tuning_p1": work / "tuning_p1.yaml",
        "rules_p1": work / "rules_p1.yaml",
        "threshold_p1": work / "threshold_p1.yaml",
        "held_out_p3": work / "held_out_p3.csv",
        "held_out_p3_family": work / "held_out_p3_family.csv",
        "held_out_p3_calibrated": work / "held_out_p3_calibrated.csv",
        "held_out_p3_alarmed": work / "held_out_p3_alarmed.csv",
        "held_out_p1": work / "held_out_p1.csv",
        "held_out_p1_family": work / "held_out_p1_family.csv",
        "held_out_p1_alarmed": work / "held_out_p1_alarmed.csv",
    }


def model_dir(root: Path) -> Path | None:
    """The single trained run under ``root`` (contract: models/<run_name>/)."""
    if not root.is_dir():
        return None
    candidates = sorted(path.parent for path in root.glob("*/training_record.json"))
    if len(candidates) != 1:
        return None
    return candidates[0]


def write_rows(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def filter_rows(source: Path, target: Path, keep) -> int:
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames or [])
        rows = [row for row in reader if keep(row)]
    write_rows(target, rows, columns)
    return len(rows)


def apply_fold_policy(source: Path, target: Path, *, threshold: float, alarm: dict) -> None:
    rows = load_table(source, require_alarm=False)
    persistence = alarm["persistence"]
    policy = AlarmPolicy(
        float(threshold), int(persistence["required_above_threshold"]),
        int(persistence["decisions_considered"]), float(alarm["cooldown_seconds"]),
    )
    episodes: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        episodes.setdefault(row["run_id"], []).append(row)
    output = []
    for run_id in sorted(episodes):
        episode_rows = sorted(episodes[run_id], key=lambda row: float(row["decision_time"]))
        output.extend(apply_alarm_policy(episode_rows, policy))
    columns = [*rows[0].keys()]
    for column in (ALARM_COLUMN, "persistent"):
        if column not in columns:
            columns.append(column)
    write_rows(target, [{k: v for k, v in row.items()} for row in output], columns)


P1_MODEL_ID = "p1_threshold_rules"
P1_SCORE_THRESHOLD = 0.5


def write_p1_threshold(paths: dict[str, Path], family: str) -> None:
    """P1 alarms on rule score >= 0.5 (the score is binary); no threshold search."""
    document = {
        "model_id": P1_MODEL_ID,
        "threshold": P1_SCORE_THRESHOLD,
        "selection_split": "validation",
        "protected_test_used": False,
        "selection": "fixed decision threshold on the binary rule score; rules tuned by "
                     "tune_threshold_rules.py under the frozen clean-mission budget",
        "excluded_family": family,
        "rules": str(paths["rules_p1"]),
        "rules_sha256": sha256_file(paths["rules_p1"]),
        "tuning_record": str(paths["tuning_p1"]),
        "tuning_record_sha256": sha256_file(paths["tuning_p1"]),
    }
    publish_new_bytes(paths["threshold_p1"], yaml.safe_dump(document, sort_keys=False).encode("utf-8"))


class Runner:
    def __init__(self, *, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.commands: list[str] = []

    def run(self, command: list[str], *, skip_if: Path | None = None) -> None:
        rendered = " ".join(shlex.quote(str(part)) for part in command)
        if skip_if is not None and skip_if.exists():
            self.commands.append(f"[exists, skipped] {rendered}")
            print(self.commands[-1], flush=True)
            return
        self.commands.append(rendered)
        print(rendered, flush=True)
        if not self.dry_run:
            subprocess.run([str(part) for part in command], check=True, cwd=ROOT)

    def internal(self, description: str, action, *, skip_if: Path | None = None) -> None:
        line = f"[internal] {description}"
        if skip_if is not None and skip_if.exists():
            line = f"[exists, skipped] {line}"
        self.commands.append(line)
        print(line, flush=True)
        if not self.dry_run and not (skip_if is not None and skip_if.exists()):
            action()


def fit_family(family: str, args, runner: Runner) -> None:
    paths = fold_paths(args.work_root, family)
    model_root = args.model_root / family
    # P3: contract training CLI with the family excluded from fitting and selection.
    target = model_root / "p3_causal_tcn"
    command = [args.python, ROOT / "scripts/train_predictor.py", "--model-id", "p3_causal_tcn",
               "--config", args.p3_config]
    for dataset in args.train_dataset:
        command.extend(["--train-dataset", dataset])
    command.extend(["--selection-dataset", args.selection_dataset, "--exclude-family", family,
                    "--seed", args.seed, "--output-root", target])
    runner.run(command, skip_if=target if model_dir(target) else None)
    trained = model_dir(target)
    model_arg = trained if trained is not None else target / "<p3_run_name>"
    runner.run([args.python, ROOT / "scripts/predict_decisions.py", "--model-dir", model_arg,
                "--dataset", args.selection_dataset, "--output", paths["validation_p3"]],
               skip_if=paths["validation_p3"])
    excluded = paths["validation_p3_excluded"]
    runner.internal(
        f"drop fault_family == {family} rows: {paths['validation_p3']} -> {excluded}",
        lambda src=paths["validation_p3"], dst=excluded: filter_rows(
            src, dst, lambda row: row["fault_family"] != family
            and not truth(row["protected_test_used"])),
        skip_if=excluded,
    )
    runner.run([args.python, ROOT / "scripts/select_calibration.py", paths["validation_p3_excluded"],
                "--model-id", "p3_causal_tcn", "--output", paths["calibration"],
                "--calibrator-output", paths["calibrator"]], skip_if=paths["calibrator"])
    runner.run([args.python, ROOT / "scripts/apply_calibration.py", paths["validation_p3_excluded"],
                paths["calibrator"], paths["validation_p3_calibrated"]],
               skip_if=paths["validation_p3_calibrated"])
    runner.run([sys.executable, ROOT / "scripts/select_alarm_threshold.py",
                paths["validation_p3_calibrated"], paths["threshold_p3"]], skip_if=paths["threshold_p3"])
    # P1: rule tuning with the family excluded from the development quantile grid and the
    # validation tuning set; the fold's frozen rules stay in the fold work root.
    command = [args.python, ROOT / "scripts/tune_threshold_rules.py"]
    for dataset in args.train_dataset:
        command.extend(["--development-dataset", dataset])
    command.extend(["--validation-dataset", args.selection_dataset, "--exclude-family", family,
                    "--output", paths["tuning_p1"], "--predictions", paths["validation_p1"],
                    "--rules-output", paths["rules_p1"]])
    runner.run(command, skip_if=paths["rules_p1"])
    runner.internal(
        f"write fixed P1 threshold {P1_SCORE_THRESHOLD} on the rule score -> {paths['threshold_p1']}",
        lambda: write_p1_threshold(paths, family), skip_if=paths["threshold_p1"],
    )


def evaluate_family_commands(family: str, args, runner: Runner, alarm: dict) -> None:
    paths = fold_paths(args.work_root, family)
    model_root = args.model_root / family
    trained = model_dir(model_root / "p3_causal_tcn")
    model_arg = trained if trained is not None else model_root / "p3_causal_tcn" / "<p3_run_name>"
    runner.run([args.python, ROOT / "scripts/predict_decisions.py", "--model-dir", model_arg,
                "--dataset", args.held_out_dataset, "--output", paths["held_out_p3"],
                "--allow-protected-after-freeze"], skip_if=paths["held_out_p3"])
    runner.run([args.python, ROOT / "scripts/predict_threshold_rules.py", "--rules", paths["rules_p1"],
                "--dataset", args.held_out_dataset, "--output", paths["held_out_p1"],
                "--allow-protected-after-freeze"], skip_if=paths["held_out_p1"])
    for key in ("p3", "p1"):
        family_table = paths[f"held_out_{key}_family"]
        runner.internal(
            f"keep fault_family in ({family}, none) rows: {paths[f'held_out_{key}']} -> {family_table}",
            lambda src=paths[f"held_out_{key}"], dst=family_table: filter_rows(
                src, dst, lambda row: row["fault_family"] in {family, "none"}),
            skip_if=family_table,
        )
    runner.run([args.python, ROOT / "scripts/apply_calibration.py", paths["held_out_p3_family"],
                paths["calibrator"], paths["held_out_p3_calibrated"], "--allow-protected-after-freeze"],
               skip_if=paths["held_out_p3_calibrated"])
    for key, source in (("p3", paths["held_out_p3_calibrated"]), ("p1", paths["held_out_p1_family"])):
        threshold_path = paths[f"threshold_{key}"]
        target = paths[f"held_out_{key}_alarmed"]

        def action(source=source, target=target, threshold_path=threshold_path):
            threshold = float(yaml.safe_load(threshold_path.read_text(encoding="utf-8"))["threshold"])
            apply_fold_policy(source, target, threshold=threshold, alarm=alarm)

        runner.internal(
            f"apply fold threshold from {threshold_path} with frozen persistence/cooldown: "
            f"{source} -> {target}", action, skip_if=target,
        )


def evaluate_folds(
    families: list[str], work_root: Path, *, bootstrap: dict, allow_protected: bool,
    gate_passed: bool, engineering_fixture: bool,
) -> dict[str, object]:
    per_family: dict[str, object] = {}
    inputs: dict[str, str] = {}
    for family in families:
        paths = fold_paths(work_root, family)
        tables = {}
        for key in ("p3", "p1"):
            path = paths[f"held_out_{key}_alarmed"]
            if not path.exists():
                per_family[family] = {"status": "missing_fold_tables", "missing": str(path)}
                break
            rows = load_table(path)
            protected = {truth(row["protected_test_used"]) for row in rows}
            if not engineering_fixture:
                enforce_protected_boundary(
                    True if protected == {True} else (False if protected == {False} else None),
                    explicitly_allowed=allow_protected, confirmatory_gate_passed=gate_passed,
                )
                if {row["split"] for row in rows} != {HELD_OUT_SPLIT}:
                    raise ValueError(f"{path}: fold tables must contain held-out episodes only")
            inputs[str(path)] = sha256_file(path)
            tables[key] = rows
        else:
            outcomes = {key: episode_outcomes(rows) for key, rows in tables.items()}
            if set(outcomes["p3"]) != set(outcomes["p1"]):
                raise ValueError(f"{family}: P3 and P1 fold tables cover different episodes")
            family_rows = [row for row in outcomes["p3"].values() if row["fault_family"] == family]
            events = [
                {**row, "detected_a": row["detected"], "detected_b": outcomes["p1"][row["run_id"]]["detected"]}
                for row in family_rows if row["has_event"] and not row["is_timeout"]
            ]
            thresholds = {}
            for key in ("p3", "p1"):
                threshold_path = paths[f"threshold_{key}"]
                if threshold_path.exists():
                    document = yaml.safe_load(threshold_path.read_text(encoding="utf-8")) or {}
                    thresholds[key] = document.get("threshold")
                    inputs[str(threshold_path)] = sha256_file(threshold_path)
            clean = {
                key: summarize([row for row in value.values() if row["is_clean"]])
                for key, value in outcomes.items()
            }
            entry: dict[str, object] = {
                "status": "evaluated",
                "family_episode_count": len(family_rows),
                "family_event_count_full_denominator": len(events),
                "fold_threshold": thresholds,
                "p3": summarize([row for row in family_rows if not row["is_timeout"]]),
                "p1": summarize([
                    row for row in outcomes["p1"].values()
                    if row["fault_family"] == family and not row["is_timeout"]
                ]),
                "false_alerts_per_clean_mission_at_fold_threshold": {
                    key: clean[key]["false_alerts_per_clean_mission"] for key in clean
                },
                "timeout_events_excluded": sum(
                    1 for row in family_rows if row["has_event"] and row["is_timeout"]
                ),
            }
            if events:
                difference = hierarchical_bootstrap(events, paired_recall_difference, **bootstrap)
                entry["p3_minus_p1_recall"] = difference
                entry["p3_exceeds_p1"] = bool(difference["point_estimate"] > 0.0)
                entry["interval_excludes_zero"] = bool(
                    difference["confidence_interval"][0] is not None
                    and difference["confidence_interval"][0] > 0.0
                )
            else:
                entry["p3_exceeds_p1"] = None
                entry["note"] = "no primary events for this family on the held-out maps"
            per_family[family] = entry
    evaluated = [f for f in families if per_family.get(f, {}).get("status") == "evaluated"]
    exceeding = [f for f in evaluated if per_family[f].get("p3_exceeds_p1") is True]
    differences = [
        per_family[f]["p3_minus_p1_recall"]["point_estimate"]
        for f in evaluated if "p3_minus_p1_recall" in per_family[f]
    ]
    complete = len(evaluated) == len(families) and not engineering_fixture
    return {
        "schema_version": 1,
        "kind": "unseen_family_folds",
        "complete": complete,
        "research_evidence": not engineering_fixture,
        "engineering_fixture": engineering_fixture,
        "protected_test_used": True,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "analysis_unit": "episode",
        "families": families,
        "families_evaluated": evaluated,
        "inputs_sha256": inputs,
        "per_family": per_family,
        "h5": {
            "label": "supporting",
            "hypothesis": (
                f"leave-one-family-out P3 recall exceeds the P1 baseline for at least "
                f"{H5_MINIMUM_FAMILIES} of {len(families)} families at each fold's own budget threshold"
            ),
            "families_where_p3_exceeds_p1": exceeding,
            "count": len(exceeding),
            "minimum_required": H5_MINIMUM_FAMILIES,
            "supported": bool(len(exceeding) >= H5_MINIMUM_FAMILIES) if complete else None,
            "pooled_claim": "none; per-family estimates only",
            "heterogeneity": {
                "per_family_differences": dict(zip(
                    [f for f in evaluated if "p3_minus_p1_recall" in per_family[f]], differences
                )),
                "range": [min(differences), max(differences)] if differences else None,
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("fit", "evaluate"), required=True)
    parser.add_argument("--family", action="append", dest="families",
                        help="restrict to given families (default: all seven folds)")
    parser.add_argument("--splits", type=Path, default=ROOT / "data/manifests/splits.template.yaml")
    parser.add_argument("--train-dataset", action="append", default=None)
    parser.add_argument("--selection-dataset", default=DEFAULT_SELECTION_DATASET)
    parser.add_argument("--held-out-dataset", default=DEFAULT_HELD_OUT_DATASET)
    parser.add_argument("--p3-config", type=Path, default=ROOT / "configs/models/p3_causal_tcn.yaml")
    parser.add_argument("--python", default=str(ROOT / ".venv/bin/python"))
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--work-root", type=Path, default=ROOT / "reports/unseen_family")
    parser.add_argument("--model-root", type=Path, default=ROOT / "models/unseen_family")
    parser.add_argument("--alarm", type=Path, default=ROOT / "configs/alarm_policy.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/confirmatory/unseen_family.yaml")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260903)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--assemble-only", action="store_true",
                        help="evaluate stage: read existing fold tables, run no commands")
    parser.add_argument("--allow-protected-after-freeze", action="store_true")
    parser.add_argument("--engineering-fixture", action="store_true",
                        help="synthetic fold tables only; output marked non-research evidence")
    args = parser.parse_args(argv)
    args.train_dataset = args.train_dataset or list(DEFAULT_TRAIN_DATASETS)

    splits = yaml.safe_load(args.splits.read_text(encoding="utf-8")) or {}
    families = args.families or list(splits.get("unseen_family_folds") or [])
    if len(families) != 7 and not args.families:
        raise SystemExit("split manifest must declare exactly seven unseen-family folds")
    alarm = yaml.safe_load(args.alarm.read_text(encoding="utf-8")) or {}
    runner = Runner(dry_run=args.dry_run)

    if args.stage == "fit":
        for family in families:
            fit_family(family, args, runner)
        print(json.dumps({"stage": "fit", "families": families, "dry_run": args.dry_run,
                          "commands": len(runner.commands), "protected_test_used": False}, indent=2))
        return 0

    gate_passed = False
    if not args.engineering_fixture and not args.dry_run:
        gate = subprocess.run(
            [sys.executable, str(ROOT / "scripts/check_readiness.py"), "--stage", "confirmatory"],
            check=False, capture_output=True, text=True,
        )
        gate_passed = gate.returncode == 0
        if not gate_passed or not args.allow_protected_after_freeze:
            print(gate.stdout)
            raise SystemExit(
                "unseen-family evaluation refused before the model freeze: pass "
                "--allow-protected-after-freeze and satisfy check_readiness --stage confirmatory"
            )
    if not args.assemble_only:
        for family in families:
            evaluate_family_commands(family, args, runner, alarm)
    if args.dry_run:
        print(json.dumps({"stage": "evaluate", "families": families, "dry_run": True,
                          "commands": len(runner.commands), "output": str(args.output)}, indent=2))
        return 0
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable report {args.output}")
    try:
        report = evaluate_folds(
            families, args.work_root,
            bootstrap={"replicates": args.replicates, "seed": args.bootstrap_seed},
            allow_protected=args.allow_protected_after_freeze, gate_passed=gate_passed,
            engineering_fixture=args.engineering_fixture,
        )
    except ValueError as error:
        raise SystemExit(f"unseen-family evaluation refused: {error}") from error
    report["fold_commands"] = runner.commands
    publish_new_bytes(args.output, yaml.safe_dump(report, sort_keys=False).encode("utf-8"))
    print(f"wrote {args.output}: H5 count {report['h5']['count']} (complete={report['complete']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
