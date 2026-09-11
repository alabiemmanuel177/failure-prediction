#!/usr/bin/env python3
"""Regenerate the protocol's minimum figures from immutable artifacts (never fabricate).

Each figure is written once (refuses to overwrite) with a ``<figure>.json`` sidecar
recording every source artifact and its sha256. Figures whose inputs are absent are
listed under PENDING in ``reports/figures/figure_index.json`` and on stdout.
Artifact locations default to ``src.reporting.ARTIFACT_DEFAULTS`` and may be
overridden with ``--input name=path``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable



ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reporting import (  # noqa: E402
    ArtifactRegistry, lead_time_curve_by_family, predictor_metrics,
    recall_false_alert_curve, reliability_by_score, split_models,
)
from src.reporting.inputs import publish_text, write_sidecar  # noqa: E402


FIGURES = (
    "fig01_architecture_causal_timeline",
    "fig02_risk_traces",
    "fig03_recall_vs_false_alerts",
    "fig04_lead_time_by_family",
    "fig05_reliability_diagrams",
    "fig06_generalisation_matrix",
    "fig07_feature_group_ablation",
    "fig08_recovery_outcomes",
    "fig09_action_confusion",
)


class Pending(Exception):
    """Raised when a figure's inputs are missing; recorded, never fabricated."""


class Partial(Exception):
    """Some panels were produced; the rest stay pending (never fabricated)."""

    def __init__(self, paths: list[Path], reason: str) -> None:
        super().__init__(reason)
        self.paths = paths


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _primary_model(registry: ArtifactRegistry, models: dict[str, list]) -> str:
    if registry.available("model_freeze"):
        frozen = registry.yaml("model_freeze").get("predictor", {}).get("model_id")
        if frozen in models:
            return str(frozen)
    return "p3_causal_tcn" if "p3_causal_tcn" in models else sorted(models)[0]


def _require(registry: ArtifactRegistry, *names: str) -> None:
    missing = [name for name in names if not registry.available(name)]
    if missing:
        raise Pending(", ".join(f"{name} ({registry.path(name)})" for name in missing))


def _save(plt, figure, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite figure: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)


def fig01(registry: ArtifactRegistry, out: Path) -> list[Path]:
    _require(registry, "failure_events", "recovery_guards", "alarm_policy")
    windowing = registry.yaml("failure_events")["windowing"]
    guards = registry.yaml("recovery_guards")
    alarm = registry.yaml("alarm_policy")
    persistence = alarm.get("persistence", {})
    history = float(windowing["history_seconds"])
    horizon = float(windowing["warning_horizon_seconds"])
    guard = float(windowing["too_late_guard_seconds"])
    width, height = 1000, 420
    left, right = 60, 940
    scale = (right - left) / (history + horizon + 4.0)
    x0 = left + history * scale  # decision time origin
    tf = x0 + (horizon + 2.0) * scale
    stages = ["telemetry", "causal features\n(value/age/missing)", "predictor P3",
              "calibration", f"alarm policy\n({persistence.get('required_above_threshold', 2)}-of-"
              f"{persistence.get('decisions_considered', 3)}, {alarm.get('cooldown_seconds', 10):g} s cooldown)",
              "independent guard", "recovery action"]
    box_w = (right - left) / len(stages) - 10
    parts = [f'<rect width="{width}" height="{height}" fill="white"/>',
             '<text x="60" y="30" font-family="sans-serif" font-size="18" font-weight="600">'
             'Architecture and causal labelling timeline</text>']
    for index, label in enumerate(stages):
        bx = left + index * (box_w + 10)
        parts.append(f'<rect x="{bx}" y="55" width="{box_w}" height="60" rx="6" fill="#eef3fb" stroke="#2455a4"/>')
        for line_index, line in enumerate(label.split("\n")):
            parts.append(f'<text x="{bx + box_w / 2:.1f}" y="{80 + 16 * line_index}" text-anchor="middle" '
                         f'font-family="sans-serif" font-size="11">{html.escape(line)}</text>')
        if index < len(stages) - 1:
            parts.append(f'<line x1="{bx + box_w}" y1="85" x2="{bx + box_w + 10}" y2="85" stroke="#2455a4" stroke-width="2"/>')
    y = 260
    parts.append(f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" stroke="#222" stroke-width="2"/>')
    parts.append(f'<rect x="{x0 - history * scale:.1f}" y="{y - 60}" width="{history * scale:.1f}" height="50" fill="#2455a4" opacity="0.25"/>')
    parts.append(f'<text x="{x0 - history * scale / 2:.1f}" y="{y - 70}" text-anchor="middle" font-family="sans-serif" font-size="12">past-only history {history:g} s</text>')
    parts.append(f'<line x1="{x0:.1f}" y1="{y - 65}" x2="{x0:.1f}" y2="{y + 30}" stroke="#222" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{x0:.1f}" y="{y + 48}" text-anchor="middle" font-family="sans-serif" font-size="12">decision time t</text>')
    parts.append(f'<rect x="{tf - horizon * scale:.1f}" y="{y + 5}" width="{(horizon - guard) * scale:.1f}" height="22" fill="#3a9d5d" opacity="0.35"/>')
    parts.append(f'<text x="{tf - horizon * scale / 2:.1f}" y="{y + 20}" text-anchor="middle" font-family="sans-serif" font-size="11">positive: t in [t_f - {horizon:g} s, t_f - {guard:g} s]</text>')
    parts.append(f'<rect x="{tf - guard * scale:.1f}" y="{y + 5}" width="{guard * scale:.1f}" height="22" fill="#b3261e" opacity="0.45"/>')
    parts.append(f'<line x1="{tf:.1f}" y1="{y - 40}" x2="{tf:.1f}" y2="{y + 30}" stroke="#b3261e" stroke-width="3"/>')
    parts.append(f'<text x="{tf:.1f}" y="{y - 48}" text-anchor="middle" font-family="sans-serif" font-size="12">first terminal event t_f</text>')
    parts.append(f'<text x="{tf:.1f}" y="{y + 48}" text-anchor="middle" font-family="sans-serif" font-size="11">too-late guard {guard:g} s</text>')
    parts.append(f'<text x="{left}" y="{y + 90}" font-family="sans-serif" font-size="12">Eligible negatives lie at least {windowing.get("negative_guard_seconds", 20):g} s from every event and injection onset. '
                 f'Guard: rear clearance ≥ {guards["minimum_rear_clearance_m"]} m, rotation clearance ≥ {guards["minimum_rotation_clearance_m"]} m, '
                 f'at most {guards["maximum_repeated_recoveries"]} repeated recoveries.</text>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
           + "".join(parts) + "</svg>\n")
    path = out / "fig01_architecture_causal_timeline.svg"
    publish_text(path, svg)
    return [path]


def fig02(registry: ArtifactRegistry, out: Path) -> list[Path]:
    _require(registry, "predictions_held_out")
    rows = registry.prediction_rows("predictions_held_out")
    models = split_models(rows)
    model = _primary_model(registry, models)
    metrics = predictor_metrics(models[model])
    picks = {"useful_warning": None, "false_alarm": None, "missed_failure": None}
    for item in metrics["per_episode"]:
        if item["has_event"] and item["detected"] and picks["useful_warning"] is None:
            picks["useful_warning"] = item["run_id"]
        elif not item["has_event"] and item["false_alerts"] > 0 and picks["false_alarm"] is None:
            picks["false_alarm"] = item["run_id"]
        elif item["has_event"] and not item["detected"] and picks["missed_failure"] is None:
            picks["missed_failure"] = item["run_id"]
    missing = [name for name, run_id in picks.items() if run_id is None]
    if len(missing) == len(picks):
        raise Pending(f"no held-out episode of any trace kind for {model}")
    # the trace plotter reads one model at a time: write a model-filtered temporary table
    filtered = out / f".{model}_held_out_rows.csv"
    import csv
    with filtered.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(models[model])
    paths = []
    try:
        for kind, run_id in picks.items():
            if run_id is None:
                continue
            path = out / f"fig02_risk_trace_{kind}.svg"
            subprocess.run([sys.executable, str(ROOT / "scripts/plot_warning_trace.py"),
                            str(filtered), run_id, str(path)], check=True)
            paths.append(path)
    finally:
        filtered.unlink(missing_ok=True)
    if missing:
        raise Partial(paths, f"no held-out episode of kind {missing} for {model}")
    return paths


def fig03(registry: ArtifactRegistry, out: Path) -> list[Path]:
    _require(registry, "predictions_held_out", "alarm_policy")
    rows = registry.prediction_rows("predictions_held_out")
    policy = registry.yaml("alarm_policy")
    plt = _plt()
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    thresholds = [index / 40 for index in range(41)]
    for model, model_rows in split_models(rows).items():
        curve = recall_false_alert_curve(model_rows, policy, thresholds)
        xs = [point["false_alerts_per_clean_mission"] or 0.0 for point in curve]
        ys = [point["event_recall"] or 0.0 for point in curve]
        axis.plot(xs, ys, marker=".", label=model)
        frozen = predictor_metrics(model_rows)
        axis.scatter([frozen["false_alerts_per_clean_mission"] or 0.0], [frozen["event_recall"] or 0.0],
                     marker="*", s=140, zorder=5)
    budget = float(policy.get("false_alert_budget_per_clean_mission", 0.10))
    axis.axvline(budget, color="#b3261e", linestyle="--", label=f"budget {budget:g}/clean mission")
    axis.set_xlabel("false alerts per clean mission (held-out maps)")
    axis.set_ylabel("event recall")
    axis.set_ylim(0, 1.02)
    axis.legend(fontsize=8)
    axis.set_title("Event recall versus false-alert burden (star = frozen threshold)")
    path = out / "fig03_recall_vs_false_alerts.svg"
    _save(plt, figure, path)
    return [path]


def fig04(registry: ArtifactRegistry, out: Path) -> list[Path]:
    _require(registry, "predictions_held_out")
    rows = registry.prediction_rows("predictions_held_out")
    models = split_models(rows)
    model = _primary_model(registry, models)
    grid = [index * 0.5 for index in range(21)]
    curves = lead_time_curve_by_family(models[model], grid)
    if not curves:
        raise Pending(f"no held-out events for {model}")
    plt = _plt()
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    for family, curve in curves.items():
        axis.step(curve["grid_seconds"], curve["warned_fraction"], where="post",
                  label=f"{family} (n={curve['event_count']}, missed={curve['undetected']})")
    axis.set_xlabel("seconds before the terminal event")
    axis.set_ylabel("fraction of events already warned")
    axis.set_ylim(0, 1.02)
    axis.set_title(f"Lead-time curve by failure family — {model}")
    axis.legend(fontsize=7)
    path = out / "fig04_lead_time_by_family.svg"
    _save(plt, figure, path)
    return [path]


def fig05(registry: ArtifactRegistry, out: Path) -> list[Path]:
    _require(registry, "predictions_validation", "predictions_held_out")
    plt = _plt()
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for axis, name in zip(axes, ("predictions_validation", "predictions_held_out")):
        rows = registry.prediction_rows(name)
        models = split_models(rows)
        model = _primary_model(registry, models)
        for label, score in (("before calibration (raw)", "raw_score"), ("after calibration", "risk_score")):
            curve = reliability_by_score(models[model], score)
            points = [(b["mean_predicted_risk"], b["observed_event_frequency"]) for b in curve["bins"] if b["count"]]
            axis.plot([p[0] for p in points], [p[1] for p in points], marker="o",
                      label=f"{label}: ECE={curve['ece']:.3f}, Brier={curve['brier_score']:.3f}")
        axis.plot([0, 1], [0, 1], color="#888", linestyle=":")
        axis.set_title(name.replace("predictions_", "").replace("_", " ") + f" — {model}")
        axis.set_xlabel("mean predicted risk")
        axis.legend(fontsize=7)
    axes[0].set_ylabel("observed event frequency")
    path = out / "fig05_reliability_diagrams.svg"
    _save(plt, figure, path)
    return [path]


def fig06(registry: ArtifactRegistry, out: Path) -> list[Path]:
    _require(registry, "predictions_held_out", "predictions_unseen_family")
    held = split_models(registry.prediction_rows("predictions_held_out"))
    unseen_rows = registry.prediction_rows("predictions_unseen_family")
    if "fold_family" in unseen_rows[0]:
        unseen_rows = [row for row in unseen_rows if row["fold_family"] == row["fault_family"]]
    unseen = split_models(unseen_rows)
    model = _primary_model(registry, held)
    families = sorted({row["fault_family"] for row in unseen_rows} | {
        family for rows in held.values() for family in predictor_metrics(rows)["by_family"] if family != "none"
    })
    columns = [f"held-out {name}" for name in held] + [f"unseen-family {model}"]
    matrix = []
    for family in families:
        line = []
        for name, rows in held.items():
            line.append(predictor_metrics(rows)["by_family"].get(family, {}).get("event_recall"))
        line.append(predictor_metrics(unseen.get(model, [])).get("by_family", {}).get(family, {}).get("event_recall")
                    if unseen.get(model) else None)
        matrix.append(line)
    plt = _plt()
    figure, axis = plt.subplots(figsize=(1.6 * len(columns) + 2, 0.5 * len(families) + 1.5))
    values = [[float("nan") if v is None else v for v in line] for line in matrix]
    image = axis.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(columns)), columns, rotation=30, ha="right", fontsize=8)
    axis.set_yticks(range(len(families)), families, fontsize=8)
    for i, line in enumerate(matrix):
        for j, value in enumerate(line):
            axis.text(j, i, "n/a" if value is None else f"{value:.2f}", ha="center", va="center", fontsize=8,
                      color="white" if (value or 0) < 0.6 else "black")
    figure.colorbar(image, ax=axis, label="event recall at frozen threshold")
    axis.set_title("Held-out-map and unseen-family event recall")
    path = out / "fig06_generalisation_matrix.svg"
    _save(plt, figure, path)
    return [path]


def fig07(registry: ArtifactRegistry, out: Path) -> list[Path]:
    _require(registry, "predictions_ablation")
    rows = registry.prediction_rows("predictions_ablation")
    if "ablation" not in rows[0]:
        raise Pending("ablation prediction table lacks an 'ablation' column")
    groups: dict[str, list] = {}
    for row in rows:
        groups.setdefault(str(row["ablation"]), []).append(row)
    names = sorted(groups)
    recalls, burdens = [], []
    for name in names:
        metrics = predictor_metrics(groups[name])
        recalls.append(metrics["event_recall"] or 0.0)
        burdens.append(metrics["false_alerts_per_clean_mission"] or 0.0)
    plt = _plt()
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].barh(names, recalls, color="#2455a4"); axes[0].set_xlim(0, 1); axes[0].set_xlabel("event recall")
    axes[1].barh(names, burdens, color="#e28a00"); axes[1].set_xlabel("false alerts per clean mission")
    figure.suptitle("Feature-group and policy ablations (identical held-out episodes, frozen threshold rule)")
    path = out / "fig07_feature_group_ablation.svg"
    _save(plt, figure, path)
    return [path]


def fig08(registry: ArtifactRegistry, out: Path) -> list[Path]:
    _require(registry, "paired_recovery")
    report = registry.yaml("paired_recovery")
    comparisons = {k: v for k, v in report.get("comparisons", {}).items() if v.get("status") == "computed"}
    if not comparisons:
        raise Pending("paired recovery report has no computed comparison")
    policies: dict[str, dict] = {}
    for item in comparisons.values():
        policies[item["baseline_policy"]] = item["baseline"]
        policies[item["proposed_policy"]] = item["proposed"]
    names = sorted(policies)
    plt = _plt()
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].bar(names, [policies[n]["mission_completion_rate"] for n in names], color="#3a9d5d")
    axes[0].set_ylim(0, 1); axes[0].set_title("mission completion rate")
    axes[1].bar(names, [policies[n]["collision_rate"] for n in names], color="#b3261e")
    axes[1].set_title("collision rate")
    axes[2].bar(names, [policies[n]["median_added_time_seconds"] for n in names], color="#e28a00")
    axes[2].set_title("median added time (s)")
    primary = comparisons.get("R3_vs_R0")
    if primary:
        ci = primary["completion_difference_bootstrap"]["confidence_interval"]
        figure.suptitle(f"Paired recovery outcomes — R3−R0 completion Δ={primary['paired_completion_rate_difference']:.3f} "
                        f"[{ci[0]:.3f}, {ci[1]:.3f}], pairs={primary['pair_count']}")
    path = out / "fig08_recovery_outcomes.svg"
    _save(plt, figure, path)
    return [path]


def fig09(registry: ArtifactRegistry, out: Path) -> list[Path]:
    _require(registry, "paired_recovery")
    report = registry.yaml("paired_recovery")
    confusion = report.get("action_confusion_vs_oracle", {})
    if "R3" not in confusion:
        raise Pending("paired recovery report lacks an oracle action confusion for R3")
    cells = {tuple(key.split("->")): count for key, count in confusion["R3"].items()}
    actions = sorted({a for pair in cells for a in pair})
    matrix = [[cells.get((oracle, chosen), 0) for chosen in actions] for oracle in actions]
    plt = _plt()
    figure, axis = plt.subplots(figsize=(0.9 * len(actions) + 2, 0.7 * len(actions) + 1.5))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(actions)), actions, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(range(len(actions)), actions, fontsize=8)
    axis.set_xlabel("R3 executed action"); axis.set_ylabel("oracle eligible action")
    for i, line in enumerate(matrix):
        for j, value in enumerate(line):
            axis.text(j, i, str(value), ha="center", va="center", fontsize=8)
    axis.set_title("Recovery action confusion versus oracle (R3)")
    path = out / "fig09_action_confusion.svg"
    _save(plt, figure, path)
    return [path]


BUILDERS: dict[str, Callable[[ArtifactRegistry, Path], list[Path]]] = {
    "fig01_architecture_causal_timeline": fig01,
    "fig02_risk_traces": fig02,
    "fig03_recall_vs_false_alerts": fig03,
    "fig04_lead_time_by_family": fig04,
    "fig05_reliability_diagrams": fig05,
    "fig06_generalisation_matrix": fig06,
    "fig07_feature_group_ablation": fig07,
    "fig08_recovery_outcomes": fig08,
    "fig09_action_confusion": fig09,
}


def build_all(registry: ArtifactRegistry, out: Path, only: list[str] | None = None) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    index: dict[str, Any] = {"generated_utc": generated_utc, "produced": {}, "pending": {}}
    for name in FIGURES:
        if only and name not in only:
            continue
        registry.used.clear()
        try:
            paths = BUILDERS[name](registry, out)
        except Pending as pending:
            index["pending"][name] = str(pending)
            continue
        except Partial as partial:
            paths = partial.paths
            index["pending"][f"{name} (partial)"] = str(partial)
        sources = registry.sources(list(registry.used))
        for path in paths:
            write_sidecar(path, {"figure": name, "file": path.name, "generated_utc": generated_utc,
                                 "sources": sources, "fabricated_inputs": False})
        index["produced"][name] = [str(path.relative_to(out)) for path in paths]
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/figures")
    parser.add_argument("--input", action="append", default=[], help="override name=path")
    parser.add_argument("--only", action="append", default=[], choices=FIGURES)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    registry = ArtifactRegistry(args.root, ArtifactRegistry.parse_overrides(args.input))
    index = build_all(registry, args.output_dir, args.only or None)
    index_path = args.output_dir / "figure_index.json"
    publish_text(index_path, json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"produced: {sorted(index['produced'])}")
    print("PENDING: " + (json.dumps(index["pending"], indent=2) if index["pending"] else "none"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
