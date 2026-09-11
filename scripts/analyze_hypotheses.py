#!/usr/bin/env python3
"""Assemble H1-H5 (and H6 when the recovery report exists) into one hypothesis table.

Reads the confirmatory reports (``held_out_map.yaml``, ``unseen_family.yaml``, optional
``natural_failure_audit.yaml`` and ``reports/recovery/paired_recovery.yaml``) and writes
the immutable ``reports/confirmatory/hypotheses.yaml`` and ``hypotheses.md``. H1 (and
H6) are labelled confirmatory, H2-H5 supporting; every other item is flagged
exploratory. The output inherits ``research_evidence: false`` if any input is an
engineering fixture.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_inventory import publish_new_bytes


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return document if isinstance(document, dict) else None


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_interval(interval: object) -> str:
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        return "n/a"
    return f"[{fmt(interval[0])}, {fmt(interval[1])}]"


def status_of(supported: object, complete: bool) -> str:
    if not complete:
        return "pending"
    if supported is None:
        return "not_evaluable"
    return "supported" if supported else "not_supported"


def assemble(
    held_out: dict | None, unseen: dict | None, natural: dict | None, recovery: dict | None,
    held_out_label: str = "reports/confirmatory/held_out_map.yaml",
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    fixture = any(
        document is not None and document.get("engineering_fixture") is True
        for document in (held_out, unseen, natural, recovery)
    )
    held_complete = bool(held_out and held_out.get("complete") is True)
    unseen_complete = bool(unseen and unseen.get("complete") is True)

    h1 = (held_out or {}).get("h1", {})
    rows.append({
        "id": "H1", "label": "confirmatory",
        "hypothesis": "P3 event recall exceeds P1 at the validation-fixed false-alert budget",
        "estimate": h1.get("point_estimate"),
        "estimate_name": "paired P3 - P1 event recall difference",
        "confidence_interval": h1.get("confidence_interval"),
        "denominator": h1.get("denominator_events"),
        "status": status_of(h1.get("supported"), held_complete and "supported" in h1),
        "source": f"{held_out_label}:h1",
    })
    h2 = (held_out or {}).get("h2", {})
    rows.append({
        "id": "H2", "label": "supporting",
        "hypothesis": "median useful lead time of detected failures >= 3 s",
        "estimate": h2.get("median_lead_seconds"),
        "estimate_name": "median lead seconds (detected events)",
        "confidence_interval": h2.get("confidence_interval"),
        "denominator": h2.get("denominator_events"),
        "undetected_events": h2.get("undetected_events"),
        "status": status_of(
            h2.get("point_estimate_meets_minimum"), held_complete and "point_estimate_meets_minimum" in h2
        ),
        "interval_lower_bound_meets_minimum": h2.get("interval_lower_bound_meets_minimum"),
        "source": f"{held_out_label}:h2",
    })
    h3 = (held_out or {}).get("h3", {})
    validation = h3.get("validation", {}) if isinstance(h3.get("validation"), dict) else {}
    rows.append({
        "id": "H3", "label": "supporting",
        "hypothesis": "calibration reduces Brier score and ECE on validation data",
        "estimate": validation.get("brier_delta_after_minus_before"),
        "estimate_name": "validation Brier delta (after - before)",
        "ece_delta": validation.get("ece_delta_after_minus_before"),
        "held_out_brier_delta": (h3.get("held_out") or {}).get("brier_delta_after_minus_before"),
        "held_out_ece_delta": (h3.get("held_out") or {}).get("ece_delta_after_minus_before"),
        "confidence_interval": None,
        "status": status_of(h3.get("supported"), held_complete and "supported" in h3),
        "source": f"{held_out_label}:h3",
    })
    h4 = (held_out or {}).get("h4", {})
    groups = h4.get("h4_groups") or {}
    deltas = {name: entry.get("recall_delta_vs_primary") for name, entry in groups.items()}
    rows.append({
        "id": "H4", "label": "supporting",
        "hypothesis": "removing planner and localisation health features materially reduces early warning",
        "estimate": min((d for d in deltas.values() if d is not None), default=None),
        "estimate_name": "largest recall decline among planner/localisation ablations",
        "ablation_deltas": deltas,
        "confidence_interval": None,
        "status": (
            "reported_no_numeric_threshold" if held_complete and groups else "pending"
        ),
        "source": f"{held_out_label}:h4 (reports/ablations)",
    })
    h5 = (unseen or {}).get("h5", {})
    rows.append({
        "id": "H5", "label": "supporting",
        "hypothesis": "leave-one-family-out P3 recall exceeds P1 for at least five of seven families",
        "estimate": h5.get("count"),
        "estimate_name": "families where P3 > P1 (of seven)",
        "families": h5.get("families_where_p3_exceeds_p1"),
        "confidence_interval": None,
        "pooled_claim": "none",
        "heterogeneity": h5.get("heterogeneity"),
        "status": status_of(h5.get("supported"), unseen_complete),
        "source": "reports/confirmatory/unseen_family.yaml:h5",
    })
    recovery_complete = bool(recovery and recovery.get("complete") is True)
    h6 = (recovery or {}).get("h6", {}) if recovery else {}
    rows.append({
        "id": "H6", "label": "confirmatory",
        "hypothesis": "prediction-triggered recovery improves mission completion without more collisions",
        "estimate": h6.get("point_estimate"),
        "estimate_name": "paired completion difference (R3 - R0)",
        "confidence_interval": h6.get("confidence_interval"),
        "status": status_of(h6.get("supported"), recovery_complete and "supported" in h6),
        "source": "reports/recovery/paired_recovery.yaml:h6 (owned by the recovery work package)",
    })

    exploratory: list[dict[str, object]] = []
    for key, entry in ((held_out or {}).get("exploratory_models") or {}).items():
        exploratory.append({
            "item": f"{key} versus P1 event recall", "label": "exploratory",
            "estimate": entry.get("point_estimate"),
            "confidence_interval": entry.get("confidence_interval"),
            "source": f"{held_out_label}:exploratory_models",
        })
    for model_key, model in ((held_out or {}).get("models") or {}).items():
        by_severity = model.get("by_severity") or {}
        if len(by_severity) > 1:
            exploratory.append({
                "item": f"{model_key} recall by severity", "label": "exploratory",
                "estimate": {sev: value.get("event_recall") for sev, value in by_severity.items()},
                "source": f"{held_out_label}:models",
            })
    timeouts = (held_out or {}).get("timeouts") or {}
    for model_key, value in timeouts.items():
        if isinstance(value, dict):
            exploratory.append({
                "item": f"{model_key} mission-timeout recall (analysed separately)",
                "label": "exploratory",
                "estimate": value.get("timeout_recall"),
                "denominator": value.get("timeout_episode_count"),
                "source": f"{held_out_label}:timeouts",
            })
    if natural:
        for entry in natural.get("warning_performance") or []:
            performance = entry.get("natural_failures_excluding_timeouts") or {}
            exploratory.append({
                "item": f"{entry.get('model_id')} recall on natural (no-injection) failures",
                "label": "exploratory", "never_merged_with_injected_results": True,
                "estimate": performance.get("event_recall"),
                "denominator": performance.get("event_count"),
                "source": "reports/confirmatory/natural_failure_audit.yaml",
            })
        if not natural.get("warning_performance"):
            exploratory.append({
                "item": "natural (no-injection) failure inventory", "label": "exploratory",
                "estimate": (natural.get("totals") or {}).get("natural_failures"),
                "source": "reports/confirmatory/natural_failure_audit.yaml",
            })
    return {
        "schema_version": 1,
        "kind": "hypothesis_table",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "research_evidence": not fixture,
        "engineering_fixture": fixture,
        "protected_test_used": bool(held_out or unseen),
        "confirmatory_claims": ["H1", "H6"],
        "supporting_hypotheses": ["H2", "H3", "H4", "H5"],
        "multiplicity_note": (
            "H1 and H6 are the two confirmatory claims; H2-H5 are prespecified supporting "
            "hypotheses; every other item is exploratory and not an independent significance claim"
        ),
        "frozen_threshold": (held_out or {}).get("frozen_threshold"),
        "model_freeze_sha256": (held_out or {}).get("model_freeze_sha256"),
        "hypotheses": rows,
        "exploratory": exploratory,
    }


def render_markdown(report: dict[str, object]) -> str:
    lines = ["# Hypothesis table", ""]
    if not report["research_evidence"]:
        lines.append("**Engineering fixture: not research evidence.**")
        lines.append("")
    lines.append(f"Frozen threshold: {fmt(report.get('frozen_threshold'))}; "
                 f"model freeze sha256: {fmt(report.get('model_freeze_sha256'))}.")
    lines.append("")
    lines.append("| ID | Label | Hypothesis | Estimate | 95% CI | Status | Source |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in report["hypotheses"]:
        estimate = row.get("estimate")
        estimate_text = fmt(estimate) if not isinstance(estimate, dict) else str(estimate)
        lines.append(
            f"| {row['id']} | {row['label']} | {row['hypothesis']} | "
            f"{estimate_text} ({row.get('estimate_name', '')}) | "
            f"{fmt_interval(row.get('confidence_interval'))} | {row['status']} | {row['source']} |"
        )
    lines.append("")
    lines.append(report["multiplicity_note"])
    lines.append("")
    if report["exploratory"]:
        lines.append("## Exploratory items (flagged; not claims)")
        lines.append("")
        lines.append("| Item | Estimate | 95% CI | Source |")
        lines.append("|---|---|---|---|")
        for item in report["exploratory"]:
            estimate = item.get("estimate")
            estimate_text = fmt(estimate) if not isinstance(estimate, dict) else str(estimate)
            lines.append(
                f"| {item['item']} | {estimate_text} | {fmt_interval(item.get('confidence_interval'))} "
                f"| {item['source']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-out", type=Path, default=ROOT / "reports/confirmatory/held_out_map.yaml")
    parser.add_argument("--unseen-family", type=Path, default=ROOT / "reports/confirmatory/unseen_family.yaml")
    parser.add_argument("--natural-failures", type=Path,
                        default=ROOT / "reports/confirmatory/natural_failure_audit.yaml")
    parser.add_argument("--recovery", type=Path, default=ROOT / "reports/recovery/paired_recovery.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/confirmatory")
    args = parser.parse_args(argv)
    held_out, unseen = load(args.held_out), load(args.unseen_family)
    if held_out is None and unseen is None:
        raise SystemExit("no confirmatory reports found; run evaluate_confirmatory and the unseen-family folds first")
    held_out_label = str(args.held_out.resolve().relative_to(ROOT)) if args.held_out.resolve().is_relative_to(ROOT) else str(args.held_out)
    report = assemble(held_out, unseen, load(args.natural_failures), load(args.recovery), held_out_label=held_out_label)
    report["inputs_sha256"] = {
        str(path): sha256_file(path)
        for path in (args.held_out, args.unseen_family, args.natural_failures, args.recovery)
        if path.exists()
    }
    yaml_path = args.output_dir / "hypotheses.yaml"
    md_path = args.output_dir / "hypotheses.md"
    for path in (yaml_path, md_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite immutable report {path}")
    publish_new_bytes(yaml_path, yaml.safe_dump(report, sort_keys=False).encode("utf-8"))
    publish_new_bytes(md_path, render_markdown(report).encode("utf-8"))
    print(f"wrote {yaml_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
