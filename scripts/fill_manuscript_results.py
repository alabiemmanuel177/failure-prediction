#!/usr/bin/env python3
"""Replace RESULT_PENDING / RELEASE_PENDING blocks in manuscript/main.md from reports.

Generated text is derived only from immutable artifacts: the alarmed held-out
prediction table (H1: P3 versus P1 event recall with the hierarchical paired
bootstrap), the results tables, the paired recovery report (H6) and the independent
rerun record. ``--dry-run`` prints the generated text; a real edit requires the
completion audit to pass with only the manuscript and release findings tolerated.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import hierarchical_paired_binary_bootstrap  # noqa: E402
from src.release import blocking_findings, run_completion_audit, sha256_file  # noqa: E402
from src.reporting import ArtifactRegistry, predictor_metrics, split_models  # noqa: E402


TOLERATED = ("manuscript", "release")
MARKERS = ("RESULT_PENDING", "RELEASE_PENDING")


def _pct(value: Any) -> str:
    return "n/a" if value in (None, "") else f"{100 * float(value):.1f}%"


def h1_statement(registry: ArtifactRegistry) -> str:
    if not registry.available("predictions_held_out"):
        return "RESULT_PENDING"  # keeps the marker so the audit still fails
    rows = registry.prediction_rows("predictions_held_out")
    models = split_models(rows)
    if "p3_causal_tcn" not in models or "p1_threshold_rules" not in models:
        return "RESULT_PENDING"
    detected: dict[str, dict[str, Any]] = {}
    for model in ("p3_causal_tcn", "p1_threshold_rules"):
        for item in predictor_metrics(models[model])["per_episode"]:
            if item["has_event"]:
                detected.setdefault(item["run_id"], {})[model] = item["detected"]
    meta = {row["run_id"]: row for row in rows}
    records = [{"map_id": meta[run_id]["map_id"], "route_id": meta[run_id]["route_id"], "episode_id": run_id,
                "p3": flags.get("p3_causal_tcn", False), "p1": flags.get("p1_threshold_rules", False)}
               for run_id, flags in detected.items()]
    if not records:
        return "RESULT_PENDING"
    boot = hierarchical_paired_binary_bootstrap(records, "p3", "p1", replicates=2000)
    p3, p1 = predictor_metrics(models["p3_causal_tcn"]), predictor_metrics(models["p1_threshold_rules"])
    ci = boot["confidence_interval"]
    return (f"On {boot['map_count']} held-out maps ({p3['event_count']} terminal events, "
            f"{p3['episode_count']} episodes) the causal TCN reached event recall {_pct(p3['event_recall'])} "
            f"against {_pct(p1['event_recall'])} for the threshold rules at the frozen validation threshold "
            f"({p3['false_alerts_per_clean_mission']:.3f} false alerts per clean mission), a paired difference of "
            f"{100 * boot['point_estimate']:+.1f} percentage points (95% hierarchical bootstrap interval "
            f"{100 * ci[0]:+.1f} to {100 * ci[1]:+.1f}); median useful lead time "
            f"{p3['median_useful_lead_seconds_detected']} s over {p3['lead_time_denominator_detected']} detected "
            f"events with {p3['undetected_event_count']} undetected.")


def h6_statement(registry: ArtifactRegistry) -> str:
    if not registry.available("paired_recovery"):
        return "RESULT_PENDING"
    report = registry.yaml("paired_recovery")
    h6 = report.get("h6", {})
    if not report.get("complete") or h6.get("supported") is None:
        return "RESULT_PENDING"
    primary = report["comparisons"]["R3_vs_R0"]
    ci = h6["completion_interval_95"]
    verdict = "supported" if h6["supported"] else "not supported"
    return (f"In the paired recovery campaign ({primary['pair_count']} map/route/seed/fault pairs), mission completion "
            f"under the cost-sensitive guarded policy R3 was {_pct(primary['proposed']['mission_completion_rate'])} versus "
            f"{_pct(primary['baseline']['mission_completion_rate'])} under default Nav2 recovery R0 "
            f"(difference {100 * h6['completion_rate_difference']:+.1f} points, 95% interval {100 * ci[0]:+.1f} to "
            f"{100 * ci[1]:+.1f}; collision-rate difference {100 * h6['collision_rate_difference']:+.1f} points; "
            f"estimator {h6['estimator']}). Guard rejections: {report['guard']['R3']['guard_rejection_count']}; "
            f"guard violations: {report['guard']['R3']['guard_violation_count']}. Hypothesis H6 is {verdict}.")


def unseen_statement(root: Path) -> str:
    path = root / "reports/tables/tab03_unseen_family.csv"
    if not path.exists():
        return "RESULT_PENDING"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["model_id"] == "p3_causal_tcn"]
    if not rows:
        return "RESULT_PENDING"
    parts = [f"{row['excluded_family']} {_pct(row['event_recall'])} ({row['events']} events)" for row in rows]
    return ("Leave-one-family-out recall for the TCN was " + "; ".join(parts)
            + ". No pooled unseen-family claim is made; heterogeneity across folds is reported as observed.")


def release_statement(root: Path) -> str:
    path = root / "reports/reproduction/independent_rerun.yaml"
    if not path.exists():
        return "RELEASE_PENDING"
    record = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if record.get("passed") is not True:
        return "RELEASE_PENDING"
    return (f"An independent clean rerun (commit {str(record.get('git_commit', ''))[:12]}, "
            f"{record.get('finished_utc', '')}) regenerated the predictions from the frozen checkpoint and "
            f"reproduced the released tables within tolerance: "
            + ", ".join(f"{name} {item['status']}" for name, item in record.get("diffs", {}).items()) + ".")


def generate(root: Path, registry: ArtifactRegistry) -> dict[str, str]:
    h1, h6, unseen, release = h1_statement(registry), h6_statement(registry), unseen_statement(root), release_statement(root)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return {
        "abstract": f"{h1} {h6}",
        "results": (f"Results were generated on {generated} from immutable artifacts (see reports/tables and "
                    f"reports/figures with their sha256 sidecars).\n\n{h1}\n\n{unseen}\n\n{h6}"),
        "discussion": ("Interpretation is restricted to the observed failure families, maps, robot platform and "
                       "warning horizon. " + ("The confirmatory claims H1 and H6 are reported exactly as estimated above, "
                       "including negative components, without protected-set tuning."
                       if "RESULT_PENDING" not in h1 + h6 else "RESULT_PENDING")),
        "release": release,
    }


def apply(text: str, generated: dict[str, str]) -> str:
    replacements = [
        (r"\*\*RESULT_PENDING:\*\*.*?only from immutable generated artifacts\.", generated["abstract"]),
        (r"## 4\. Results\n\n\*\*RESULT_PENDING\.\*\*.*?(?=\n## 5\. Discussion)",
         "## 4. Results\n\n" + generated["results"] + "\n\nRequired figures are the architecture/causal timeline, "
         "annotated risk traces, recall versus false-alert burden, lead-time curves, reliability diagrams, held-out "
         "and unseen-family matrices, feature ablations, recovery outcomes and recovery-action errors."),
        (r"## 5\. Discussion\n\n\*\*RESULT_PENDING\.\*\*.*?(?=\n## 6\.)", "## 5. Discussion\n\n" + generated["discussion"]),
        (r"\*\*RELEASE_PENDING:\*\*.*?not yet been performed\.", generated["release"]),
    ]
    output = text
    for pattern, replacement in replacements:
        output, count = re.subn(pattern, lambda _m, r=replacement: r, output, count=1, flags=re.S)
        if count != 1:
            raise ValueError(f"manuscript block not found for pattern: {pattern[:40]}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--input", action="append", default=[], help="override artifact name=path")
    args = parser.parse_args()
    root = args.root
    manuscript = root / "manuscript/main.md"
    text = manuscript.read_text(encoding="utf-8")
    if not any(marker in text for marker in MARKERS):
        raise SystemExit("manuscript has no pending markers; nothing to fill")
    registry = ArtifactRegistry(root, ArtifactRegistry.parse_overrides(args.input))
    generated = generate(root, registry)
    if args.dry_run:
        for block, value in generated.items():
            print(f"===== {block} =====\n{value}\n")
        return 0
    if any(marker in value for value in generated.values() for marker in MARKERS):
        raise SystemExit("generated text still contains pending markers; required reports are missing")
    blocking = blocking_findings(run_completion_audit(root), TOLERATED)
    if blocking:
        raise SystemExit("manuscript untouched; completion audit findings:\n- " + "\n- ".join(blocking))
    filled = apply(text, generated)
    if any(marker in filled for marker in MARKERS):
        raise SystemExit("a pending marker survived filling; manuscript untouched")
    before = sha256_file(manuscript)
    manuscript.write_text(filled, encoding="utf-8")
    print(f"filled {manuscript} (previous sha256 {before})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
