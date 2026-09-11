#!/usr/bin/env python3
"""Compare two validation-only model-development passes model by model.

Used to judge a fitting-pool change (for example adding the outcome-selected Research 1
source) on Research 2-only validation before the freeze. It reads the immutable
`validation_comparison.yaml` of each pass and writes an immutable delta report. It
selects nothing and touches no protected data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import publish_new_bytes, sha256_file  # noqa: E402

METRICS = {
    "event_recall": ("event_recall",),
    "false_alerts_per_clean_mission": ("false_alerts_per_clean_mission",),
    "false_alerts_per_non_event_mission": ("false_alerts_per_non_event_mission",),
    "median_lead_seconds_detected": ("lead_time", "median_seconds_detected"),
    "auprc": ("discrimination", "auprc"),
    "auroc": ("discrimination", "auroc"),
    "brier_score": ("calibration", "brier_score"),
    "ece": ("calibration", "ece"),
    "threshold": ("threshold_selection", "threshold"),
}


def dig(record: dict, path: tuple[str, ...]):
    value = record
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-tag", required=True)
    parser.add_argument("--candidate-tag", required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    paths = {
        tag: ROOT / "reports/model_selection" / tag / "validation_comparison.yaml"
        for tag in (args.baseline_tag, args.candidate_tag)
    }
    documents = {tag: yaml.safe_load(path.read_text(encoding="utf-8")) for tag, path in paths.items()}
    baseline, candidate = documents[args.baseline_tag], documents[args.candidate_tag]
    if baseline.get("protected_test_used") or candidate.get("protected_test_used"):
        raise SystemExit("pass comparison accepts validation-only reports")
    models = sorted(set(baseline["models"]) & set(candidate["models"]))
    per_model = {}
    for model_id in models:
        entry = {}
        for name, path in METRICS.items():
            before = dig(baseline["models"][model_id], path)
            after = dig(candidate["models"][model_id], path)
            entry[name] = {
                "baseline": before, "candidate": after,
                "delta": (after - before) if isinstance(before, (int, float)) and isinstance(after, (int, float)) else None,
            }
        per_model[model_id] = entry
    families = sorted(set().union(*(set(baseline["models"][m].get("by_family", {})) for m in models)))
    by_family = {}
    for model_id in models:
        by_family[model_id] = {
            family: {
                "baseline_recall": dig(baseline["models"][model_id], ("by_family", family, "event_recall")),
                "candidate_recall": dig(candidate["models"][model_id], ("by_family", family, "event_recall")),
            }
            for family in families
        }
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "validation_only_fitting_pool_comparison",
        "protected_test_used": False,
        "frozen": False,
        "baseline": {"tag": args.baseline_tag, "report": str(paths[args.baseline_tag].relative_to(ROOT)),
                     "report_sha256": sha256_file(paths[args.baseline_tag])},
        "candidate": {"tag": args.candidate_tag, "report": str(paths[args.candidate_tag].relative_to(ROOT)),
                      "report_sha256": sha256_file(paths[args.candidate_tag])},
        "per_model": per_model,
        "per_family_event_recall": by_family,
        "interpretation": (
            "Deltas are validation-only rehearsal evidence for a fitting-pool decision; "
            "they are not confirmatory results and select no threshold."
        ),
    }
    output = args.output or (ROOT / "reports/model_selection" /
                             f"pass_comparison_{args.baseline_tag}_vs_{args.candidate_tag}.yaml")
    publish_new_bytes(output, yaml.safe_dump(report, sort_keys=False).encode("utf-8"))
    for model_id, entry in per_model.items():
        print(f"{model_id:24s} recall {entry['event_recall']['baseline']!s:>8} -> {entry['event_recall']['candidate']!s:>8}"
              f"  fa/clean {entry['false_alerts_per_clean_mission']['baseline']!s:>8} -> {entry['false_alerts_per_clean_mission']['candidate']!s:>8}"
              f"  auprc {entry['auprc']['baseline']!s:>8} -> {entry['auprc']['candidate']!s:>8}")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
