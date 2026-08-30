#!/usr/bin/env python3
"""Regenerate a synthetic engineering-only end-to-end metric artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import AlarmPolicy, apply_alarm_policy, evaluate_event_warnings
from src.features import FeatureSpec, LeakagePolicy, ScalarSample, extract_decision_rows
from src.models import Rule, ThresholdRuleSet


def episode(run_id: str, event_time: float | None, risky: bool) -> list[dict]:
    decisions = [float(value) for value in range(5, 31)]
    samples = [
        ScalarSample(time, 0.3 if risky and 20 <= time <= 25 else 0.95)
        for time in decisions
    ]
    features = extract_decision_rows(
        run_id=run_id, decision_times=decisions,
        specs=[FeatureSpec("valid_return_fraction", "/scan", 1.0)],
        samples_by_feature={"valid_return_fraction": samples},
        leakage_policy=LeakagePolicy.from_yaml(ROOT / "configs/leakage_denylist.yaml"),
    )
    rules = ThresholdRuleSet([Rule("scan_dropout", "valid_return_fraction", "below", 0.5)])
    rows = []
    for row in features:
        decision = float(row["decision_time"])
        if event_time is None:
            eligibility = "eligible_negative"
        elif event_time - 10 <= decision <= event_time - 1:
            eligibility = "eligible_positive"
        elif decision > event_time - 1:
            eligibility = "excluded_too_late"
        else:
            eligibility = "eligible_negative"
        rows.append({
            **row, **rules.predict(row), "eligibility": eligibility,
            "primary_event_time": event_time,
        })
    return apply_alarm_policy(rows, AlarmPolicy(0.5, 2, 3, 10.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reports/engineering/offline_pipeline_smoke.json",
    )
    args = parser.parse_args()
    episodes = {
        "synthetic-failure": episode("synthetic-failure", 30.0, True),
        "synthetic-clean": episode("synthetic-clean", None, False),
    }
    payload = {
        "artifact_role": "engineering_fixture_not_research_evidence",
        "synthetic": True,
        "protected_data_used": False,
        "metrics": evaluate_event_warnings(episodes),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote synthetic engineering smoke to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
