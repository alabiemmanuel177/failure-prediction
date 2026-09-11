#!/usr/bin/env python3
"""Sweep persistence and cooldown on validation predictions to check the preregistered policy.

Nothing is selected automatically: the preregistered 2-of-3 + 10 s setting in
configs/alarm_policy.yaml stays unless the researcher amends the protocol.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import sha256_file
from src.evaluation import AlarmPolicy, apply_alarm_policy, select_validation_threshold
from src.evaluation.discrimination import alert_burden_summary, lead_time_summary
from src.evaluation.prediction_tables import (
    clean_run_ids, group_episodes, model_id_of, policy_settings, read_prediction_table,
    require_validation_only, write_yaml_report,
)

PERSISTENCE_GRID = ((1, 1), (2, 3), (3, 3), (2, 4))
COOLDOWN_GRID = (0.0, 5.0, 10.0, 20.0)


def evaluate_setting(
    episodes, clean, *, required_above, decisions_considered, cooldown, budget, decision_rate_hz
) -> dict[str, object]:
    try:
        selection = select_validation_threshold(
            episodes, clean, false_alert_budget=budget, required_above=required_above,
            decisions_considered=decisions_considered, cooldown_seconds=cooldown,
        )
    except ValueError as error:
        return {"feasible": False, "error": str(error)}
    policy = AlarmPolicy(selection["threshold"], required_above, decisions_considered, cooldown)
    alarmed = [row for rows in episodes.values() for row in apply_alarm_policy(rows, policy)]
    metrics = selection["metrics"]
    return {
        "feasible": True,
        "threshold": selection["threshold"],
        "feasible_threshold_count": selection["feasible_threshold_count"],
        "event_recall": metrics["event_recall"],
        "event_count": metrics["event_count"],
        "detected_event_count": metrics["detected_event_count"],
        "false_alerts_per_clean_mission": selection["false_alerts_per_clean_mission"],
        "false_alerts_per_non_event_mission": metrics["false_alerts_per_non_event_mission"],
        "false_alerts_per_mission": metrics["false_alerts_per_mission"],
        "lead_time": lead_time_summary(metrics),
        "alert_burden": alert_burden_summary(alarmed, decision_rate_hz=decision_rate_hz),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="validation prediction table (calibrated)")
    parser.add_argument("--alarm", type=Path, default=ROOT / "configs/alarm_policy.yaml")
    parser.add_argument("--output", type=Path, default=None,
                        help="default reports/alarm_policy/<model_id>.validation_sweep.yaml")
    args = parser.parse_args()
    rows = read_prediction_table(args.predictions)
    try:
        require_validation_only(rows, "alarm-policy validation")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    model_id = model_id_of(rows)
    output = args.output or ROOT / "reports/alarm_policy" / f"{model_id}.validation_sweep.yaml"
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    alarm = yaml.safe_load(args.alarm.read_text(encoding="utf-8"))
    settings = policy_settings(alarm)
    rate = float(alarm.get("decision_rate_hz", 2.0))
    episodes = group_episodes(rows)
    clean = clean_run_ids(rows)
    preregistered = (
        settings["required_above"], settings["decisions_considered"], settings["cooldown_seconds"]
    )
    sweep = []
    for required_above, decisions_considered in PERSISTENCE_GRID:
        for cooldown in COOLDOWN_GRID:
            result = evaluate_setting(
                episodes, clean, required_above=required_above,
                decisions_considered=decisions_considered, cooldown=cooldown,
                budget=settings["false_alert_budget"], decision_rate_hz=rate,
            )
            sweep.append({
                "persistence": f"{required_above}-of-{decisions_considered}",
                "required_above_threshold": required_above,
                "decisions_considered": decisions_considered,
                "cooldown_seconds": cooldown,
                "preregistered": (required_above, decisions_considered, cooldown) == preregistered,
                **result,
            })
    chosen = [entry for entry in sweep if entry["preregistered"]]
    if not chosen:
        chosen = [{
            **evaluate_setting(
                episodes, clean, required_above=preregistered[0],
                decisions_considered=preregistered[1], cooldown=preregistered[2],
                budget=settings["false_alert_budget"], decision_rate_hz=rate,
            ),
            "persistence": f"{preregistered[0]}-of-{preregistered[1]}",
            "cooldown_seconds": preregistered[2], "preregistered": True,
        }]
    feasible = [entry for entry in sweep if entry["feasible"]]
    best_recall = max((entry["event_recall"] or 0.0) for entry in feasible) if feasible else None
    report = {
        "schema_version": 1,
        "protocol_version": alarm.get("protocol_version"),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "scope": "validation_only_policy_check",
        "confirmatory": False,
        "automatic_selection": False,
        "preregistered_setting_retained": True,
        "note": (
            "Persistence and cooldown are swept for transparency only. The preregistered "
            "setting stays in configs/alarm_policy.yaml; any change requires a protocol "
            "amendment and a research-log protocol_change record before the freeze."
        ),
        "protected_test_used": False,
        "inputs": {
            "prediction_table": str(args.predictions),
            "prediction_table_sha256": sha256_file(args.predictions),
            "alarm_policy": str(args.alarm),
            "alarm_policy_sha256": sha256_file(args.alarm),
        },
        "false_alert_budget_per_clean_mission": settings["false_alert_budget"],
        "clean_mission_count": len(clean),
        "episode_count": len(episodes),
        "preregistered_setting": {
            "persistence": f"{preregistered[0]}-of-{preregistered[1]}",
            "cooldown_seconds": preregistered[2],
            "result": chosen[0],
        },
        "best_validation_recall_in_sweep": best_recall,
        "sweep": sweep,
    }
    write_yaml_report(output, report)
    for entry in sweep:
        marker = "*" if entry["preregistered"] else " "
        if entry["feasible"]:
            print(
                f"{marker} {entry['persistence']:>6} cooldown={entry['cooldown_seconds']:>4}s "
                f"tau={entry['threshold']:.4f} recall={entry['event_recall']} "
                f"fa/clean={entry['false_alerts_per_clean_mission']:.3f} "
                f"lead_median={entry['lead_time']['median_seconds_detected']}"
            )
        else:
            print(f"{marker} {entry['persistence']:>6} cooldown={entry['cooldown_seconds']:>4}s infeasible")
    print(f"report -> {output} (preregistered setting retained; no automatic selection)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
