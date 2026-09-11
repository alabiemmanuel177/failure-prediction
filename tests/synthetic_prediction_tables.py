"""Small synthetic prediction tables with hand-computed alarm outcomes (test helper).

Layout (all validation split, 2 Hz decisions t = 0.0 .. 9.5):
- clean C1..C4 (fault_family none): eligible_negative throughout.
- event E1..E4 (primary_event_time 10.0): eligible_negative for t < 5, eligible_positive
  for 5 <= t <= 9, excluded_too_late at 9.5.
- F1 (lidar_dropout, no event): eligible_negative throughout.

Model "p3_causal_tcn": clean 0.1; E1..E3 rise to 0.9 from t = 5 (2-of-3 persistence
fires at t = 5.5, lead 4.5 s); E4 missed. -> tau 0.9, recall 3/4, 0 false alerts.
Model "p1_threshold_rules": 0/1 scores; E1 fires from t = 7 (alarm 7.5, lead 2.5 s);
C1 has a single-decision spike at t = 3 (no persistence); F1 fires t = 2..3 (one
false alert on a non-event mission). -> tau 1.0, recall 1/4, 0 false alerts per clean
mission, 1 false alert over 5 non-event missions.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.evaluation.prediction_tables import PREDICTION_COLUMNS

TIMES = [index * 0.5 for index in range(20)]
EPISODES = [
    # run_id, map, route, family, severity, event_time
    ("C1", "m0", "m0_r0", "none", "none", None),
    ("C2", "m0", "m0_r1", "none", "none", None),
    ("C3", "m1", "m1_r0", "none", "none", None),
    ("C4", "m1", "m1_r1", "none", "none", None),
    ("E1", "m0", "m0_r0", "lidar_dropout", "high", 10.0),
    ("E2", "m0", "m0_r1", "wheel_slip", "medium", 10.0),
    ("E3", "m1", "m1_r0", "lidar_dropout", "high", 10.0),
    ("E4", "m1", "m1_r1", "planner_oscillation", "low", 10.0),
    ("F1", "m1", "m1_r0", "lidar_dropout", "low", None),
]


def eligibility(event_time, time):
    if event_time is None:
        return "eligible_negative"
    if time < 5.0:
        return "eligible_negative"
    if time <= 9.0:
        return "eligible_positive"
    return "excluded_too_late"


def score_p3(run_id, time):
    if run_id in {"E1", "E2", "E3"} and time >= 5.0:
        return 0.9
    return 0.1


def score_p1(run_id, time):
    if run_id == "E1" and time >= 7.0:
        return 1.0
    if run_id == "C1" and time == 3.0:
        return 1.0
    if run_id == "F1" and 2.0 <= time <= 3.0:
        return 1.0
    return 0.0


SCORERS = {"p3_causal_tcn": score_p3, "p1_threshold_rules": score_p1}


def build_rows(model_id: str, *, split: str = "validation", scorer=None) -> list[dict]:
    scorer = scorer or SCORERS[model_id]
    rows = []
    for run_id, map_id, route_id, family, severity, event_time in EPISODES:
        for index, time in enumerate(TIMES):
            state = eligibility(event_time, time)
            score = scorer(run_id, time)
            rows.append({
                "run_id": run_id, "decision_index": index, "decision_time": time,
                "split": split, "map_id": map_id, "route_id": route_id,
                "fault_family": family, "severity": severity, "seed": 1,
                "protected_test_used": "false", "eligibility": state,
                "label": {"eligible_positive": 1, "eligible_negative": 0}.get(state, -1),
                "primary_event_class": "collision" if event_time is not None else "",
                "primary_event_time": "" if event_time is None else event_time,
                "model_id": model_id, "raw_score": score, "risk_score": score,
            })
    return rows


def write_table(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(PREDICTION_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return path
