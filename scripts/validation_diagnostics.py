#!/usr/bin/env python3
"""Exploratory, validation-only diagnostics of the confirmatory result (no protected data).

Three questions the researcher asked after the held-out evaluation, answered only from
the validation split (324 episodes) and the three fitted P3 seeds:

  D1  precursor structure per fault family: where the 10 s warning window lies relative
      to the injection interval, how many warnable decisions each event offers, and
      whether the calibrated risk score rises inside the window at all
  D2  stability of the budgeted threshold selection: hierarchical bootstrap of the
      validation routes/episodes, re-running the frozen selection rule, and the
      false-alert rate each resampled threshold would give on the full validation set
  D3  selection optimism: leave-one-validation-map-out; seed, Platt calibrator and
      threshold selected on two maps, evaluated on the third

Outputs reports/exploratory/validation_diagnostics_v1.{yaml,md}. Flagged exploratory;
no preregistered quantity is changed. Requires no protected artefact.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
import random
import statistics
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.calibrators import apply_calibrator_rows, fit_calibrator  # noqa: E402
from src.evaluation.event_metrics import evaluate_event_warnings  # noqa: E402
from src.evaluation.policy import AlarmPolicy, apply_alarm_policy, select_validation_threshold  # noqa: E402
from src.evaluation.prediction_tables import clean_run_ids, group_episodes, read_prediction_table  # noqa: E402

DATASET = ROOT / "data/derived/balanced_validation_v1-validation-324"
SEEDS = {"final_v1": 20260903, "final_v1_seed20260904": 20260904, "final_v1_seed20260905": 20260905}
PRIMARY_TAG = "final_v1_seed20260904"
BUDGET = 0.10
HORIZON, GUARD = 10.0, 1.0


def med(values):
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 3) if values else None


def frac(flags):
    flags = list(flags)
    return round(sum(1 for f in flags if f) / len(flags), 3) if flags else None


def metrics_at(episodes, clean, threshold):
    policy = AlarmPolicy(threshold, 2, 3, 10.0)
    alarmed = {run: apply_alarm_policy(rows, policy) for run, rows in episodes.items()}
    m = evaluate_event_warnings(alarmed)
    clean_false = sum(e["false_alerts"] for e in m["per_episode"] if e["run_id"] in clean)
    return {"event_recall": m["event_recall"], "false_alerts_per_clean_mission": clean_false / len(clean) if clean else None,
            "event_count": m["event_count"], "detected": m["detected_event_count"], "clean_missions": len(clean)}


# ---------------------------------------------------------------- D1 precursors
def d1_precursors(threshold: float) -> dict:
    rows = read_prediction_table(ROOT / f"reports/predictions/{PRIMARY_TAG}/p3_causal_tcn.validation.calibrated.csv")
    alarmed_rows = read_prediction_table(ROOT / f"reports/predictions/{PRIMARY_TAG}/p3_causal_tcn.validation.alarmed.csv")
    alarms = defaultdict(list)
    for r in alarmed_rows:
        if str(r.get("alarm")).lower() == "true":
            alarms[r["run_id"]].append(float(r["decision_time"]))
    episodes = group_episodes(rows)
    per_family = defaultdict(list)
    for run_id, ep in episodes.items():
        first = ep[0]
        if first.get("primary_event_time") in (None, "", "None"):
            continue
        event = float(first["primary_event_time"])
        ann = yaml.safe_load((DATASET / "annotations" / f"{run_id}.yaml").read_text(encoding="utf-8"))
        inj = [i for i in ann.get("injections", []) if i.get("eligible", True)]
        onset = float(inj[0]["actual_onset"]) if inj and inj[0].get("actual_onset") is not None else None
        end = onset + float(inj[0].get("duration_seconds") or 0.0) if onset is not None else None
        w0, w1 = event - HORIZON, event - GUARD
        window = [r for r in ep if w0 <= float(r["decision_time"]) <= w1]
        eligible_pos = [r for r in window if r["eligibility"] == "eligible_positive"]
        baseline = [r for r in ep if onset is not None and float(r["decision_time"]) < onset]
        window_max = max((float(r["risk_score"]) for r in window), default=None)
        baseline_max = max((float(r["risk_score"]) for r in baseline), default=None)
        overlap = None
        if onset is not None:
            overlap = max(0.0, min(w1, end) - max(w0, onset))
        per_family[first["fault_family"]].append({
            "run_id": run_id, "event_class": first.get("primary_event_class"), "event_time": event,
            "onset": onset, "injection_end": end,
            "seconds_from_injection_end_to_window_start": None if end is None else round(w0 - end, 2),
            "window_overlap_with_injection_s": None if overlap is None else round(overlap, 2),
            "decisions_in_window": len(window), "eligible_positive_in_window": len(eligible_pos),
            "window_max_risk": window_max, "baseline_max_risk": baseline_max,
            "single_decision_exceeds_threshold": window_max is not None and window_max >= threshold,
            "policy_detected": any(w0 <= t <= w1 for t in alarms.get(run_id, [])),
        })
    summary = {}
    for family, items in sorted(per_family.items()):
        summary[family] = {
            "events": len(items),
            "event_classes": dict(sorted(((c, sum(1 for i in items if i["event_class"] == c)) for c in {i["event_class"] for i in items}))),
            "median_seconds_injection_end_to_window_start": med(i["seconds_from_injection_end_to_window_start"] for i in items),
            "fraction_window_overlaps_injection": frac(i["window_overlap_with_injection_s"] and i["window_overlap_with_injection_s"] > 0 for i in items if i["window_overlap_with_injection_s"] is not None),
            "median_eligible_positive_decisions": med(i["eligible_positive_in_window"] for i in items),
            "median_window_max_risk": med(i["window_max_risk"] for i in items),
            "median_baseline_max_risk": med(i["baseline_max_risk"] for i in items),
            "fraction_window_max_above_frozen_threshold": frac(i["single_decision_exceeds_threshold"] for i in items),
            "fraction_detected_by_frozen_policy": frac(i["policy_detected"] for i in items),
        }
    return {"summary": summary, "episodes": {f: items for f, items in per_family.items()}}


# ---------------------------------------------------------------- D2 threshold stability
_EPISODES = None
_CLEAN = None


def _init(episodes, clean):
    global _EPISODES, _CLEAN
    _EPISODES, _CLEAN = episodes, clean


def _resample(seed: int):
    rng = random.Random(seed)
    by_map = defaultdict(lambda: defaultdict(list))
    for run, rows in _EPISODES.items():
        by_map[rows[0]["map_id"]][rows[0]["route_id"]].append(run)
    sample = {}
    k = 0
    for map_id, routes in by_map.items():
        names = list(routes)
        for route in rng.choices(names, k=len(names)):
            runs = routes[route]
            for run in rng.choices(runs, k=len(runs)):
                sample[f"{run}#{k}"] = [{**r, "run_id": f"{run}#{k}"} for r in _EPISODES[run]]
                k += 1
    clean = {run for run, rows in sample.items() if rows[0]["fault_family"] == "none"}
    chosen = select_validation_threshold(sample, clean, false_alert_budget=BUDGET, maximum_candidates=201)
    threshold = float(chosen["threshold"])
    on_full = metrics_at(_EPISODES, _CLEAN, threshold)
    frozen_on_sample = metrics_at(sample, clean, FROZEN)
    return {"seed": seed, "threshold": threshold, "in_sample_recall": chosen.get("event_recall"),
            "recall_on_full_validation": on_full["event_recall"], "fa_per_clean_on_full_validation": on_full["false_alerts_per_clean_mission"],
            "frozen_threshold_fa_per_clean_on_sample": frozen_on_sample["false_alerts_per_clean_mission"],
            "frozen_threshold_recall_on_sample": frozen_on_sample["event_recall"]}


FROZEN = None


def d2_threshold_stability(threshold: float, replicates: int, workers: int) -> dict:
    global FROZEN
    FROZEN = threshold
    rows = read_prediction_table(ROOT / f"reports/predictions/{PRIMARY_TAG}/p3_causal_tcn.validation.calibrated.csv")
    episodes = group_episodes(rows)
    clean = clean_run_ids(rows)
    with Pool(workers, initializer=_init, initargs=(episodes, clean)) as pool:
        results = pool.map(_resample, range(1, replicates + 1))
    def q(values, p):
        s = sorted(values); return round(s[min(len(s) - 1, int(p * len(s)))], 4)
    thresholds = [r["threshold"] for r in results]
    fa_full = [r["fa_per_clean_on_full_validation"] for r in results]
    rec_full = [r["recall_on_full_validation"] for r in results]
    fa_frozen = [r["frozen_threshold_fa_per_clean_on_sample"] for r in results]
    return {
        "replicates": replicates, "resampling": "routes within each map, then episodes within route, with replacement (maps fixed)",
        "candidate_grid": 201, "frozen_threshold": threshold,
        "threshold_quantiles": {"p05": q(thresholds, .05), "p25": q(thresholds, .25), "p50": q(thresholds, .5), "p75": q(thresholds, .75), "p95": q(thresholds, .95)},
        "fraction_replicates_selecting_never_alarm": frac(t >= 1.0 for t in thresholds),
        "resampled_threshold_applied_to_full_validation": {
            "fa_per_clean_quantiles": {"p05": q(fa_full, .05), "p50": q(fa_full, .5), "p95": q(fa_full, .95)},
            "fraction_exceeding_budget": frac(f > BUDGET for f in fa_full),
            "recall_quantiles": {"p05": q(rec_full, .05), "p50": q(rec_full, .5), "p95": q(rec_full, .95)},
        },
        "frozen_threshold_on_resamples": {
            "fa_per_clean_quantiles": {"p05": q(fa_frozen, .05), "p50": q(fa_frozen, .5), "p95": q(fa_frozen, .95)},
            "fraction_exceeding_budget": frac(f > BUDGET for f in fa_frozen),
        },
        "replicate_rows": results,
    }


# ---------------------------------------------------------------- D3 selection optimism
def d3_leave_one_map_out() -> dict:
    raw = {tag: read_prediction_table(ROOT / f"reports/predictions/{tag}/p3_causal_tcn.validation.raw.csv") for tag in SEEDS}
    maps = sorted({r["map_id"] for r in raw[PRIMARY_TAG]})
    folds = {}
    for held in maps:
        per_seed = {}
        for tag, rows in raw.items():
            pool_rows = [r for r in rows if r["map_id"] != held]
            held_rows = [r for r in rows if r["map_id"] == held]
            calibrator = fit_calibrator("platt_scaling", pool_rows)
            pool_cal = apply_calibrator_rows(pool_rows, calibrator)
            held_cal = apply_calibrator_rows(held_rows, calibrator)
            pool_eps, pool_clean = group_episodes(pool_cal), clean_run_ids(pool_cal)
            chosen = select_validation_threshold(pool_eps, pool_clean, false_alert_budget=BUDGET, maximum_candidates=401)
            threshold = float(chosen["threshold"])
            held_eps, held_clean = group_episodes(held_cal), clean_run_ids(held_cal)
            per_seed[tag] = {
                "seed": SEEDS[tag], "threshold": threshold,
                "selection_maps": {"event_recall": chosen.get("event_recall"), "false_alerts_per_clean_mission": chosen.get("false_alerts_per_clean_mission")},
                "held_out_map": metrics_at(held_eps, held_clean, threshold),
            }
        best = max(per_seed, key=lambda t: (per_seed[t]["selection_maps"]["event_recall"] or 0.0,
                                            -(per_seed[t]["selection_maps"]["false_alerts_per_clean_mission"] or 0.0)))
        folds[held] = {"selected_tag": best, "seeds": per_seed, "selected_on_selection_maps": per_seed[best]["selection_maps"],
                       "selected_on_held_out_map": per_seed[best]["held_out_map"]}
    sel_rec = [f["selected_on_selection_maps"]["event_recall"] for f in folds.values()]
    out_rec = [f["selected_on_held_out_map"]["event_recall"] for f in folds.values()]
    out_fa = [f["selected_on_held_out_map"]["false_alerts_per_clean_mission"] for f in folds.values()]
    return {
        "design": "for each validation map: Platt calibrator, budgeted threshold and seed chosen on the other two maps (seed rule: recall at budget), evaluated on the map left out",
        "seeds": SEEDS, "folds": folds,
        "mean_recall_on_selection_maps": round(statistics.mean(sel_rec), 3),
        "mean_recall_on_left_out_map": round(statistics.mean(out_rec), 3),
        "mean_fa_per_clean_on_left_out_map": round(statistics.mean(out_fa), 3),
        "selection_optimism_recall": round(statistics.mean(sel_rec) - statistics.mean(out_rec), 3),
        "in_sample_frozen_reference": {"event_recall": 0.3387, "false_alerts_per_clean_mission": 0.0972, "note": "primary seed on all three validation maps"},
    }


def render(report: dict) -> str:
    d1, d2, d3 = report["d1_precursors"]["summary"], report["d2_threshold_stability"], report["d3_selection_optimism"]
    out = ["# Validation-only diagnostics (exploratory)", "",
           f"Generated {report['generated_utc']}. Frozen threshold {report['frozen_threshold']:.4f}; no protected data used.", "",
           "## D1 Precursor structure per family (validation events, primary seed)", "",
           "| Family | Events | Median s from injection end to window start | Window overlaps injection | Median warnable decisions | Median window max risk | Median pre-onset max risk | Window max >= threshold | Detected by frozen policy |",
           "|---|---|---|---|---|---|---|---|---|"]
    for fam, s in d1.items():
        out.append(f"| {fam} | {s['events']} | {s['median_seconds_injection_end_to_window_start']} | {s['fraction_window_overlaps_injection']} | {s['median_eligible_positive_decisions']} | {s['median_window_max_risk']} | {s['median_baseline_max_risk']} | {s['fraction_window_max_above_frozen_threshold']} | {s['fraction_detected_by_frozen_policy']} |")
    tq, fa = d2["threshold_quantiles"], d2["resampled_threshold_applied_to_full_validation"]
    out += ["", "## D2 Threshold-selection stability", "",
            f"{d2['replicates']} hierarchical resamples of the validation routes/episodes; threshold re-selected each time with the frozen rule.", "",
            f"- Selected threshold p05/p50/p95: {tq['p05']} / {tq['p50']} / {tq['p95']} (frozen {d2['frozen_threshold']:.4f}); never-alarm chosen in {d2['fraction_replicates_selecting_never_alarm']} of replicates",
            f"- Resampled thresholds applied to the full validation set: false alerts per clean mission p05/p50/p95 = {fa['fa_per_clean_quantiles']['p05']} / {fa['fa_per_clean_quantiles']['p50']} / {fa['fa_per_clean_quantiles']['p95']}; exceed the 0.10 budget in {fa['fraction_exceeding_budget']} of replicates; recall p05/p50/p95 = {fa['recall_quantiles']['p05']} / {fa['recall_quantiles']['p50']} / {fa['recall_quantiles']['p95']}",
            f"- Frozen threshold on resamples: false alerts per clean mission p05/p50/p95 = {d2['frozen_threshold_on_resamples']['fa_per_clean_quantiles']['p05']} / {d2['frozen_threshold_on_resamples']['fa_per_clean_quantiles']['p50']} / {d2['frozen_threshold_on_resamples']['fa_per_clean_quantiles']['p95']}; exceeds budget in {d2['frozen_threshold_on_resamples']['fraction_exceeding_budget']}",
            "", "## D3 Selection optimism (leave one validation map out)", "",
            "| Left-out map | Selected seed | Recall on selection maps | Recall on left-out map | FA/clean on left-out map |", "|---|---|---|---|---|"]
    for m, f in d3["folds"].items():
        out.append(f"| {m} | {f['seeds'][f['selected_tag']]['seed']} | {f['selected_on_selection_maps']['event_recall']:.3f} | {f['selected_on_held_out_map']['event_recall']:.3f} | {f['selected_on_held_out_map']['false_alerts_per_clean_mission']:.3f} |")
    out += ["", f"Mean recall: {d3['mean_recall_on_selection_maps']} on selection maps versus {d3['mean_recall_on_left_out_map']} on the left-out map (optimism {d3['selection_optimism_recall']}); mean FA/clean on left-out maps {d3['mean_fa_per_clean_on_left_out_map']}. In-sample frozen reference: recall 0.339 at 0.097 FA/clean; held-out maps gave 0.229 at 0.183.", ""]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/exploratory/validation_diagnostics_v1.yaml")
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    freeze = yaml.safe_load((ROOT / "configs/model_freeze.yaml").read_text(encoding="utf-8"))
    alarm = yaml.safe_load((ROOT / "configs/alarm_policy.yaml").read_text(encoding="utf-8"))
    threshold = float(alarm["threshold"])
    print("D1 precursors", flush=True)
    d1 = d1_precursors(threshold)
    print("D3 leave-one-map-out", flush=True)
    d3 = d3_leave_one_map_out()
    print("D2 threshold stability", flush=True)
    d2 = d2_threshold_stability(threshold, args.replicates, args.workers)
    report = {
        "schema_version": 1, "kind": "validation_only_diagnostics", "exploratory": True, "protected_test_used": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(), "frozen_threshold": threshold,
        "model_freeze_sha256": freeze.get("checkpoint_sha256") or freeze.get("model", {}).get("checkpoint_sha256"),
        "d1_precursors": d1, "d2_threshold_stability": d2, "d3_selection_optimism": d3,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    args.output.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print("wrote", args.output, "and", args.output.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
