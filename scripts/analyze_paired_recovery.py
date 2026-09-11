#!/usr/bin/env python3
"""Analyse the paired closed-loop recovery campaign (confirmatory hypothesis H6).

Input: one CSV row per executed episode with the columns
  map_id, route_id, seed, fault_family, severity, policy_id, run_id, mission_complete,
  collision, guard_violation, guard_rejected, added_time_seconds, added_path_length_m,
  intervention_count, recovery_action, action_regret_vs_oracle [, oracle_action]
Output: immutable ``reports/recovery/paired_recovery.yaml`` whose ``complete`` flag is
true only when every manifest pair is present under every policy and the safety gate
(zero guard violations) passes. H6 uses a mixed-effects logistic model when
statsmodels fits it, and always reports the prespecified hierarchical paired
bootstrap; the fallback is recorded explicitly.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
from statistics import mean, median
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import publish_new_bytes, sha256_file  # noqa: E402
from src.evaluation import evaluate_paired_recovery, hierarchical_paired_binary_bootstrap  # noqa: E402
from src.evaluation.recovery_metrics import PAIR_FIELDS  # noqa: E402
from src.protected_data import enforce_protected_boundary  # noqa: E402
from scripts.build_recovery_campaign_manifest import expand_paired_recovery  # noqa: E402


REQUIRED_COLUMNS = (
    *PAIR_FIELDS, "policy_id", "mission_complete", "collision", "guard_violation",
    "guard_rejected", "added_time_seconds", "added_path_length_m", "intervention_count",
    "recovery_action", "action_regret_vs_oracle",
)
POLICIES = ("R0", "R1", "R2", "R3")
COMPARISONS = (("R0", "R3"), ("R2", "R3"))


def _truth(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def read_outcomes(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"outcome table lacks columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError("outcome table is empty")
    unknown = {row["policy_id"] for row in rows} - set(POLICIES)
    if unknown:
        raise ValueError(f"unknown recovery policies: {sorted(unknown)}")
    return rows


def paired_records(rows: list[dict[str, str]], baseline: str, proposed: str) -> list[dict[str, Any]]:
    by_key: dict[tuple, dict[str, dict[str, str]]] = {}
    for row in rows:
        if row["policy_id"] in {baseline, proposed}:
            key = tuple(str(row[field]) for field in PAIR_FIELDS)
            by_key.setdefault(key, {})[row["policy_id"]] = row
    records = []
    for key, pair in sorted(by_key.items()):
        if set(pair) != {baseline, proposed}:
            continue
        records.append({
            "map_id": key[0], "route_id": key[1],
            "episode_id": f"{key[2]}:{key[3]}:{key[4]}",
            "proposed_complete": _truth(pair[proposed]["mission_complete"]),
            "baseline_complete": _truth(pair[baseline]["mission_complete"]),
            "proposed_collision": _truth(pair[proposed]["collision"]),
            "baseline_collision": _truth(pair[baseline]["collision"]),
        })
    return records


def mixed_effects_logistic(rows: list[dict[str, str]], baseline: str, proposed: str) -> dict[str, Any]:
    """Mission completion ~ policy with map and route random intercepts (statsmodels)."""
    try:
        import pandas as pd
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except Exception as error:  # statsmodels or pandas absent
        return {"method": "fallback_hierarchical_bootstrap", "reason": f"import failed: {error}"}
    frame = pd.DataFrame([{
        "complete": int(_truth(row["mission_complete"])),
        "proposed": int(row["policy_id"] == proposed),
        "map_id": row["map_id"], "route_id": row["route_id"],
    } for row in rows if row["policy_id"] in {baseline, proposed}])
    if frame["complete"].nunique() < 2 or frame["map_id"].nunique() < 2:
        return {"method": "fallback_hierarchical_bootstrap",
                "reason": "outcome or map variation insufficient for a mixed model"}
    try:
        model = BinomialBayesMixedGLM.from_formula(
            "complete ~ proposed", {"map": "0 + C(map_id)", "route": "0 + C(route_id)"}, frame,
        )
        result = model.fit_vb()
        names = list(result.model.exog_names)
        index = names.index("proposed")
        estimate = float(result.fe_mean[index])
        sd = float(result.fe_sd[index])
        if not (math.isfinite(estimate) and math.isfinite(sd)):
            raise ValueError("non-finite fixed effect")
    except Exception as error:
        return {"method": "fallback_hierarchical_bootstrap", "reason": f"mixed model failed: {error}"}
    return {
        "method": "mixed_effects_logistic_variational_bayes_statsmodels",
        "fixed_effect": f"{proposed}_vs_{baseline}",
        "policy_log_odds": estimate,
        "policy_log_odds_sd": sd,
        "policy_log_odds_interval_95": [estimate - 1.96 * sd, estimate + 1.96 * sd],
        "random_intercepts": ["map_id", "route_id"],
        "episode_rows": int(len(frame)),
    }


def analyse(
    rows: list[dict[str, str]], expected: list[dict[str, Any]] | None, *,
    replicates: int, seed: int, collision_margin: float,
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for baseline, proposed in COMPARISONS:
        try:
            summary = evaluate_paired_recovery(rows, baseline, proposed)
        except ValueError as error:
            comparisons[f"{proposed}_vs_{baseline}"] = {"status": "unavailable", "reason": str(error)}
            continue
        records = paired_records(rows, baseline, proposed)
        completion = hierarchical_paired_binary_bootstrap(
            records, "proposed_complete", "baseline_complete", replicates=replicates, seed=seed,
        )
        collision = hierarchical_paired_binary_bootstrap(
            records, "proposed_collision", "baseline_collision", replicates=replicates, seed=seed,
        )
        comparisons[f"{proposed}_vs_{baseline}"] = {
            "status": "computed", **summary,
            "completion_difference_bootstrap": completion,
            "collision_difference_bootstrap": collision,
            "mixed_effects_model": mixed_effects_logistic(rows, baseline, proposed),
        }
    primary = comparisons.get("R3_vs_R0", {})
    h6: dict[str, Any] = {"hypothesis": (
        "prediction-triggered recovery (R3) improves mission completion over default Nav2 "
        "recovery (R0) without increasing collision rate"
    ), "decision_rule": {
        "completion": "hierarchical paired bootstrap 95% interval lower bound > 0",
        "collision": f"paired collision-rate difference point estimate <= {collision_margin}",
        "safety": "zero guard violations under every predictor policy",
    }}
    if primary.get("status") == "computed":
        completion_ci = primary["completion_difference_bootstrap"]["confidence_interval"]
        collision_delta = primary["paired_collision_rate_difference"]
        h6.update({
            "estimator": primary["mixed_effects_model"]["method"],
            "completion_rate_difference": primary["paired_completion_rate_difference"],
            "completion_interval_95": completion_ci,
            "collision_rate_difference": collision_delta,
            "collision_interval_95": primary["collision_difference_bootstrap"]["confidence_interval"],
            "supported": bool(completion_ci[0] > 0 and collision_delta <= collision_margin
                              and primary["safety_gate_passed"]),
        })
    else:
        h6.update({"supported": None, "reason": primary.get("reason", "R3/R0 pairs unavailable")})

    by_policy: dict[str, list[dict[str, str]]] = {policy: [] for policy in POLICIES}
    for row in rows:
        by_policy[row["policy_id"]].append(row)
    guard = {
        policy: {
            "episode_count": len(items),
            "guard_violation_count": sum(_truth(item["guard_violation"]) for item in items),
            "guard_rejection_count": sum(_truth(item["guard_rejected"]) for item in items),
        } for policy, items in by_policy.items()
    }
    safety_gate_passed = bool(rows) and all(
        entry["guard_violation_count"] == 0 for policy, entry in guard.items() if policy != "R0"
    )
    regret = {}
    for policy, items in by_policy.items():
        values = []
        for item in items:
            text = str(item.get("action_regret_vs_oracle", "")).strip()
            if text and text.lower() not in {"none", "nan", "null"}:
                values.append(float(text))
        regret[policy] = {
            "episodes_with_regret": len(values),
            "mean_regret": mean(values) if values else None,
            "median_regret": median(values) if values else None,
            "zero_regret_fraction": (sum(1 for v in values if v == 0.0) / len(values)) if values else None,
        }
    has_oracle = all("oracle_action" in row for row in rows)
    confusion: dict[str, Any]
    if has_oracle:
        confusion = {}
        for policy in ("R1", "R2", "R3"):
            counts = Counter(
                (str(item["oracle_action"]), str(item["recovery_action"])) for item in by_policy[policy]
            )
            confusion[policy] = {
                f"{oracle}->{chosen}": count for (oracle, chosen), count in sorted(counts.items())
            }
    else:
        confusion = {"status": "unavailable", "reason": "oracle_action column absent"}

    observed = Counter((*(str(row[field]) for field in PAIR_FIELDS), row["policy_id"]) for row in rows)
    duplicates = sorted(key for key, count in observed.items() if count > 1)
    completeness: dict[str, Any] = {"duplicate_cells": [list(key) for key in duplicates]}
    if expected is None:
        completeness.update({"manifest": None, "expected_cells": None, "missing_cells": None,
                             "unexpected_cells": None,
                             "reason": "no manifest supplied; completeness cannot be established"})
        all_present = False
    else:
        expected_keys = {
            (item["map"], item["route"], str(item["seed"]), item["family"], item["severity"],
             item["recovery_policy_id"]) for item in expected
        }
        observed_keys = set(observed)
        missing = sorted(expected_keys - observed_keys)
        unexpected = sorted(observed_keys - expected_keys)
        completeness.update({
            "expected_cells": len(expected_keys), "observed_cells": len(observed_keys),
            "missing_cells": [list(key) for key in missing[:200]],
            "missing_cell_count": len(missing),
            "unexpected_cells": [list(key) for key in unexpected[:200]],
            "unexpected_cell_count": len(unexpected),
        })
        all_present = not missing and not unexpected and not duplicates
    complete = bool(all_present and safety_gate_passed
                    and all(item.get("status") == "computed" for item in comparisons.values()))
    return {
        "comparisons": comparisons,
        "h6": h6,
        "guard": guard,
        "safety_gate_passed": safety_gate_passed,
        "regret_vs_oracle": regret,
        "action_confusion_vs_oracle": confusion,
        "completeness": completeness,
        "complete": complete,
        "incomplete_reasons": [reason for reason, failed in (
            ("pairs missing, unexpected or duplicated", not all_present),
            ("guard violation observed", not safety_gate_passed),
            ("a prespecified comparison is unavailable", any(
                item.get("status") != "computed" for item in comparisons.values())),
        ) if failed],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outcomes", type=Path)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/paired_recovery_v1.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/recovery/paired_recovery.yaml")
    parser.add_argument("--freeze", type=Path, default=ROOT / "configs/model_freeze.yaml")
    parser.add_argument("--allow-protected-after-freeze", action="store_true")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--collision-noninferiority-margin", type=float, default=0.0)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable report: {args.output}")
    freeze = yaml.safe_load(args.freeze.read_text(encoding="utf-8")) if args.freeze.exists() else {}
    frozen = bool(freeze and freeze.get("frozen") is True)
    try:
        enforce_protected_boundary(
            True, explicitly_allowed=args.allow_protected_after_freeze,
            confirmatory_gate_passed=frozen,
        )
    except ValueError as error:
        raise SystemExit(f"paired recovery analysis blocked: {error}")
    rows = read_outcomes(args.outcomes)
    expected = None
    inputs = {"outcomes": str(args.outcomes), "outcomes_sha256": sha256_file(args.outcomes),
              "model_freeze": str(args.freeze), "model_freeze_sha256": sha256_file(args.freeze)}
    if args.manifest.exists():
        manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
        expected = expand_paired_recovery(manifest)
        inputs.update({"manifest": str(args.manifest), "manifest_sha256": sha256_file(args.manifest)})
    analysis = analyse(
        rows, expected, replicates=args.replicates, seed=args.seed,
        collision_margin=args.collision_noninferiority_margin,
    )
    report = {
        "schema_version": 1,
        "evidence_type": "paired_closed_loop_recovery_confirmatory",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protected_test_used": True,
        "analysis_unit": "paired_map_route_seed_fault_episode",
        "policies": list(POLICIES),
        "episode_count": len(rows),
        "inputs": inputs,
        **analysis,
    }
    publish_new_bytes(args.output, yaml.safe_dump(report, sort_keys=False).encode("utf-8"))
    print(f"wrote {args.output}: complete={report['complete']} h6_supported={report['h6'].get('supported')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
