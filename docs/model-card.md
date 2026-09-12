# Model card: Research 2 early-warning predictor

Status: final (2026-09-12T07:54:53Z, release commit 4e6d2d3a45a1).

## Intended use

Estimate the probability that the first terminal mobile-navigation event will occur
within ten seconds, using only telemetry available at the current decision time. The
score feeds a separately frozen alarm policy and an independently guarded recovery
system. It is not a safety certificate or a replacement for Nav2 collision checking.

## Required model record

- Predictor ID, architecture ID, checkpoint hash and source commit.
- Feature-schema, split-manifest and normalisation hashes.
- Training episodes, class/event weighting and random seeds.
- Window length, horizon, optimizer, loss, stopping rule and parameter count.
- Calibration method/parameters and validation-only threshold record.
- Median, p95 and maximum feature-plus-inference latency.

## Required evaluation

- Event recall at the frozen false-alert budget with complete denominators.
- False alerts per clean and all non-event missions; alert burden and lead time.
- AUPRC/AUROC as supporting window metrics only.
- Brier score, ECE and reliability curves before and after calibration.
- Held-out maps, seven unseen-family folds and mandatory feature/policy ablations.
- Paired recovery completion, collision, overhead, guard rejection and regret results.

## Prohibited evidence

No injected-fault commands, parameters, onset timestamps, simulator ground truth,
terminal action results, recovery results, future samples or protected-set tuning may
enter fitting, calibration or deployable features. Random window splitting and window
accuracy cannot support the primary claim.

## Limitations to update after evaluation

Populate missed-event and false-alarm failure modes, family/map heterogeneity,
unforecastable horizons, calibration drift, missing-data sensitivity, latency limits and
simulation-to-real constraints. Negative results remain part of the card.

## Final record (generated 2026-09-12T07:54:53Z)

- Release commit: `4e6d2d3a45a1c93a8ce33bb984080fbf49d393e0`.
- Predictor: `p3_causal_tcn`; checkpoint `models/final_v1_seed20260904/p3_causal_tcn__seed20260904/checkpoint.pt` (sha256 `171451f0fa21036a05a5b9eb2181568976e65131fa30f74ad3d0c88f6fea57a6`).
- Normalisation bundle sha256 `d716700cf40feb0527ece1e7b5f5edca21ac819a1de686d31bad249e58aaa084`; training config sha256 `e8de871172a425978419fe2b1a7f1447b896c5bedcfc6ff388a34167c395bebf`.
- Training seed 20260904; parameters 83457; epochs 11; early stopping on validation AUPRC.
- Calibration `p3_causal_tcn:platt_scaling:6810746b3671`; frozen threshold 0.23509196030026114 with 2-of-3 persistence and 10.0 s cooldown at a budget of 0.1 false alerts per clean mission.

### Event-level evaluation (frozen threshold)

| split | model_id | episodes | events | event_recall | false_alerts_per_clean_mission | median_useful_lead_seconds | brier_calibrated | ece_calibrated |
|---|---|---|---|---|---|---|---|---|
| validation | p3_causal_tcn | 324 | 62 | 0.3387096774193548 | 0.09722222222222222 | 5.700999999999993 | 0.029579626239055325 | 0.005357780368302284 |
| held_out_map_test | p1_threshold_rules | 1002 | 157 | 0.03184713375796178 | 0.46825396825396826 | 7.10499999999999 | 0.04073057444737778 | 0.04073057444737778 |
| held_out_map_test | p3_causal_tcn | 1002 | 157 | 0.22929936305732485 | 0.18253968253968253 | 3.8175000000000026 | 0.016036737531570006 | 0.008084453608610593 |
| held_out_map_test | p4_gru | 1002 | 157 | 0.19745222929936307 | 0.06349206349206349 | 3.811000000000007 | 0.015691502033678997 | 0.005868639624378052 |
| held_out_map_test | p5_compact_transformer | 1002 | 157 | 0.267515923566879 | 0.25396825396825395 | 5.841999999999999 | 0.016497606748826558 | 0.008976764511372772 |

### Unseen-family folds

| model_id | excluded_family | events | event_recall | false_alerts_per_mission |
|---|---|---|---|---|
| p1_threshold_rules | camera_occlusion | 16 | 0.0625 | 0.20634920634920634 |
| p1_threshold_rules | dynamic_blockage | 31 | 0.0 | 0.0 |
| p1_threshold_rules | lidar_dropout | 15 | 0.0 | 0.2222222222222222 |
| p1_threshold_rules | localisation_perturbation | 23 | 0.0 | 0.20634920634920634 |
| p1_threshold_rules | planner_oscillation | 28 | 0.0 | 0.0 |
| p1_threshold_rules | semantic_corruption | 8 | 0.0 | 0.2222222222222222 |
| p1_threshold_rules | wheel_slip | 20 | 0.05 | 0.208 |
| p3_causal_tcn | camera_occlusion | 16 | 0.3125 | 0.18253968253968253 |
| p3_causal_tcn | dynamic_blockage | 31 | 0.12903225806451613 | 0.392 |
| p3_causal_tcn | lidar_dropout | 15 | 0.0 | 0.23015873015873015 |
| p3_causal_tcn | localisation_perturbation | 23 | 0.34782608695652173 | 0.4444444444444444 |
| p3_causal_tcn | planner_oscillation | 28 | 0.0 | 0.0 |
| p3_causal_tcn | semantic_corruption | 8 | 0.0 | 0.023809523809523808 |
| p3_causal_tcn | wheel_slip | 20 | 0.1 | 0.296 |

### Ablations

| ablation | event_recall | false_alerts_per_clean_mission |
|---|---|---|
| no_localisation | 0.3225806451612903 | 0.19444444444444445 |
| no_motion_and_goal | 0.08064516129032258 | 0.013888888888888888 |
| no_perception_confidence | 0.3870967741935484 | 0.09722222222222222 |
| no_planner_controller | 0.3387096774193548 | 0.09722222222222222 |
| no_raw_sensor | 0.06451612903225806 | 0.0 |
| single_timestamp | 0.04838709677419355 | 0.08333333333333333 |

### Latency

- pending: latency report not generated

### Paired recovery

- Complete: True; H6 supported: False; completion difference R3−R0 -0.017857142857142905 (95% interval [-0.07738095238095238, 0.031746031746031744]); collision difference 0.0; estimator mixed_effects_logistic_variational_bayes_statsmodels.
- Guard rejections/violations by policy: {"R0": {"episode_count": 504, "guard_violation_count": 0, "guard_rejection_count": 0}, "R1": {"episode_count": 0, "guard_violation_count": 0, "guard_rejection_count": 0}, "R2": {"episode_count": 504, "guard_violation_count": 0, "guard_rejection_count": 0}, "R3": {"episode_count": 504, "guard_violation_count": 0, "guard_rejection_count": 0}}.
