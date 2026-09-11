# Model-development contracts (Protocol 1.1, post training-admission)

Status: engineering contract for stages 2–7 of the remaining work. Every rule in
`RESEARCH PROTOCOL - AUTONOMOUS ROBOT RELIABILITY.md` still applies; this file only
fixes file layouts and interfaces so parallel work packages compose.

## Non-negotiable rules (restated)

- Independent unit is the episode. Never split, sample, weight or bootstrap over windows
  as if independent. Hierarchical resampling is map → route → episode.
- Fitting uses development episodes only. Validation episodes select window/model/
  calibration/threshold/persistence/cooldown only. `held_out_map_test` episodes are
  never read before `configs/model_freeze.yaml` has `frozen: true`; every script that
  can touch them must call `src.protected_data.enforce_protected_boundary`.
- Normalisation is fitted on development episodes only (`NormalizationBundle`).
- Deployable inputs are exactly the 28 primary features × {value, age_seconds, missing}
  = 84 columns in the frozen order of `src.features.model_columns(primary)`. Nothing in
  `configs/leakage_denylist.yaml` may enter a model matrix. Fault family, severity,
  onset, ground truth, outcome fields and `run_id` are label-only.
- Threshold selection uses `src.evaluation.select_validation_threshold` under the
  frozen budget of 0.10 false alerts per clean validation mission with 2-of-3
  persistence and 10 s cooldown. Test-time threshold adaptation is forbidden.
- Every generated artifact is immutable (write with `x` mode or
  `src.dataset_inventory.publish_new_bytes`), carries `protected_test_used: false`
  (or `true` only for post-freeze confirmatory artifacts) and records SHA-256 of its
  inputs. Never overwrite; refuse if the path exists.
- Negative results are reported, never repaired by test-set tuning.

## Python environments

- System `python3` (3.12) has ROS 2 Jazzy, numpy, scipy, pandas, matplotlib, PyYAML.
  Use it (after `source scripts/env_research2.sh`) for anything that reads MCAP bags.
- `.venv/bin/python` = system site packages + `torch==2.13.0+rocm7.2` +
  `scikit-learn`. The GPU is an AMD Radeon AI PRO R9700 (gfx1201, 32 GB); PyTorch
  exposes it as `torch.device("cuda")`. Use it for all model fitting. Set seeds with
  `torch.manual_seed`, `numpy.random.default_rng` and `torch.use_deterministic_algorithms`
  where supported; record the seed in every training record.
- The simulator campaign runs concurrently on this host. Training jobs may run on the
  GPU but keep CPU dataloader workers ≤ 4 and `nice -n 19`.

## Derived per-episode artifacts

`scripts/extract_dataset_sequences.py` publishes, for one immutable inventory
(`data/manifests/<campaign>.episodes.jsonl`), the directory
`data/derived/<dataset_id>/` with:

| Sub-directory | File | Producer | Content |
|---|---|---|---|
| `operational_events/` | `<run_id>.yaml` | derive_operational_events | label-only event candidates |
| `annotations/` | `<run_id>.yaml` | extract_episode_annotation | causal annotation (episode start/end, injection, first terminal event) |
| `labels/` | `<run_id>.csv` | generate_labels | one row per decision on the label grid: `run_id,decision_index,decision_time,window_start,window_end,label,eligibility,primary_event_class,primary_event_time` |
| `telemetry/` | `<run_id>.csv` | extract_bag_scalar_telemetry | long-form scalar samples |
| `causal_features/` | `<run_id>.csv` | extract_scalar_features | value/age/missing at label-grid decisions |
| `window_features/` | `<run_id>.csv` | derive_window_features | + derived temporal features (label-grid history only) |
| `sequences/` | `<run_id>.npz` | assemble_episode_sequences | **eligible decisions only**: `X[n,10,84] float32`, `y∈{0,1}`, `decision_index`, `decision_time`, `feature_names`, `metadata_json` |
| `decisions/` | `<run_id>.npz` | assemble_episode_decisions | **every label-grid decision** with complete 5 s history: same arrays plus `eligibility` (str array) and `y∈{-1,0,1}` (−1 = excluded) |
| `extraction_manifest.jsonl` | | driver | one row per episode: identity, split, map/route/family/severity/seed, `artifact_sha256` per step, event class/time, sequence counts |
| `decisions_manifest.jsonl` | | assemble_episode_decisions | `run_id`, `decisions_sha256`, counts by eligibility |
| `extraction_report.yaml` | | driver | counts, config hashes, status |

Dataset ids in use:

- `balanced_pilot_v1-development-648` (development)
- `balanced_validation_v1-validation-324` (validation, selection-only)
- `targeted_development_v1-development-1212` (development; available after the
  campaign finalises)
- `research1_development_v1-development-647` (audit-only under PA-2026-09-03-03: excluded
  from the frozen fitting pool; separately reported natural-failure/domain-shift set)
- `development_supplement_v1-development-504` (development; preregistered, collected
  after the targeted campaign)

Training code must take a list of dataset ids (or extraction manifests) and must
refuse any manifest row whose `split` is not `development` for fitting or not
`validation` for selection. `decisions/` is what predictors score at inference time
so that the alarm policy sees every consecutive decision; `sequences/` is the
fitting set. Both are built from the same causal grid, so for eligible decisions
`X` is identical in the two files.

## Ablation feature groups (over the 28 primary values; companions follow)

```yaml
localisation:          [pose_covariance_trace, pose_jump]
planner_controller:    [global_path_length, global_path_curvature, local_path_length,
                        local_path_curvature, replan_rate, command_sign_changes, tracking_error]
perception_confidence: [confidence_mean, uncertainty_mean, inference_latency_ms]
raw_sensor:            [valid_return_fraction, minimum_front_range, minimum_left_range,
                        minimum_right_range, valid_return_trend, near_obstacle_trend,
                        left_right_imbalance]
motion_and_goal:       [command_linear, command_angular, measured_linear, measured_angular,
                        stopped_while_commanded, jerk, goal_distance,
                        remaining_distance_slope, progress_slope]
```

Removing a group zeroes its value channel and sets `missing=1`, `age=max_age`, so the
architecture and column order never change between ablations. Policy ablations:
`single_timestamp` (only the last time step is visible: earlier steps masked as
missing), `no_calibration` (raw sigmoid output), `no_persistence` (1-of-1, cooldown
kept). Configured in `configs/ablations.yaml`.

## Model identifiers

| ID | model_id | Notes |
|---|---|---|
| P1 | `p1_threshold_rules` | `src.models.threshold_rules`; thresholds tuned on development, frozen after validation check, stored in `configs/baseline_rules.yaml` |
| P2 | `p2_reconstruction_ae` | normal-only autoencoder trained on eligible-negative development windows of clean episodes; score = min-max-scaled reconstruction error (validation-fitted scaling stays monotone) |
| P3 | `p3_causal_tcn` | primary; causal dilated Conv1d residual blocks + dropout, sigmoid head |
| P4 | `p4_gru` | GRU, parameter budget within ±25 % of P3 |
| P5 | `p5_compact_transformer` | optional; include only if p95 latency ≤ P3 × 2 and validation AUPRC ≥ P4 |
| P6 | `p6_oracle` | risk = 1 on eligible-positive decisions else 0; analysis upper bound only |

## Training and prediction CLI (agents code against this)

```
.venv/bin/python scripts/train_predictor.py \
    --model-id p3_causal_tcn --config configs/models/p3_causal_tcn.yaml \
    --train-dataset balanced_pilot_v1-development-648 [--train-dataset ...] \
    --selection-dataset balanced_validation_v1-validation-324 \
    [--exclude-family <family>] [--ablation <name>] --seed 20260903 \
    --output-root models/
```

Writes `models/<run_name>/` containing `checkpoint.pt`, `normalization.json`,
`training_record.json` (seed, config sha256, dataset manifests and their sha256,
episode counts, positive/negative window counts, natural and rebalanced prevalence,
epochs, early-stopping metric = validation AUPRC, parameter count, wall time, GPU
name, git commit) and `checkpoint.sha256`. Never overwrites.

```
.venv/bin/python scripts/predict_decisions.py \
    --model-dir models/<run_name> --dataset <dataset_id> [--dataset ...] \
    --output reports/predictions/<name>.csv [--allow-protected-after-freeze]
```

Writes the immutable **prediction table** with exactly these columns, one row per
decision in `decisions/<run_id>.npz`, sorted by `run_id, decision_index`:

```
run_id, decision_index, decision_time, split, map_id, route_id, fault_family,
severity, seed, protected_test_used, eligibility, label, primary_event_class,
primary_event_time, model_id, raw_score, risk_score
```

`raw_score` is the uncalibrated model output in [0, 1]; `risk_score` equals
`raw_score` until a calibrator is applied by `scripts/apply_calibration.py`, which
writes a new table with the same columns and `risk_score` replaced. Alarm decisions
are added by `scripts/apply_alarm_policy.py` (adds `alarm`, `persistent`). Existing
consumers: `scripts/select_alarm_threshold.py`, `src.evaluation.evaluate_event_warnings`,
`src.evaluation.calibration`.

Latency: `scripts/measure_inference_latency.py --model-dir ...` reports median, p95,
max of feature-window preparation + model + policy per decision on CPU (deployment
target is the robot host, not the GPU) with batch size 1.

## Freeze record

`scripts/freeze_model.py` fills `configs/model_freeze.template.yaml` into
`configs/model_freeze.yaml` (refuses if present), sets `configs/alarm_policy.yaml`
`threshold` from the selection record, and appends a `protocol_change` record to the
research log. Only after that may `scripts/assign_protected_split.py` run.

## Research log

Every stage transition appends to `logs/research-log.jsonl` through
`scripts/research_log.py add --kind ... --actor "<name>" --message ... --metadata '{}'`.
Actor for AI-generated records: `Claude Fable 5.1 (AI assistant, directed by the researcher)`.
