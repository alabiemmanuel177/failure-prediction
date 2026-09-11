# Closed-loop recovery: live bring-up smoke plan (work package Q)

Status: engineering plan. Nothing in this document has run live yet; every node it
names is marked UNTESTED LIVE in its module docstring. Follow the order exactly.
The sequential campaign on ROS domain 52 must not be disturbed: every command below
runs inside a worker slot (domain 60-65, its own Gazebo partition) or after that
campaign has finished.

## 0. What exists

| Piece | Path | Runs when |
|---|---|---|
| Online failure monitor (2 Hz windows, frozen checkpoint on CPU, calibrator, alarm policy, signal-group diagnosis, warning hand-off) | `ros_ws/src/failure_monitor/` | launch argument `recovery_policy` is not `R0` |
| Recovery manager (guard with final authority, two-phase safe stop, live executor, sidecar) | `ros_ws/src/recovery_manager/` | same condition |
| Policy plumbing | `src/recovery/plumbing.py`, `src/experiments/recovery_campaigns.py` | always importable, inert for R0 |
| Episode runner argument | `scripts/run_research2_episode.py --recovery-policy {R0,R1,R2,R3,RP_<action>}` | absent => byte-identical R0 path |
| Campaign runner | `scripts/run_balanced_pilot.py` (manifests that declare `recovery_policies`) | `recovery_pilot_v1`, `paired_recovery_v1` |
| Tables | `scripts/build_recovery_outcome_table.py`, `scripts/build_recovery_cost_table.py` | after a campaign |

Monitor data path (identical code to the offline pipeline): live messages ->
`scripts/extract_bag_scalar_telemetry.feature_values` -> `SampleBuffer` (receive-time
stamps on the simulation clock) -> `src.features.extract_decision_rows` at
`mission_start + 0.5 k` -> `src.features.derive_window_features` -> 84 columns of
`model_columns(primary)` -> `scripts/predict_decisions.LoadedModel.prepare/score_prepared`
-> `Calibrator.apply` -> `apply_alarm_policy` replayed over the stream. The replay test
`tests/test_failure_monitor.py::test_replay_windows_equal_offline_decisions` proves the
windows equal `decisions/<run_id>.npz` rows (max |diff| 0.0) on a recorded episode.

Diagnosed signal group: after mask and normalisation, the last time step's value
channels are grouped by `configs/ablations.yaml` `feature_groups`; the group with the
largest mean |z| over observed features is reported (localisation -> localisation,
planner_controller -> planning, raw_sensor -> frontal_blockage, perception_confidence
-> perception, motion_and_goal -> unknown; ties -> unknown).

Hand-off: on every alarm the monitor publishes `/research2/events` (`predictor_alarm`)
and `/research2/recovery_requests` (risk, group, guard inputs). The manager, when the
live gate is open and the robot is moving, first executes the safe stop
(`cancel_task`, `zero_cmd_vel`), measures `stopped` on `/odom`, then runs the policy
against the frozen guard with the measured state and executes the admitted action.
`controlled_stop` = cancel, zero, resume (it is a first response, not an abort);
`request_assistance` = cancel, zero, no resume.

## 1. Build (once, never while the sequential campaign is mid-episode is not required, but do it between waves)

```bash
source scripts/env_research2.sh
cd ros_ws && colcon build --symlink-install --packages-select failure_monitor recovery_manager && cd ..
source scripts/env_research2.sh       # picks up ros_ws/install
ros2 pkg executables failure_monitor   # expect: failure_monitor failure_monitor
ros2 pkg executables recovery_manager  # expect: recovery_manager recovery_manager
python3 -m pytest -q tests/test_recovery_plumbing.py tests/test_failure_monitor.py
.venv/bin/python -m pytest -q tests/test_failure_monitor.py   # torch-backed scoring test
```

## 2. Choose a worker slot (never domain 52)

```bash
export RESEARCH2_WORKER_SLOT=5 RESEARCH2_CONCURRENCY_WORKERS=6
export ROS_DOMAIN_ID=65 RESEARCH2_ROS_DOMAIN_ID=65 GZ_PARTITION=research2_w5
export ROS_LOG_DIR=$RESEARCH2_ROOT/logs/ros/parallel/research2_w5 GZ_HOMEDIR=$RESEARCH2_ROOT/logs/gazebo/parallel/research2_w5
mkdir -p "$ROS_LOG_DIR" "$GZ_HOMEDIR"
```
(`src/experiments/worker_slots.py` is the authority; `run_research2_episode.py` refuses
an inconsistent slot.)

## 3. Recommendation-only smoke on one development map (no execution, no evidence needed)

Before the freeze this is an engineering smoke: `--recovery-smoke-unfrozen` makes the
monitor accept an explicit checkpoint and calibrator and marks every event and the
summary `engineering_smoke: true`. Such episodes are never analysed.

```bash
python3 scripts/run_research2_episode.py \
  --map dev_00 --route dev_00_r0 --system s0 --family lidar_dropout --severity medium \
  --seed 777001 --clean-prefix-seconds 5 --planned-onset-seconds 8 \
  --placement-mode path_fraction --recording-profile compact_v2 \
  --campaign-id recovery_smoke_manual --episode-key smoke-R2-001 \
  --output-root data/raw_engineering_smoke \
  --recovery-policy R2 --recovery-smoke-unfrozen \
  --recovery-model-dir models/preliminary_v1/p3_causal_tcn__seed20260903 \
  --recovery-calibrator reports/calibration/preliminary_v1/p3_causal_tcn.calibrator.json
```

Proof lines (in `data/raw_engineering_smoke/logs/<run_id>/sim.log` or the launch
output):

| Item | Log line |
|---|---|
| monitor started with the right assets | `failure monitor ready: policy=R2 threshold=<t> engineering_smoke=True` |
| grid anchored on mission start | `mission start observed at <sim time>` |
| decisions flowing at 2 Hz | `logs/failure-monitor/<run_id>.json` `decision_count` grows; `ros2 topic hz /research2/warning` ~2 Hz |
| alarm and hand-off | `ALARM <run_id>-w001 t=... risk=... group=...` then manager `recommendation_only` on `/research2/recovery_decisions` |
| manager locked | `recovery manager policy=R2 live gate: live_execution parameter is false; recommendation-only mode` |
| summary block | `system.recovery_policy_id: R2`, `system.recovery.warnings_issued`, `system.recovery.actions[*].execution_status == refused_live_execution_locked` |

After the freeze drop `--recovery-smoke-unfrozen`, `--recovery-model-dir` and
`--recovery-calibrator`: the monitor reads `configs/model_freeze.yaml`, verifies the
checkpoint and calibrator SHA-256 and refuses any override.

## 4. Live-execution smoke (after the freeze, after the evidence file exists)

Same command with `--recovery-live-execution` (and `--recovery-relocalisation-available`
only once item 1 below is verified). The manager refuses to start unless
`configs/recovery_live_evidence.yaml` verifies every item; that refusal is the
expected result of the first attempt.

Proof lines:

| Item | Log line |
|---|---|
| gate open | `recovery manager policy=R2 live gate: live execution authorised by frozen evidence` |
| safe stop before selection | event `recovery_safe_stop` with `status: executed`, then `measured_stop.stopped: true` in `logs/recovery/<run_id>.json` |
| guard rejections logged | event `recovery_guard_rejection` with `rejected_actions` on every warning |
| action executed | event `recovery_action_executed` with `steps` and `seconds`; summary `system.recovery.intervention_count >= 1` |
| resumption | Nav2 accepts the new goal; episode terminal state is not `invalid` |

## 5. The five live-evidence items (configs/recovery_guards.yaml) and how to prove each

1. `relocalisation_procedure_availability_is_explicit` -- the relocalise action calls
   `/reinitialize_global_localization` (`recovery_manager/nav2_commander.py`).
   Operationally: with the gate open, force `RP_relocalise` on a
   `localisation_perturbation` episode and confirm the service responds within the
   step timeout and AMCL re-converges. Proof: `recovery_action_executed` with
   `action: relocalise`, step `relocalise` `seconds < 10`, no `TimeoutError`; the
   monitor's `pose_covariance_trace` falls afterwards. Until then keep
   `--recovery-relocalisation-available` off (the guard then rejects relocalise).
2. `rear_clearance_sensor_and_frame_validated` -- the monitor's `rear_clearance_m` is
   the minimum finite `/scan` beam with |angle| >= 5pi/6 in the laser frame
   (`monitor_core.scan_clearances`); the guard threshold 0.35 m compares the raw beam,
   not the footprint. Proof: park the robot with a wall 0.5 m behind it and read
   `rear_clearance_m` in the `predictor_alarm` request (`ros2 topic echo
   /research2/recovery_requests`); confirm the laser frame's x-axis points forward in
   `robot_state_publisher` (`ros2 run tf2_ros tf2_echo base_link base_scan`).
3. `rotation_swept_volume_clearance_validated` -- `rotation_clearance_m` is the
   all-round minimum finite beam. Proof: same parked test with an obstacle at 0.3 m
   on one side gives `rotation_clearance_m ~ 0.3` and the guard rejects
   `spin_active_rescan` (`recovery_guard_rejection` lists it); at 0.5 m the spin
   executes without a `/collision_event`.
4. `stop_command_and_nav2_cancel_order_validated` -- every sequence starts with
   `cancel_task` then `zero_cmd_vel` (`live_execution.SAFE_STOP_PREFIX`). Proof:
   `recovery_safe_stop` steps in that order and `measured_stop.stopped: true` within
   `stop_settle_seconds` (2 s); `ros2 topic echo /cmd_vel` shows zeros after the
   cancel.
5. `every_guard_rejection_logged` -- proof: for each warning in
   `logs/recovery/<run_id>.json` the `events` list contains one
   `recovery_guard_rejection` whose `rejected_actions` equals the ineligible set of
   that decision's `guard_results`; `tests/test_recovery_live_execution.py` covers the
   executor, the sidecar check is manual on the smoke episode.

## 6. Writing configs/recovery_live_evidence.yaml

```yaml
schema_version: 1
frozen: true
frozen_utc: "2026-09-04T12:00:00Z"
frozen_by: Emmanuel Alabi Olasubomi
guard_config_sha256: <sha256sum configs/recovery_guards.yaml>
smoke_run_ids: [<run_id of the section-3 episode>, <run_id of the section-4 episode>]
required_live_evidence_before_execution:
  relocalisation_procedure_availability_is_explicit:
    verified: true
    verified_utc: "2026-09-04T11:40:00Z"
    evidence: "run <id>: relocalise executed in 1.8 s, AMCL covariance trace 0.9 -> 0.2"
  rear_clearance_sensor_and_frame_validated:
    verified: true
    verified_utc: "..."
    evidence: "run <id>: rear_clearance_m 0.49 with wall at 0.50 m; base_scan x forward"
  rotation_swept_volume_clearance_validated:
    verified: true
    verified_utc: "..."
    evidence: "run <id>: spin rejected at 0.30 m, executed at 0.50 m, no contact"
  stop_command_and_nav2_cancel_order_validated:
    verified: true
    verified_utc: "..."
    evidence: "run <id>: recovery_safe_stop cancel_task -> zero_cmd_vel, stopped in 0.6 s"
  every_guard_rejection_logged:
    verified: true
    verified_utc: "..."
    evidence: "run <id>: 3 warnings, 3 recovery_guard_rejection events"
```

`recovery_manager.live_execution.evaluate_live_gate` accepts the file only when
`frozen: true`, the SHA-256 matches the current `configs/recovery_guards.yaml`, and all
five items are `verified: true` with an ISO timestamp. Editing the guard config
afterwards invalidates the evidence by construction. Append a `protocol_change`
record to the research log when the file is frozen.

## 7. Campaign order after the freeze

1. `python3 scripts/build_recovery_pilot_manifest.py` (882 episodes on validation maps;
   writes `data/manifests/recovery_pilot_v1.yaml`).
2. Run it with six workers (PA-2026-09-03-04) through the parallel runner, or
   sequentially with `scripts/run_balanced_pilot.py --manifest data/manifests/recovery_pilot_v1.yaml`;
   the runner refuses until the freeze, threshold and evidence exist.
3. `scripts/build_recovery_outcome_table.py --manifest data/manifests/recovery_pilot_v1.yaml --output reports/recovery/recovery_pilot_v1_outcomes.csv`
4. `scripts/build_recovery_cost_table.py --outcomes reports/recovery/recovery_pilot_v1_outcomes.csv --manifest data/manifests/recovery_pilot_v1.yaml --output reports/recovery/recovery_pilot_v1_costs.csv`
5. `scripts/train_recovery_selector.py reports/recovery/recovery_pilot_v1_costs.csv --output models/recovery_selector/r3_cost_sensitive_ridge_v1.json --fit-split validation --amendment-id PA-2026-09-03-04`
6. `scripts/build_recovery_campaign_manifest.py` for `paired_recovery_v1` (held-out maps;
   set the policy list to R0, R2, R3 per PA-2026-09-03-04), run it, then
   `build_recovery_outcome_table.py --allow-protected-after-freeze` and
   `scripts/analyze_paired_recovery.py`.

## 8. Known limitations to state in the cards

* Guard inputs other than `stopped` are engineering estimates from the last window
  (`monitor_core.StateEstimatorConfig`, thresholds recorded in the monitor sidecar).
* Rear and rotation clearances are raw beam minima, not footprint-corrected.
* The live grid anchors on the first `/behavior_tree_log` message, a few milliseconds
  after the label-only `goal_dispatched` event the offline grid uses; the model is
  translation-invariant to that phase but the live decision indices are not the
  offline ones.
* R2 never selects `wait` by diagnosis (transient obstruction is not identifiable
  from one window); the pilot's `RP_wait` arm supplies the selector's evidence for it.
