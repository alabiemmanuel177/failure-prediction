# Closed-loop recovery: live bring-up evidence draft (work package Q)

Status: engineering draft for the researcher's signature. Nothing here is signed; the
researcher writes `configs/recovery_live_evidence.yaml` (section 6) and the research-log
record. Every run below is an engineering smoke: campaign `recovery_smoke_manual`,
output root `data/raw_engineering_smoke` (never `data/raw`), worker slot 5
(ROS domain 65, Gazebo partition `research2_w5`), development map `dev_00` only,
system S0, recording profile `compact_v2`. The model was frozen throughout
(`configs/model_freeze.yaml`, threshold 0.23509196030026114, no overrides). Sidecars:
`logs/failure-monitor/<run_id>.json`, `logs/recovery/<run_id>.json`; launch output:
`data/raw_engineering_smoke/logs/<run_id>/sim.log`.

Guard config hash at the time of every run:
`sha256(configs/recovery_guards.yaml) = 04017b745d8dc81baa19b1b50d7a08a32dc3064b4af3dcb114b97d306000f12a`.

## 1. Runs (11 episodes, budget 20)

| # | run_id | episode_key | route / family / severity / seed | policy, gate | terminal | purpose |
|---|---|---|---|---|---|---|
| 1 | 91a4dfe1-2a17-4d5a-a2fd-739a4636d073 | smoke-R2-001 | dev_00_r0 lidar_dropout medium 777001 | R2, locked | invalid (preflight) | failed: `malformed launch argument 'recovery_model_dir:='` (defect A) |
| 2 | a9e6e941-1d7a-4cf0-b373-79d12fe88017 | smoke-R2-001 | same | R2, locked | invalid (sidecars missing) | failed: both nodes crashed at start (defects B, C) |
| 3 | a244fab2-7d8a-45b6-8e18-d03ab0198225 | smoke-R2-001 | same | R2, locked | success | section 3 recommendation-only smoke: PASS |
| 4 | da555be3-6026-4688-8241-00de78559a62 | smoke-live-refusal-001 | same | R2, `--recovery-live-execution`, no evidence file | invalid (manager refused, as required) | section 4 first attempt: refusal proven |
| 5 | 804f94b7-d28f-49ad-bbc1-0fca7c4866e8 | smoke-live-R2-001 | same | R2, gate open (draft override) | success, 2 backups | items 4, 5 |
| 6 | 8c7e0119-08d8-4e6b-b865-b73a5ac2f910 | smoke-live-RPreloc-off | dev_00_r0 localisation_perturbation medium 777002 | RP_relocalise, gate open | false_arrival, no alarm | item 1 attempt (peak risk 0.195 < threshold) |
| 7 | d5ceb629-7c0a-4cfd-81db-cf12a174f59a | smoke-live-RPreloc-on | same + `--recovery-relocalisation-available` | RP_relocalise, gate open | false_arrival, no alarm | item 1 attempt (no alarm) |
| 8 | 3936c2b5-9f53-49f8-a76c-6ced027110b4 | smoke-live-RPspin-open | dev_00_r0 lidar_dropout medium 777001 | RP_spin_active_rescan, gate open | timeout (budget exhausted -> request_assistance) | item 3: spin executed at 0.59 m, no contact |
| 9 | b6a52308-c07a-4267-af9f-b689abd3d493 | smoke-live-RPspin-block | dev_00_r0 dynamic_blockage medium 777003 | RP_spin_active_rescan, gate open | timeout | item 3: spin rejected at 0.34 m |
| 10 | 34546989-ffc9-4be3-b683-a409becc6ed9 | smoke-live-RPreloc-high-off | dev_00_r1 localisation_perturbation high 777004 | RP_relocalise, gate open, availability off | timeout | item 1: relocalise rejected by guard |
| 11 | b482c289-80bb-4f9d-a178-4e6256797182 | smoke-live-RPreloc-high-on | same + `--recovery-relocalisation-available` | RP_relocalise, gate open | timeout | item 1: relocalise executed twice |

Parked test (not an episode; `ros2 launch failure_experiment shared_sim.launch.py` in slot 5,
`run_id:=parked-clearance-check3`, robot at the dev_00_r0 start pose (-2.0, -0.5, yaw 0),
`family:=none`, obstacles spawned with `ros2 run ros_gz_sim create -world default`):
items 2 and 3 measurements, 2026-09-05T23:32Z.

## 2. Section 3 proof lines (recommendation-only, run a244fab2, frozen model)

| Item | Observed |
|---|---|
| monitor ready | `[research2_failure_monitor]: failure monitor ready: policy=R2 threshold=0.23509196030026114 engineering_smoke=False` (sidecar `assets.threshold_source: configs/alarm_policy.yaml`, `assets.frozen: true`) |
| grid anchored | `[research2_failure_monitor]: mission start observed at 13.056` |
| 2 Hz decisions | `ros2 topic hz /research2/warning --window 20`: `average rate: 2.000  min: 0.498s max: 0.501s`; sidecar `decision_count: 29`, decision stride 0.5 s (18.056 .. 32.056), compute_ms median 2.15 / p95 2.58 |
| alarm and hand-off | `ALARM a244fab2-...-w001 t=22.056 risk=0.613 group=frontal_blockage`, `ALARM ...-w002 t=32.056 risk=0.607 group=frontal_blockage`; manager sidecar decisions w001 `controlled_stop` ("backup rejected by guard": robot moving), w002 `backup` ("preferred action eligible"), both `execution_status: refused_live_execution_locked` |
| manager locked | `[research2_recovery_manager]: recovery manager policy=R2 live gate: live_execution parameter is false; recommendation-only mode` |
| summary block | `system.recovery_policy_id: R2`, `label_only.recovery_policy_id: R2`, `system.recovery.warnings_issued: 2`, `decisions_scored: 29`, `engineering_smoke: false`, `live_execution_enabled: false`, every `actions[*].execution_status == refused_live_execution_locked`, `intervention_count: 0`; terminal `success` |

## 3. Section 4 refusal without the evidence file (run da555be3)

```
[recovery_manager-9]     raise RuntimeError(f"live execution refused at startup: {self.gate.reason}")
[recovery_manager-9] RuntimeError: live execution refused at startup: live evidence file is absent: /home/eao/failure-prediction/configs/recovery_live_evidence.yaml
[ERROR] [recovery_manager-9]: process has died [pid 431337, exit code 1, ...]
```
The runner then invalidated the episode (`recovery_node_sidecar_missing`). `configs/recovery_live_evidence.yaml`
did not exist at any point during this work and was never written.

Gate open (all later live runs) used the provisional file through the new engineering-only flag:
```
[research2_recovery_manager]: recovery manager policy=R2 live gate: live execution authorised by frozen evidence (ENGINEERING EVIDENCE OVERRIDE: /home/eao/failure-prediction/configs/recovery_live_evidence.draft.yaml)
```
The manager sidecar records `gate.evidence_override: true` and `gate.evidence_path`; the episode
summary records `system.recovery_live_evidence_override`.

## 4. The five items

### Item 1: relocalisation_procedure_availability_is_explicit -- verified: TRUE (service and guard); re-convergence PARTIAL

* Run 34546989 (availability off, dev_00_r1, localisation_perturbation high, seed 777004):
  6 warnings, every alarm `localisation_poor: true`, `relocalisation_available: false`. w001, w002:
  `forced relocalise rejected by guard; controlled stop executed`, guard reason
  `requires stopped robot, poor localisation health, and an available procedure`; w003-w006:
  `request_assistance` (budget exhausted). No `relocalise` step ran.
* Run b482c289 (availability on, same configuration): w001 t=75.996 and w002 t=88.496
  `recovery_action_executed`, `action: relocalise`, steps
  `cancel_task 0.001 s -> zero_cmd_vel 0.060 s -> relocalise 0.003 s -> resume_navigation 0.001 s`,
  no `TimeoutError`, no `RuntimeError`; `/reinitialize_global_localization` responded in 3 ms.
  Both executions preceded by `recovery_safe_stop` (executed) and `measured_stop.stopped: true`.
* AMCL response (bag `/amcl_pose`, trace = cov[0]+cov[7]+cov[35] = the monitor's
  `pose_covariance_trace`): before the first relocalise 0.457 (t=70.6); after the global
  reinitialisation 10.64 (t=76.2), 12.69 (t=81.1) falling to 8.79 (t=82.3); after the second
  relocalise 9.87 (t=88.9) falling monotonically to 6.21 (t=93.5). The trace falls after each
  relocalise as the doc requires, but it did NOT return below the `localisation_poor` threshold
  (0.5) before the episode ended (third alarm exhausted the budget, `request_assistance`, no
  resume, runner timeout after the 30 s grace); episode `localisation_error_max_m: 1.94`.
* Unverified part: full AMCL re-convergence to the pre-fault level. `/reinitialize_global_localization`
  spreads particles over the whole map; on this map the pose had not re-converged within ~17 s
  of driving. The commander also supports an `initialpose` republish of a last known pose
  (`Nav2LiveCommander(last_known_pose=...)`) but the node does not wire it. The researcher
  should decide whether "availability is explicit" (service exists, is called, responds within
  the step timeout, guard gates it on the explicit flag) is sufficient for the pilot's
  `RP_relocalise` arm, or whether the relocalise procedure itself needs changing first.

### Item 2: rear_clearance_sensor_and_frame_validated -- verified: TRUE

Parked test, `failure_monitor.monitor_core.scan_clearances` applied to live `/scan`
(frame_id `base_scan`, 360 beams, angle_min 0, angle_max 6.28), 10 scans per condition:

| Condition | rear_clearance_m (min / median) | rotation_clearance_m (min / median) | beam angle of the minimum |
|---|---|---|---|
| A. map walls only | 0.476 / 0.489 | 0.476 / 0.486 | -2.645 rad (rear right: map wall) |
| B. slab with its face 0.40 m behind the laser | 0.377 / 0.385 | 0.377 / 0.385 | 3.131 rad (directly behind) |
| C. + slab with its face 0.30 m behind | 0.276 / 0.283 | 0.276 / 0.283 | -2.907 rad |
| after removing both slabs | 0.484 / 0.489 | 0.476 / 0.488 | -2.75 rad |

The rear sector responds to an object placed directly behind the robot with a beam reading
about 1.5-2 cm shorter than the geometric distance (the LDS beam noise plus the raw beam vs
box face; the error is on the conservative side for the 0.35 m guard). At 0.40 m the guard
admits `backup`; at 0.30 m it rejects it. Frame: `ros2 run tf2_ros tf2_echo base_link base_scan`
-> `Translation: [-0.064, 0.000, 0.122]`, `Rotation: in RPY (radian) [0.000, -0.000, 0.000]`
(from `robot_state_publisher`, TB3 waffle URDF `scan_joint` rpy 0 0 0): the laser x axis
points forward, so |angle| >= 5pi/6 is the rear sector. Live confirmation: run 804f94b7 w001
request `rear_clearance_m 1.808`, w002 `1.101` while driving down the open corridor.

### Item 3: rotation_swept_volume_clearance_validated -- verified: TRUE

* Parked, side post with its face 0.30 m to the left of the laser: `rotation_clearance_m`
  0.271 / 0.284 (min / median) at beam angle 1.539 rad (left), while `rear_clearance_m` stayed
  0.482 / 0.488 (the two quantities are independent as designed). Side post at 0.50 m: rotation
  0.466 / 0.480 (the map wall behind at 0.49 m is then the all-round minimum).
* Live rejection, run b6a52308 (dynamic_blockage medium, RP_spin_active_rescan): w001
  `rotation_clearance_m 0.3359`, w002 `0.3401` -> `recovery_guard_rejection` lists
  `spin_active_rescan` (`requires stopped robot and verified rotation clearance`), forced
  action degraded to `controlled_stop`; no `/collision_event` (`collision: false`).
* Live execution, run 3936c2b5 (lidar_dropout medium, RP_spin_active_rescan): w001
  `rotation_clearance_m 0.588`, w002 `0.599` -> `recovery_action_executed`, `action:
  spin_active_rescan`, step `spin` 1.90 s (Nav2 Spin 1.5708 rad, result SUCCEEDED), resumed
  (`Navigating to goal: 1.0 -0.5`); `collision: false`, `collision_count: 0`. At w003 the
  budget (2) was exhausted and the guard rejected everything but `request_assistance`.
* Not exercised: a live spin executed at exactly 0.50 m (the executed spins had 0.59-0.60 m,
  the rejected ones 0.34 m; the parked probe covers 0.30 and 0.50 m). The guard threshold is 0.40 m.

### Item 4: stop_command_and_nav2_cancel_order_validated -- verified: TRUE

* Every `recovery_safe_stop` event in every live run (16 safe stops over runs 804f94b7,
  3936c2b5, 34546989, b482c289) has steps `['cancel_task', 'zero_cmd_vel']` in that order,
  0.062-0.063 s total, and `measured_stop.stopped: true`. Settle times measured on `/odom`
  after the commands: 0.20 s (804f94b7 w001, w002; 3936c2b5 w001), 0.05 s, 0.55 s (34546989
  w001, from 0.17 m/s), 0.25 s, 0.20 s (b482c289 w002, w003); 0.0 s when the robot was
  already stationary. All within `stop_settle_seconds` (2 s). Robot speed before the stop
  0.17-0.18 m/s (`/odom`), first stationary `/odom` sample 0.71 s after the alarm decision time
  (the alarm is issued up to 0.5 s after the window; the commands run ~0.5 s later).
* `/cmd_vel` from the bag of run 804f94b7 around the first safe stop (recorder time):
  `23.013 lx=0.1825`, `23.025 0.0`, `23.046 0.0`, `23.082 0.0575` (the controller server's last
  command, in flight while the cancel completed), then `0.0` x 8, then `-0.05` (Nav2 BackUp).
  The runner-side proof is the label-only event `mission_goal_cancelled_by_recovery`
  (`/navigate_to_pose` result status CANCELED on the runner's own goal), present in every live
  run.
* Cancel order defect found and fixed before this evidence (defect D below): the original
  `cancel_task` cancelled only a goal the navigator itself had sent, never the runner's mission
  goal. It now also sends the action protocol's cancel-all request (`last_cancel` recorded:
  `return_code 0`).

### Item 5: every_guard_rejection_logged -- verified: TRUE

Over the five live runs: 22 warnings, 22 `recovery_guard_rejection` events, and for every
decision `sorted(ineligible actions of guard_results) == rejected_actions` of exactly one
`recovery_guard_rejection` event with that `warning_id` (checked programmatically on the
sidecars; also true for the recommendation-only run a244fab2: 2 warnings, 2 events). The
events are also published on `/research2/events` (`causal_role: label_only`), and the
summary's `system.recovery.guard_rejections` mirrors them. Rejections cover the guard
reasons observed live: not stopped, unverified rear/rotation clearance, clearance below
threshold, no relocalisation procedure, no transient obstruction, recovery budget exhausted.

## 5. Defects found and fixed during the bring-up (all covered by the test suite)

| | Defect | Fix |
|---|---|---|
| A | `recovery_launch_arguments` emitted `recovery_model_dir:=` (empty) and `ros2 launch` refused the whole launch as malformed, so the post-freeze form never reached the simulator | empty-valued keys are omitted; declared launch defaults apply (`src/recovery/plumbing.py`) |
| B | `RecoveryManager.__init__` assigned `self.executor`, which is rclpy's `Node.executor` property setter -> `AttributeError: 'LiveExecutor' object has no attribute 'add_node'` | renamed to `self.live_executor` |
| C | the monitor console script runs under `/usr/bin/python3`, which has no torch (`ModuleNotFoundError`) | launch file runs the monitor with the project venv interpreter as a prefix (the venv includes the system site-packages; same numpy) |
| D | `BasicNavigator.cancelTask()` only cancels the navigator's own goal; the runner's mission goal was never cancelled | `Nav2LiveCommander.cancel_task` also sends a zero-id `CancelGoal` (cancel all) to `/navigate_to_pose/_action/cancel_goal` and checks the return code |
| E | `/reinitialize_global_localization` client lived on the manager node whose only callback thread is blocked inside the request handler: the future could never complete (relocalise would always time out) | service clients live on the navigator node and are spun explicitly (`_call_service`) |
| F | the episode runner ended the episode the moment its goal was cancelled, tearing the stack down mid-action; the manager only snapshotted its sidecar after the whole sequence | for `--recovery-live-execution` the runner follows the newest `navigate_to_pose` goal on the action status topic (`resumed_mission_terminal`, 30 s grace, `timeout` when nothing resumes, e.g. `request_assistance`); the manager snapshots after every event |
| G | Nav2 behaviour failures (rejected goal, non-SUCCEEDED result) were reported as executed | `backup`/`spin`/`resume_navigation` raise on rejection or a non-SUCCEEDED result |
| H | when the monitor asserted `stopped: true` the manager skipped the safe stop and the `/odom` measurement, so the guard saw an asserted stop | the safe stop and measurement are unconditional when the gate is open (`measured_stop.asserted_by_monitor` records what the monitor claimed) |
| I | no way to run a live smoke before the signed evidence exists | `--recovery-evidence-override <path>` on the runner, refused unless `--campaign-id` starts with `recovery_smoke_` and `--output-root` is `data/raw_engineering_smoke`, never the signed path; launch argument `recovery_live_evidence`; manager logs and records the override |

Also: the resumed goal now carries the runner's `goal_yaw` (previously the route's missing
yaw defaulted to 0). The `recovery_live_evidence` launch argument is empty by default, so the
manager reads the signed `configs/recovery_live_evidence.yaml`.

Tests: `python3 -m pytest -q tests/test_recovery_plumbing.py tests/test_failure_monitor.py tests/test_recovery_live_execution.py`
-> 41 passed, 1 skipped (was 34 passed, 1 skipped before the changes);
`.venv/bin/python -m pytest -q tests/test_failure_monitor.py` -> 11 passed.

## 6. Ready-to-sign block for configs/recovery_live_evidence.yaml

The researcher signs this; `frozen_by` and `frozen_utc` are the researcher's. The hash is the
current `configs/recovery_guards.yaml`; editing the guard config invalidates it.

```yaml
schema_version: 1
frozen: true
frozen_utc: "<researcher fills in>"
frozen_by: Emmanuel Alabi Olasubomi
guard_config_sha256: 04017b745d8dc81baa19b1b50d7a08a32dc3064b4af3dcb114b97d306000f12a
smoke_run_ids: [a244fab2-7d8a-45b6-8e18-d03ab0198225, 804f94b7-d28f-49ad-bbc1-0fca7c4866e8]
engineering_smoke_run_ids:
  recommendation_only: a244fab2-7d8a-45b6-8e18-d03ab0198225
  refusal_without_evidence: da555be3-6026-4688-8241-00de78559a62
  live_r2: 804f94b7-d28f-49ad-bbc1-0fca7c4866e8
  spin_executed: 3936c2b5-9f53-49f8-a76c-6ced027110b4
  spin_rejected: b6a52308-c07a-4267-af9f-b689abd3d493
  relocalise_rejected: 34546989-ffc9-4be3-b683-a409becc6ed9
  relocalise_executed: b482c289-80bb-4f9d-a178-4e6256797182
  parked_clearance: parked-clearance-check3
required_live_evidence_before_execution:
  relocalisation_procedure_availability_is_explicit:
    verified: true
    verified_utc: "2026-09-05T23:43:51Z"
    evidence: "run b482c289: relocalise executed at w001/w002, /reinitialize_global_localization responded in 0.003 s, no TimeoutError; AMCL covariance trace 0.46 -> 10.6 (global reset) then falling 12.7 -> 8.8 and 9.9 -> 6.2 while driving, not below 0.5 before the episode ended; run 34546989: guard rejected relocalise with the availability flag off"
  rear_clearance_sensor_and_frame_validated:
    verified: true
    verified_utc: "2026-09-05T23:32:29Z"
    evidence: "parked-clearance-check3: rear_clearance_m 0.385 with a slab face at 0.40 m and 0.283 at 0.30 m (map wall only: 0.489); base_link->base_scan translation [-0.064, 0, 0.122], RPY 0, x forward"
  rotation_swept_volume_clearance_validated:
    verified: true
    verified_utc: "2026-09-05T23:38:01Z"
    evidence: "parked: rotation_clearance_m 0.284 with a side post at 0.30 m (rear unchanged 0.488), 0.480 at 0.50 m; run b6a52308: spin rejected at 0.336/0.340 m; run 3936c2b5: spin executed at 0.588/0.599 m in 1.9 s, no /collision_event"
  stop_command_and_nav2_cancel_order_validated:
    verified: true
    verified_utc: "2026-09-05T23:13:59Z"
    evidence: "run 804f94b7: recovery_safe_stop cancel_task -> zero_cmd_vel (0.063 s), measured_stop.stopped true in 0.20 s; /cmd_vel zeros after the cancel then Nav2 BackUp -0.05 m/s; 16/16 live safe stops across 4 runs in that order, all stopped within 2 s"
  every_guard_rejection_logged:
    verified: true
    verified_utc: "2026-09-05T23:13:59Z"
    evidence: "runs 804f94b7, 3936c2b5, b6a52308, 34546989, b482c289: 22 warnings, 22 recovery_guard_rejection events, rejected_actions equal to the ineligible set of guard_results for every decision"
```

## 7. Unverified or partially verified, and why

1. Item 1, AMCL re-convergence: the covariance trace falls after `/reinitialize_global_localization`
   but did not return below 0.5 within the episode (see item 1). Whether that satisfies
   "AMCL re-converges" is the researcher's call; the service availability and the guard gating
   are proven.
2. Item 3, a spin executed at exactly 0.50 m clearance: executed spins were at 0.59-0.60 m,
   rejected at 0.34 m; the 0.50 m case exists only as the parked measurement (0.48, which the
   guard would admit).
3. Items 2-3 measurement bias: raw beam minima read ~1.5-2 cm short of the geometric distance
   on the LDS; conservative for the guard, not footprint-corrected (known limitation, doc section 8).
4. The recommendation-only `recommendation_only` message on `/research2/recovery_decisions`
   was inferred from the sidecar (`execution_status: refused_live_execution_locked`) for run
   a244fab2; the topic echo was captured on the live run 804f94b7 (`message: executed`, twice).
5. Mission outcomes after live recoveries: runs 3936c2b5, 34546989 and b482c289 ended in
   `timeout` because the third alarm exhausted `maximum_repeated_recoveries: 2` and
   `request_assistance` never resumes; the runner then times out after a 30 s grace
   (`RESUMED_MISSION_GRACE_SECONDS`). That is the designed behaviour, but it means the pilot's
   outcome tables will record `timeout` for every episode whose alarms outlast the budget; the
   alarm cooldown (10 s) makes three alarms in 25 s common on high-severity faults.
6. `Nav2LiveCommander` and the two nodes remain marked UNTESTED LIVE in their docstrings until
   the researcher accepts this draft; the live path has now run on 5 episodes without a
   collision, guard violation, failed step or crash.
