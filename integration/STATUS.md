# Shared-platform integration status

Last verified: 3 September 2026

## Boundary

Research 2 reuses the Research 1 G6 scientific platform at the content boundary recorded
in `integration/research1.lock.yaml`. Descendant operations-only commits are accepted only
when every locked Git object remains exact. It writes no files to Research 1.
Development and validation maps are permitted. The Research 1 `test` split is rejected
by the Research 2 runner before simulator launch.

## Verified

- Research 1 and Research 2 overlays source together through `scripts/env_research2.sh`.
- The `failure_experiment` ROS 2 package builds with `colcon`.
- Raw camera, scan, and odometry streams are routed through `/research2/raw/*` and
  restored onto the standard topics consumed by Research 1/Nav2.
- Clean proxy smoke on `dev_00` sustained approximately 9–10 Hz camera/scan and 27.7 Hz odometry.
- A complete clean `S0/dev_00_r0` episode succeeded with a retained MCAP.
- A LiDAR-dropout smoke produced a clean prefix, eligible onset, exact 10% invalid-beam
  treatment, injection end, and terminal event. Raw LiDAR remained unchanged.
- The artifact validator passed the clean and LiDAR smokes.
- Successful and failed bags follow the same retention path; no success deletion exists
  in the Research 2 recorder.
- Topic-health counts are checkpointed every second and finalized by the episode runner.
- Research 1 G0-G7 passed, including the complete 3,648-episode pilot, 6,840-episode
  confirmatory archive and independent release reproduction. Its campaign service is inactive.
- The G6 content boundary passes at Research 1 release head
  `a6d5bf49f549c88317a4db5b6488ce451aeae88a`; post-G6 changes do
  not alter shared simulator, navigation, logging, perception, map or route objects.
- Development and validation map/route identifiers are assigned before extraction;
  protected Research 2 routes remain deliberately unassigned.
- A 25-attempt development-only campaign is frozen for four retained controls and all
  seven fault families at three severities. Its runner refuses execution while the
  Research 1 protected campaign is active.
- The offline pipeline now covers MCAP-to-scalar adapters, receipt-time causal
  resampling, explicit age/missingness channels, trailing-window features, development-
  only normalization, leakage denial, transparent rules, M-of-N/cooldown alarms,
  event-level metrics, dependency-free SVG traces and replay-only recovery guards.
- A synthetic engineering smoke regenerates an end-to-end metric artifact and is marked
  structurally as non-research evidence.
- Research 2 now uses dedicated ROS domain 52. Shared domain 42 admitted a foreign
  simulation clock during treatment supplements; every affected pre-injection attempt
  is retained and excluded.
- The definitive v3 campaign retained 4 clean controls and 21 causally activated fault
  episodes. A 20-episode audit sample was selected before outcome review and exported
  as review-pending annotation files.
- All 20 frozen audit bags were rechecked with persistence-based localisation-loss and
  immobilisation derivation. No earlier operational event changed an automatic primary
  candidate (20/20 agreement); this remains engineering evidence, not human review.
- Direct or supplemental treatment reconstruction passes all 21 family/severity cells;
  the evidence matrix is `reports/integrity/fault_integrity_gate_v1.yaml`.
- Route-relative placement for environment faults was verified on `dev_01_r0`: dynamic
  blockage and planner oscillation both produced valid terminal-failure bags, and their
  injection coordinates were reconstructed exactly from the last pre-onset global plan.
- The balanced pilot is preregistered as 648 development-only episodes (12 map-route
  pairs, six replicates, two clean controls and seven medium-severity fault conditions).
  Raw collection is extraction-gated and runs in balanced 18-episode waves while human
  review independently blocks model fitting. A 100 GiB free-space reserve prevents the
  campaign from exhausting the workstation volume.
- The `recovery_manager` ROS 2 package now converts warning-state requests into
  guard-constrained R1/R2/R3 decisions. A domain-52 transport smoke returned a guarded
  stop/replan recommendation with `execution_performed=false`; setting live execution
  true is rejected at startup until Nav2 cancellation ordering and clearance evidence
  are frozen.
- The balanced development campaign is complete at 648/648 valid scientific episodes:
  125 terminal events and 523 non-events. Two pre-goal infrastructure-invalid attempts
  are retained and each has exactly one same-cell, same-seed linked replacement. The
  hash-addressed inventory validates all 648 source summaries, event streams, topic-
  health records and retained bags. It contains 53.20 GiB and 41,701.374 seconds of
  telemetry over 54 `full_v1`, 42 `compact_v1`, and 552 `compact_v2` recordings.
- A protected-outcome-free Research 1 reuse audit found 825 development bags and 364
  validation bags with the core temporal topic contract. Those bags remain only
  structurally eligible as a separately audited natural/control source; episode
  aggregates cannot be used as temporal windows. Only development bags may later enter
  fitting, validation remains selection-only, and all Research 1 test bags remain
  forbidden. The minimized v2 structural catalog contains exactly those 1,189 rows,
  exports only approved-topic counts, and remains explicitly non-admitted.
- Recovery guards were exhaustively checked over 3,456 discrete states and 51,840 R1,
  R2 and R3 policy decisions with zero violations. Live recovery execution remains
  locked; paired recovery is intentionally post-model.
- The 324-episode validation-only raw campaign is complete over three maps, six routes,
  six replicates, two clean controls and seven fault families. Its hash-addressed
  inventory and every retained payload validate. It may not select models, calibration
  or thresholds before human training admission.
- The post-pilot size decision preregisters 1,212 targeted development additions in
  complete 12-route blocks. Together with the 648 pilot episodes and at most 825
  admissible Research 1 development bags, it yields a pre-supplement fitting ceiling of
  2,685. The split-corrected plan therefore requires at least 324 more route-balanced
  development episodes after the human gate and causal-adapter audit. Collection remains
  sequential because concurrent simulators would change compute-load and timing
  conditions that are candidate health signals; model-training jobs can be parallelised
  later without contaminating episode generation. The targeted campaign is now running;
  its authoritative progress is recorded in
  `reports/status/targeted_development_monitor.yaml`.
- Confirmatory readiness now requires a hash-addressed model, normalisation,
  calibration, alarm-policy and analysis freeze in addition to assigned protected maps
  and a frozen threshold. The protected split remains unassigned and inaccessible.

- Offline derivation covers whole inventories: 648 development and 324 validation
  episodes have fitting sequences and all-decisions artifacts under `data/derived/`,
  regenerated after the goal-distance frame correction with labels, annotations,
  telemetry and causal features byte-identical to the first derivation.
- The Research 1 causal adapter audit admitted 647/825 development bags (event times
  exact for collisions, bounded-upper for Nav2 result classes; wall-clock recordings are
  mapped to simulation time per episode). Adapted episodes derive through the unchanged
  per-episode pipeline into `research1_development_v1-development-647`.
- The 504-episode development supplement is preregistered and validated by the runner
  (`campaign_kind: development_supplement`); it requires the finalised targeted
  inventory and starts automatically after it.
- P2–P6 predictors, the training/prediction CLIs, calibration selection, budget
  threshold selection with capped candidate sweeps, predictor comparison, alarm-policy
  validation, ablations and the model-freeze writer exist and are tested. A preliminary
  validation-only pass on the 648/324 pool exists as a rehearsal; nothing is frozen.
- Confirmatory tooling (post-freeze protected split assignment, 960/1512 manifests,
  gated runner path, held-out and unseen-family evaluation, natural-failure audit,
  H1–H5 analysis) and recovery/release tooling (cost function, selector training,
  paired campaign manifest, H6 analysis, guarded live-execution path marked untested
  live, figures, tables, release report, reproduction, storyboard) are implemented and
  fail closed before the freeze.
- Protocol Amendment PA-2026-09-03-02 (`configs/protocol_amendment_1.2.yaml`) fixes six
  routes per held-out map, as the content-locked Research 1 platform provides, and keeps
  the preregistered minimum sizes by adding seeds: held-out 3 x 6 x 7 x 8 = 1,008,
  severity stress 3 x 6 x 4 x 21 = 1,512, paired recovery 3 x 6 x 4 x 7 x 4 policies
  = 2,016 (1,008 for R0 versus R3). Decided before the freeze without inspecting
  protected maps or outcomes.

- Fitting-pool rehearsal on Research 2-only validation (single seed, nothing frozen):
  adding the 647 outcome-selected Research 1 episodes lowered the primary TCN's event
  recall at the budget from 0.290 to 0.161 (AUPRC 0.195 to 0.177, median lead 5.4 s to
  3.5 s) and the transformer's from 0.323 to 0.113, while the GRU rose from 0.210 to
  0.274 (`reports/model_selection/pass_comparison_preliminary_v1_vs_preliminary_v2_with_research1.yaml`).
  Protocol Amendment PA-2026-09-03-03 excludes Research 1 from the frozen fitting pool
  (648 + 1,212 + 504 = 2,364 Research 2 episodes) and keeps it as a separately reported
  natural-failure and domain-shift audit set.
- The workstation's user processes were killed at about 14:14 UTC on 3 September and
  the host rebooted at 17:24 UTC; the targeted ledger ended cleanly at 966/1,212 and the
  exact-once runner, watcher and monitor were resumed without retry or replacement.

- Protocol Amendment PA-2026-09-03-04 (deadline 5 September): six concurrent simulators
  on the Research 1-qualified domains 60-65 are admitted for the remaining campaigns
  only if a pre-declared 36-episode concurrency shift check on development maps passes;
  otherwise every campaign stays sequential. Severity stress and the supplement are
  retained; the paired recovery campaign compares R0, R2 and R3 (R1 deferred); the R3
  selector trains on a post-freeze validation-map recovery pilot.

- A six-worker campaign dispatcher (`scripts/run_campaign_parallel.py`) reuses the
  sequential runner's expansion, ledger, validation and reports; the sequential path is
  byte-identical. The 36-episode concurrency shift check and its pre-declared policy
  (`configs/concurrency_shift_policy.yaml`) gate every parallel campaign.
- Recovery plumbing exists end to end: policy ids threaded from manifest to launch, an
  online `failure_monitor` node whose windows equal the offline decision artifacts
  exactly (347/347 decisions, zero difference) and whose scores equal the offline
  predictor, a two-phase safe-stop-then-guard recovery node, a validation-map recovery
  pilot (882 episodes) that trains the R3 selector, and outcome/cost table builders.
  The ROS side has never run in a simulator; `docs/recovery-live-bringup.md` lists the
  ordered post-freeze smoke sequence and the five live-evidence items.
- The overnight plan (`scripts/run_deadline_night_plan.py`) runs the shift check, the
  supplement and the final model pass unattended after the targeted campaign.

- Concurrency shift check v1 (36 episodes, six workers) FAILED on 4 September:
  clean S3 missions produced 15 false alerts in 12 missions (0.71 per clean mission
  overall versus 0.18 sequential) and `inference_latency_ms` shifted beyond 0.5 SMD in
  S3 episodes; S0 episodes matched sequential conditions. Per PA-2026-09-03-04 the
  supplement fell back to sequential execution. An S3-serialised topology (six workers,
  at most one GPU-perception episode at a time) is being prepared with a second
  pre-declared check; it is not admitted until that check passes.

- Concurrency shift check v2 (six workers, at most one S3 episode at a time) also FAILED
  on 4 September: pooled clean-mission false alerts 0.25 versus 0.18 sequential; clean S0
  missions were within budget and below the sequential reference, but clean S3 missions
  still exceeded it and `inference_latency_ms` still shifted beyond 0.5 SMD. GPU
  contention from concurrent Gazebo rendering, not only from concurrent perception
  nodes, is implicated. Per PA-2026-09-04-01 the remaining campaigns stay sequential
  unless the researcher amends the plan; both check reports are retained.

- Protocol Amendment PA-2026-09-04-02 admits phased execution (S0 episodes six-wide,
  then S3 episodes strictly alone) on the derived admission record
  `reports/integrity/concurrency_phased_admission_v2.yaml` (non-inferiority to the
  sequential reference; the failed absolute-criterion record v1 is retained and the
  criterion change disclosed). The dispatcher enforces `--systems` phases and per-system
  caps; every ledger row records phase, slot, domain, partition and caps.
- The final validation pass on the 2,364-episode pool (seed 20260903) gave P3 recall
  0.177, P4 0.306 and P5 0.290 at 0.097 false alerts per clean mission (AUPRC 0.208,
  0.211, 0.231). Two further seeds are trained and the seed is chosen by the
  preregistered validation-AUPRC rule with every seed reported; P3 stays the
  preregistered primary and P4/P5 are frozen as secondary held-out comparisons.

- G5 model freeze signed on 4 September 17:22 UTC: primary P3 causal TCN seed 20260904
  (Platt calibration, alarm threshold 0.2351, validation recall 0.339 at 0.097 false
  alerts per clean mission, median lead 5.7 s); P4 (seed 20260905) and P5 (seed 20260904)
  frozen as secondary comparisons; P1 thresholds frozen; seed rule per PA-2026-09-04-03.
- The protected split is assigned (three Research 1 test maps, six routes each), the
  held-out (1,008) and severity-stress (1,512) manifests and the recovery pilot (882) and
  paired recovery (1,512) manifests are written, and the confirmatory readiness gate
  passes. The held-out campaign runs under the phased topology; derivation, scoring,
  confirmatory evaluation and unseen-family folds follow unattended
  (`scripts/run_post_freeze_plan.py`). Severity stress runs after the recovery bring-up.

- CONFIRMATORY HELD-OUT RESULT (one-time, 5 September, 1,008 episodes on three unseen
  maps, 157 primary events excluding 6 mission timeouts analysed separately): P3 event
  recall 0.229 versus P1 0.032 at the validation-frozen threshold; paired difference
  +0.198 with 95 % hierarchical bootstrap interval [-0.0001, 0.447], so H1 is NOT
  supported under the preregistered criterion (the interval touches zero). Median useful
  lead of detected events 3.8 s (interval [1.7, 6.1] s; point estimate meets the 3 s
  minimum, the interval does not). Calibration reduced held-out Brier 0.034 to 0.016 and
  ECE 0.041 to 0.008 (H3 supported). Held-out false alerts per clean mission were 0.18
  for P3 and 0.47 for P1 against the 0.10 validation budget, i.e. the unseen maps shift
  the alarm burden for every model. Exploratory: P5 recall 0.268 (difference to P1
  [0.043, 0.421]) at 0.25 false alerts per clean mission; P4 recall 0.198 at 0.064.
  H4 awaits the post-freeze ablation runs.

Smoke outputs live under `/tmp` and are not research data.

The first `live_integrity_v1` execution on 30 August is quarantined as engineering
evidence: one pre-simulation import failure and 24 artifact-invalid false arrivals
revealed missing Nav2 wall-to-simulation-time activation and misaligned recorder-health
start intervals. None of its episodes may enter training, validation or the manual-label
agreement set.

## Fault qualification matrix

| Family | Deterministic implementation | Unit tested | Live integrity smoke | Frozen for dataset |
|---|---:|---:|---:|---:|
| Camera occlusion | Yes | Yes | 3/3 exact reconstruction | Yes |
| LiDAR dropout | Yes | Yes | 3/3 exact reconstruction | Yes |
| Wheel slip / odometry bias | Yes | Yes | 3/3 raw/deployed checks | Yes |
| Localisation perturbation | Yes | Boundary tested | 3/3 recorded reset checks | Yes |
| Dynamic blockage | Yes, delayed entity spawn | Config tested | 3/3 successful spawn checks | Yes |
| Planner oscillation | Yes, delayed symmetric entities | Config tested | 3/3 successful spawn checks | Yes |
| Semantic corruption | Yes, dual risk-grid proxy | Yes | 3/3 exact reconstruction | Yes |

All severities are frozen for development and validation extraction. This engineering
gate complements the completed primary human causal-label review.

## Remaining before model-fitting admission

1. Retain the completed primary-reviewer 20/20 exact agreement under
   `configs/protocol_amendment_1.1.yaml`.
2. Preserve the frozen development and validation route assignments during extraction.
3. Complete and inventory the preregistered targeted development campaign.

Immutable raw development collection is complete and validation collection is permitted
under the extraction gate. Primary human review is complete; model fitting becomes
admissible when the remaining machine/data gates pass. Protected-map execution remains
blocked until the model, calibration, alarm policy and analysis are frozen. Independent
review remains an optional pre-publication improvement and no inter-rater claim is made.
