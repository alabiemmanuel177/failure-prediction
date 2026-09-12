# Deviations and disclosures

Research 2, autonomous robot failure prediction and recovery. Every item below is
also a chained record in `logs/research-log.jsonl` (verify with
`python3 scripts/research_log.py verify`). Amendments are the YAML files
`configs/protocol_amendment_1.x.yaml`, each approved in the researcher's name before
the work it admits. Nothing in this list changes a preregistered estimator; the
preregistered results are reported exactly as evaluated, including every negative
component.

## 1. Protocol amendments (all approved before the admitted work)

| Amendment | Date | Scope | Effect on the study |
|---|---|---|---|
| PA-2026-09-03-01 | 3 Sept | Human-review admission | The completed 20/20 primary-researcher annotation audit and prospective threshold approval satisfy the model-development gate. |
| PA-2026-09-03-02 | 3 Sept | Confirmatory design route count | Six frozen routes per Research 1 held-out map instead of eight: held-out 1,008 episodes, severity stress 1,512, paired recovery 1,512. |
| PA-2026-09-03-03 | 3 Sept | Fitting-pool composition | The 647 causal-adapter Research 1 development bags are excluded from fitting; the pool is 648 + 1,212 + 504 = 2,364 Research 2 episodes. |
| PA-2026-09-03-04 | 3 Sept | Execution plan and recovery policy set | Six concurrent simulators admitted behind a pre-declared shift check; severity stress retained before completion; recovery policies R0, R2, R3 (R1 dropped); R3 selector fitted on the validation split of the recovery pilot. |
| PA-2026-09-04-01 | 4 Sept | Concurrency topology | After the first shift check failed on GPU-perception latency: S3 episodes at most one at a time. |
| PA-2026-09-04-02 | 4 Sept | Phased execution | After both shift checks failed on S3: every remaining campaign runs S0 six-wide, then S3 strictly alone (the sequential reference condition). The absolute false-alert criterion of the checks was unattainable even sequentially on development maps; the phased admission uses non-inferiority to the sequential reference (tolerance 0.05 alerts per mission). Both admission records are retained. |
| PA-2026-09-04-03 | 4 Sept | Seed selection rule | The frozen seed of each learned predictor is chosen on validation by event recall at the frozen false-alert budget, AUPRC as tie-break; every seed is reported. |
| PA-2026-09-10-01 | 10 Sept | Re-execution of the paired recovery campaign | See section 4. |

## 2. Model freeze and protected data

- G5 freeze on 4 September: P3 causal TCN, seed 20260904, Platt calibration, validation
  threshold 0.2351 at 0.10 false alerts per clean mission with 2-of-3 persistence and a
  10 s cooldown. The held-out maps were unread at the freeze; the protected split was
  assigned afterwards from the Research 1 test route manifests.
- Secondary predictors P4 (GRU) and P5 (compact transformer) carry their own frozen
  validation thresholds. During held-out scoring they were first alarmed with the primary
  threshold by mistake; the tables were superseded before evaluation and the correction
  is logged.
- The analysis-only oracle P6 refused to score protected episodes, as its contract
  requires, and is excluded from held-out scoring.
- One disclosure of aggregate protected outcomes before evaluation: while diagnosing why
  the pilot summariser refused the held-out campaign, a scratch summary displayed pooled
  attempt terminal-state counts across all held-out conditions. No per-model or
  per-condition outcome was seen before the frozen evaluation.

## 3. Execution incidents and how each was handled

All campaigns are exact-once per design key. A pre-goal infrastructure failure (the
simulator or Nav2 did not come up, no goal dispatched, no seed consumed) is declared as
one same-cell replacement with a distinct key and run id, executed sequentially after
all phases; the original attempt is retained and excluded. One replacement per cell is
allowed. Replacement counts per campaign:

| Campaign | Expected | Usable | Invalid attempts | Replacements | Notes |
|---|---|---|---|---|---|
| balanced_pilot_v1 | 648 | 648 | 2 | 2 | |
| balanced_validation_v1 | 324 | 324 | 4 | 4 | one post-validation payload loss, one teardown-isolation loss declared |
| targeted_development_v1 | 1,212 | 1,212 | 0 | 0 | one attempt killed by a host session termination before any record; declared and re-run |
| development_supplement_v1 | 504 | 504 | 0 | 0 | |
| held_out_map_v1 | 1,008 | 1,008 | 14 | 14 | 13 startup, 1 undelivered treatment |
| severity_stress_v1 | 1,512 | 1,512 | 12 | 12 | all startup |
| recovery_pilot_v1 | 882 | 881 | 34 | 33 | 27 startup, 5 undelivered treatment, 1 unrecorded required topic; one cell excluded (below) |
| paired_recovery_v1 | 1,512 | 1,510 | 16 | 16 | superseded (section 4) |
| paired_recovery_v2 | 1,512 | 1,512 | 14 | 14 | 11 startup, 3 unrecorded perception topic |

Incidents:

- Host GPU wedges (4, 5 and 6 September, four times) and stuck reboots. Root cause: the
  workstation's integrated GPU was selected by Gazebo's headless rendering and hung; the
  researcher disabled it in firmware on 6 September. Every attempt in flight during a
  wedge was a pre-goal invalid and was replaced. The plan may reboot the host on a wedge
  by researcher authorisation (logged).
- Host resets on 6 September 13:19 UTC, 7 September 05:37 UTC, 7 September 22:47 UTC
  (deliberate, drive installation) and 8 September 16:40 UTC. Two show a discrete-GPU
  display-driver signature; the researcher added `amdgpu.gfxoff=0` on 8 September, after
  which no reset occurred. Files half-written at a reset were repaired from intact
  payloads with checksums verified (two rosbag metadata files, one research-log line)
  and logged; 129 attempt directories that never produced a summary were quarantined
  under `data/raw_unrecorded_attempts/` and entered no ledger or dataset.
- After the 05:37 UTC reset the dispatcher crashed on the corrupted research-log line at
  every start and orphaned one unrecorded attempt of one severity key 108 times; all
  108 are quarantined, the key was attempted once and recorded, and the dispatcher now
  finalises completed waves before claiming work.
- Concurrent Nav2 bring-up on validation map val_01 failed at about 40 percent under
  six workers (a timing race absent when sequential); every failure was pre-goal and
  replaced sequentially. This cost time, not data.
- On 8 September the researcher's Research 1 follow-up work added files under
  directories pinned by the Research 2 platform boundary; three paired attempts were
  refused before launch and the campaign halted for 14 hours. The commits added 64 files
  and modified none; the boundary check now compares pinned blobs individually and still
  refuses any modified or deleted pinned file. The three attempts were replaced.
- Recovery pilot cell val_01-val_01_r5-blockage-r0, policy R0: the original attempt and
  its single replacement both timed out before the injection could occur. The cell's R0
  reference is excluded (documented in `data/manifests/recovery_pilot_exclusions_v1.yaml`);
  its other policies keep their episodes but contribute no regret rows.
- Two attempts of the pilot and three of paired v2 reached a terminal state but failed
  post-episode artifact validation (undelivered injection or a required topic with no
  recorded messages); each was replaced under the treatment-delivery or
  post-goal-validation policy. The summariser and outcome-table builder derive attempt
  validity from the campaign ledgers, not from summary terminal states; an earlier pilot
  completion record that counted such attempts as usable was superseded and the
  selector was refitted from the corrected table before any paired episode ran.

## 4. The paired recovery campaign was executed twice

`paired_recovery_v1` (10 September) ran the online failure monitor and the recovery
manager in recommendation-only mode: the campaign manifest omitted
`execution_policy.recovery_live_execution: true`, which the pilot manifest carried and
whose validator required it. The manager decided actions on 217 to 226 warnings per arm
and executed none. The R2 and R3 arms therefore delivered no treatment, and the H6
analysis computed from v1 (completion difference +0.008 [-0.022, +0.040]) compares R0
with itself. It is retained under `reports/recovery/_v1_recovery_execution_not_requested/`
and is not a result.

Under PA-2026-09-10-01 the campaign was re-executed as `paired_recovery_v2` with an
identical design (maps, routes, conditions, replicates, seeds, policies, pairing), the
same frozen predictor, calibrator, threshold, guard evidence and R3 selector, and live
execution requested. In v2, R3 executed 115 recovery actions in 504 episodes. H6 is
evaluated once, on v2, by the unchanged preregistered analysis. The v1 R0 arm is a
baseline replication with the same seeds; the v1 R2 and R3 arms are monitor-only shadow
runs. The v1 aggregates were displayed during diagnosis; because those arms delivered no
treatment, the aggregates carry no information about the effect of recovery.

## 5. Analysis-tooling changes after the freeze

None changes a preregistered estimator. Each is logged with the reason.

- The leave-one-family-out trainer refused the planner_oscillation fold because its
  no-op guard was evaluated per training manifest; the guard now applies to the pooled
  fitting set. No fitted model was touched.
- The reporting layer analyses mission timeouts separately, matching the confirmatory
  evaluator, and takes the H1 statement from the confirmatory report rather than
  recomputing it.
- Held-out feature ablations (H4) are scored on the identical held-out episodes under
  the primary frozen rule, as the protocol specifies; the validation-only ablation
  report remains exploratory.
- The dispatcher gained a drain marker, staggered slot start-up, a forced-reboot
  escalation, and campaign-level recovery-policy expansion; the summariser and outcome
  builder read ledger validity and merge declared replacements.

## 6. Results disclosed against the preregistered criteria

| Hypothesis | Preregistered criterion | Outcome |
|---|---|---|
| H1 | paired P3 minus P1 event recall, lower interval bound above zero | +0.197 [-0.000, +0.447]; not supported |
| H2 | median useful lead time of detected events at least 3 s | 3.8 s [1.7, 6.1]; supported |
| H3 | calibration lowers Brier and ECE on validation | supported |
| H4 | planner and localisation features materially reduce early warning | no decline on held-out; not supported |
| H5 | P3 above P1 in at least five of seven unseen-family folds | four of seven; not supported. The planner_oscillation fold has no feasible operating point within the false-alert budget on validation, so its held-out recall is zero by construction. |
| H6 | R3 improves completion over R0 without more collisions | -0.018 [-0.077, +0.032], zero collisions; not supported |

Exploratory and validation-only diagnostics (`reports/exploratory/`) indicate that the
preregistered warning window opens after the injected fault has ended in every family,
that the false-alert budget estimated from 72 clean validation missions is unstable,
and that three maps per split cannot separate transfer from map-to-map variation.
These are design limitations, not re-analyses.

## 7. Schedule

The researcher's target of 5 September was missed; the confirmatory work ended on
12 September. The delay is attributable to the incidents in section 3 and to the
re-execution in section 4.
