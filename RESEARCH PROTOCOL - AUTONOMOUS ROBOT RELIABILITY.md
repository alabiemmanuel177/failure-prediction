RESEARCH PROTOCOL  |  AUTONOMOUS ROBOT RELIABILITY

ROBOTICS RESEARCH PROGRAMME




Early Failure Prediction and Recovery
for Mobile Robot Navigation

A complete research protocol and implementation manual

CORE CLAIM  A navigation robot can use its recent sensor and internal-state history to forecast impending failure, then select a low-cost recovery before the mission terminates. The scientific test is whether this remains true for maps and failure families excluded from training.



Field

Value

Researcher

Emmanuel Alabi Olasubomi

Protocol version

1.0

Prepared

24 August 2026

Dependency

Research 1 simulator, navigation variants and episode logs

Primary platform

ROS 2 Jazzy, Gazebo Harmonic, Nav2

Target duration

10 additional weeks after the shared platform is stable

Document status

Start-ready working protocol

Confidentiality

Private admissions and research planning document


Page 

RESEARCH PROTOCOL  |  AUTONOMOUS ROBOT RELIABILITY

How to use this protocol

Research 2 should begin only after Research 1 can generate repeatable navigation episodes with complete provenance. Build the offline dataset and causal labeling pipeline before any temporal model. A model that accidentally receives post-failure state, injection metadata or future samples is invalid even if its scores are excellent.

FIRST DECISION  Use failure forecasting as the primary task and recovery selection as the downstream systems test. Do not let the project become a generic anomaly-detection benchmark or an unstructured collection of ROS errors.

The first 72 hours

Day 1: Freeze the operational failure definitions and list every ROS topic, feature and event source. Mark forbidden leakage fields explicitly.

Day 2: Replay ten Research 1 bags, align timestamps, generate candidate windows and plot at least three failures from precursor to termination.

Day 3: Implement a threshold-rule baseline, an event-level evaluator and one recovery action in replay or simulation. Produce the first reproducible metric table.

Non-negotiable controls

All splits are made by complete episode, map and route before windows are extracted.

Injection commands, injected-fault parameters, terminal result codes and post-event samples are forbidden model inputs.

Thresholds, window length, warning horizon and recovery costs are selected on development and validation data only.

The held-out map set and unseen-failure folds remain untouched until the model and analysis plan are frozen.

False alarms and unnecessary recovery cost are treated as primary outcomes, not footnotes.


Page 

RESEARCH PROTOCOL  |  AUTONOMOUS ROBOT RELIABILITY

1. Executive research specification

Element

Protocol decision

Working title

Early Failure Prediction and Recovery for Mobile Robot Navigation

Primary question

Can recent multimodal robot telemetry predict a mission-ending navigation failure early enough for a recovery action to improve mission completion?

Primary contribution

A causal, event-level benchmark for early failure warning under held-out maps and unseen failure families.

Systems contribution

A calibrated warning policy connected to a cost-sensitive recovery selector using Nav2 behaviors.

Unit of independence

Navigation episode. Windows from one episode never cross data splits.

Primary endpoint

Event recall at a validation-fixed false-alarm budget, with useful warning lead time.

Downstream endpoint

Paired improvement in mission completion after prediction-triggered recovery.

Target artefacts

Preprint, code, dataset manifest, model cards, experiment logs, result figures and a 90-second demonstration.

Contribution ladder

Minimum publishable: A rigorous dataset, causal labels and fair comparison between threshold rules and temporal predictors on held-out maps.

Strong contribution: Reliable early warning at a controlled false-alarm rate, including leave-one-failure-family-out tests.

Best credible contribution: Warnings improve mission completion through recovery selection while unnecessary interventions remain bounded.

Not sufficient: High window-level accuracy from randomly shuffled telemetry or a demo that predicts an injected fault label directly.

2. Research questions and hypotheses

Primary research question

Can a model using only observations available up to the current time predict a mission-ending navigation failure at least H seconds before it occurs, while respecting a fixed false-alarm budget?

Secondary questions

Which signal groups contribute most: raw sensors, localisation health, planner-controller state, perception confidence or motion history?

Does calibration remain reliable across held-out maps and severity levels?

Can a predictor trained without one failure family warn about that unseen family?

Does a learned temporal model outperform transparent threshold rules after false-alarm rates are matched?

Can the warning improve mission outcomes when connected to recovery, rather than merely predict an eventual failure offline?

Preregistered hypotheses

ID

Hypothesis

Primary test

H1

The primary temporal predictor has higher event recall than threshold rules at the same validation-fixed false alarms per mission.

Paired held-out-map comparison with hierarchical bootstrap confidence interval.

H2

The predictor provides a median useful lead time of at least 3 seconds for detected failures.

Event-level median and 95% interval.

H3

Calibration reduces Brier score and ECE on validation data without reducing event recall beyond the declared tolerance.

Before-versus-after calibration comparison.

H4

Removing planner and localisation health features causes a material decline in early-warning performance.

Feature-group ablation.

H5

Leave-one-failure-family-out performance remains above the threshold-rule baseline for at least five of seven families.

Seven prespecified unseen-family folds.

H6

Prediction-triggered recovery improves mission completion over default Nav2 recovery without increasing collision rate.

Paired recovery trial by map, route, seed and fault.

3. Scope, novelty and non-goals

In scope

Simulated differential-drive mobile robot using the frozen Research 1 navigation stack.

Seven controlled failure families plus naturally occurring failures retained from clean runs.

Past-only multivariate telemetry windows, calibrated event-risk scores and explicit abstention or recovery decisions.

Known-failure evaluation, held-out-map evaluation and leave-one-family-out generalisation.

Offline predictor evaluation followed by closed-loop recovery trials.

Out of scope

Formal safety certification or a claim that prediction guarantees collision avoidance.

End-to-end replacement of Nav2 with a learned driving policy.

Training from the injection label or directly reading a simulator fault-control topic.

Physical-robot validation before the simulation evidence and action guards are stable.

Large language models, vision-language models or human assistance as the central method.

Claiming universal anomaly detection from a benchmark constructed on one robot family.

NOVELTY TEST  The work is research-shaped only if the alert is genuinely early, false alarms are controlled, test episodes are protected and unseen failures are evaluated. Model complexity by itself is not a contribution.

4. Operational definitions and causal labels

Terminal failure event

An episode has a terminal failure at time t_f when the first prespecified event occurs. A timeout without goal completion is an event at the timeout boundary, but it must be analysed separately because it can have weak or diffuse precursors.

Failure class

Operational event rule

Confirmation source

Collision

Contact sensor event or minimum range below the frozen collision threshold while commanded speed is nonzero.

Gazebo contacts plus command velocity.

Navigation abort

NavigateToPose returns an aborted terminal state before reaching the goal.

Nav2 action result and error code.

Localisation loss

Pose error exceeds the frozen translation or yaw threshold continuously for K seconds.

Ground truth used only for labels, never predictor input.

Immobilisation

Commanded movement persists while measured progress stays below epsilon for K seconds.

Odometry, command velocity and progress checker.

Unsafe perception

A critical obstacle is missed within the protected stopping region and leads to emergency intervention.

Ground-truth semantic or depth label used only offline.

Mission timeout

Goal not reached within the route-specific time budget.

Experiment controller.

Warning-window construction



Figure 1. Causal event labeling for one navigation failure.

Symbol

Default

Meaning and tuning rule

W

5 seconds

Input history ending at decision time t. Tune only on validation data.

H

10 seconds

Maximum early-warning horizon before t_f.

delta

1 second

Too-late guard interval. Windows ending after t_f minus delta are excluded from early-warning scoring.

G

20 seconds

Negative guard distance from any fault onset or failure event.

stride

0.5 seconds

Decision interval for offline and closed-loop inference.

Label rules

A window ending at time t is positive if t lies in [t_f - H, t_f - delta].

A window is eligible negative only when it is at least G seconds from every labeled event and injected onset.

Windows after t_f, after episode termination or inside the too-late guard interval are excluded.

If multiple failure events occur, only the first terminal event defines the primary label. Later events may be retained for secondary analysis.

Natural failures from no-injection episodes receive the same event rules and are reported separately.

Ground truth and injection metadata are used to construct labels offline, then removed before feature export.

5. System architecture



Figure 2. End-to-end early-warning and recovery architecture.

Online decision loop

At 2 Hz, align the most recent W-second telemetry window using message timestamps.

Apply the training-time normalisation parameters and missing-data masks.

Predict event risk p(failure within H seconds) and optional failure-family probabilities.

Calibrate the risk score using validation data only.

Apply a threshold, persistence rule and cooldown to control alarm chatter.

When triggered, cancel or pause navigation safely, select an eligible recovery and log the action.

Resume, replan or request assistance according to the recovery outcome and action budget.

6. Platform, interfaces and instrumentation

Layer

Decision

Why it is used

Base platform

Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic and Nav2

Matches the supported Research 1 environment and prevents duplicated platform work.

Recording

rosbag2 with MCAP storage and simulation time

Timestamped replay, indexed storage and reproducible offline extraction.

Diagnostics

ROS diagnostic messages plus explicit research-health topics

Provides interpretable system status without treating diagnostics as ground truth.

Recovery execution

Nav2 Behavior Server, behavior tree nodes and Simple Commander API

Supports stop, backup, spin, wait, costmap clearing and replanning through defined interfaces.

Offline data

Parquet episode features plus immutable raw bags

Efficient model training without discarding raw evidence.

Models

PyTorch with scikit-learn baselines

One reproducible training stack with transparent classical comparisons.

Configuration

YAML manifests validated against a schema

Makes every fault, model, route, seed and threshold traceable.

ROS topics and derived signals

Signal group

Candidate inputs

Derived health indicators

Motion

/cmd_vel, /odom, acceleration estimate

Tracking error, stopped-while-commanded, jerk and progress slope.

Localisation

/amcl_pose, particle cloud summary, map to odom TF

Covariance trace, pose jump, TF age and innovation-like residuals.

Range sensing

/scan, minimum sector ranges, valid-return mask

Dropout fraction, near-obstacle trend and left-right imbalance.

Vision and semantics

Perception confidence summaries, entropy, class area and inference time

Confidence drift, temporal disagreement and missed-frame rate.

Planning

Global path, local path, planner/controller status and costmaps

Path age, replan rate, curvature, oscillation and cost trend.

System

Diagnostics, node heartbeat, CPU/GPU load and message rates

Stale-topic flags, latency, queue loss and compute saturation.

Goal context

Distance and bearing to goal, elapsed time and route budget

Remaining-distance slope and route-normalised progress.

LEAKAGE RULE  Do not expose Gazebo ground truth, the fault-injector command, severity, injection timestamp, terminal action result, recovery result or future-derived features to the predictor. Store them in a label-only namespace.

7. Failure injection benchmark

Family

Injection mechanism

Severity ladder

Typical precursor

Camera occlusion

Mask an increasing image region or suspend frames for bounded intervals.

20%, 50%, 80% masked or equivalent frame loss.

Confidence drift, entropy and missed-frame rate.

LiDAR dropout

Remove beams, sectors or complete scans with a seeded mask.

10%, 35%, 70% invalid returns.

Valid-return decline and range inconsistency.

Wheel slip

Alter wheel-ground friction or bias odometry relative to ground motion.

Small, medium and severe slip ratio.

Command-odometry mismatch and localisation covariance growth.

Localisation perturbation

Inject a bounded pose offset or degrade observation quality.

Translation and yaw offsets defined in the manifest.

Pose jump, TF inconsistency and path-tracking error.

Dynamic blockage

Introduce actors that close the selected corridor or repeatedly cross the local path.

Actor speed, density and blockage duration.

Cost increase, replanning and reduced progress.

Planner oscillation

Create symmetric local choices or alter a frozen controller parameter at episode start.

Increasing oscillation opportunity or parameter offset.

Command sign changes, curvature and repeated recoveries.

Semantic corruption

Swap or suppress selected semantic classes without altering geometry.

Class subset and corruption probability.

Confidence disagreement and unsafe semantic cost changes.

Injection controls

Every injection has a deterministic seed, planned onset rule, maximum duration and severity manifest.

Onset is sampled after a minimum clean prefix and before the route becomes trivially complete.

Eligibility checks prevent impossible or irrelevant injections, such as wheel slip while stationary.

Each faulted episode has a paired clean episode with the same map, route, robot configuration and seed.

Combined faults are exploratory only. Confirmatory experiments use one family per episode.

The benchmark records whether an injected fault caused no terminal failure. Those episodes are important negatives, not failed data generation.


Page 

RESEARCH PROTOCOL  |  AUTONOMOUS ROBOT RELIABILITY

8. Dataset design and protected splits

Episode hierarchy

The independent unit is the episode. A bag can create many overlapping windows, but those windows are correlated and must remain in one split. Statistical uncertainty is therefore estimated over maps, routes and episodes, never over individual windows as if they were independent.

Split

Maps and routes

Permitted use

Development

Six Research 1 development maps with designated routes

Feature engineering, code debugging, model fitting and exploratory plots.

Validation

Three maps and routes not used for fitting

Window length, model selection, calibration, threshold, persistence and recovery-cost tuning.

Held-out map test

Three protected maps with eight frozen routes each

One-time confirmatory evaluation after protocol freeze.

Unseen-family test

One failure family excluded from all fitting in each of seven folds

Tests whether warning transfers beyond known injected faults.

Natural-failure audit

No-injection episodes that fail under the operational rules

Separate ecological-validity analysis; never silently merge with injected events.

Campaign sizes

Stage

Minimum episodes

Purpose and stop rule

Smoke test

30

Ten clean and twenty faulted episodes. Stop until labels and replay are exact.

Balanced pilot

600

Estimate event prevalence, lead-time distribution, runtime and class balance.

Predictor dataset

At least 3,000 usable episodes from Research 1 plus targeted additions

Fit models only after group splits and manifest freeze.

Held-out confirmatory set

960

3 maps x 8 routes x 5 seeds x 8 conditions: clean plus seven fault families at frozen primary severity.

Severity stress set

1,512

3 maps x 8 routes x 3 seeds x 7 families x 3 severities.

Recovery trial

At least 1,008 paired episodes

3 maps x 8 routes x 3 seeds x 7 families x 2 recovery policies, increased after pilot if intervals are too wide.

COMPUTE CONTROL  The full matrix is an upper target, not permission to generate data blindly. Complete the pilot, estimate event prevalence and runtime, then revise the final count transparently before unblinding the test set.

Class balance and sampling

Train on all eligible positive windows but cap highly overlapping windows through stride or event-balanced sampling.

Sample negative windows by episode and operating phase so long clean episodes do not dominate.

Use class or event weights computed from training data only.

Report the natural event prevalence and the rebalanced training prevalence separately.

Never oversample before the split, because duplicate windows can cross partitions.

9. Model variants and baselines

ID

System

Role

P0

No-warning control

Quantifies outcomes under the default navigation and recovery stack.

P1

Transparent threshold rules

Primary interpretable baseline using validation-tuned rules on covariance, progress, scan health and oscillation.

P2

Reconstruction anomaly model

Unsupervised or normal-only baseline such as a compact autoencoder or USAD-style model.

P3

Temporal convolutional network

Primary learned predictor because it is causal, parallelisable and simpler than a transformer.

P4

GRU or LSTM

Recurrent temporal baseline with comparable parameter budget.

P5

Compact transformer encoder

Optional capacity baseline, included only if data volume and latency justify it.

P6

Oracle warning

Upper-bound analysis using event labels, never an operational method.

Primary model specification

Input: W seconds of synchronized numeric features, missingness masks and feature-age channels.

Encoder: causal dilated 1D convolutions with residual blocks and dropout.

Heads: binary event-within-H risk and optional failure-family classification.

Loss: weighted binary cross-entropy or focal loss selected on validation AUPRC, plus an optional family loss.

Calibration: temperature scaling, isotonic regression or Platt scaling selected before the test set.

Deployment: deterministic 2 Hz inference with measured end-to-end latency and a frozen normalisation bundle.

Mandatory ablations

Ablation

Question answered

No localisation features

Are pose covariance and TF health carrying most of the signal?

No planner-controller features

Does internal autonomy state add value beyond sensors?

No perception-confidence features

Is Research 1 uncertainty useful for failure prediction?

No raw-sensor summaries

Can the predictor operate mainly from system health and motion?

Single timestamp instead of temporal window

Is temporal modelling necessary?

No calibration

Does calibration change the warning-cost tradeoff?

No persistence or hysteresis

How much alert stability comes from the decision rule rather than the model?

10. Warning policy and recovery selector

Alarm policy

The model score is not the final decision. The operational warning policy declares an alert only when calibrated risk exceeds threshold tau for M of the last N decisions and no cooldown or recovery lockout is active. Tau, M, N and cooldown are selected on validation data to meet the declared false-alarm budget.

Policy element

Primary setting

Reason

False-alarm budget

At most 0.10 false alerts per clean mission

Makes the comparison operationally meaningful.

Persistence

2 of 3 decisions above threshold

Reduces single-frame spikes while preserving useful lead time.

Cooldown

10 seconds after a warning or recovery

Prevents repeated alerts for one event.

Minimum lead time

1 second before t_f

Alerts inside the guard interval are too late for the primary endpoint.

Abstention

Request assistance when no safe automated recovery is eligible

Avoids forcing an action under severe uncertainty.

Recovery actions

Action

Eligibility guard

Intended failure modes

Controlled stop

Always available unless stopping itself violates a frozen safety rule.

Any high-risk state; default first response.

Relocalise

Robot stopped, localisation health poor and a relocalisation procedure is available.

Pose jump, wheel slip and observation degradation.

Replan and clear costmaps

Robot stopped or moving safely; planning state is stale or blocked.

Dynamic blockage and corrupted local planning state.

Backup

Rear clearance verified and action distance bounded.

Local minima and frontal blockage.

Spin or active rescan

Rotation clearance verified.

Perception or localisation ambiguity.

Wait

No immediate collision risk and obstruction may be transient.

Crossing actors and short sensor dropout.

Request assistance

No eligible action, repeated failure or risk above hard threshold.

Unknown or severe failures.

Recovery-policy variants

ID

Policy

Comparison purpose

R0

Default Nav2 recovery tree with no predictor

Operational baseline.

R1

Predictor warning plus fixed conservative stop and replan

Separates warning value from learned action selection.

R2

Predictor plus rule-matched recovery by diagnosed signal group

Interpretable recovery baseline.

R3

Predictor plus cost-sensitive learned recovery selector

Proposed complete system.

R4

Oracle warning plus oracle eligible recovery

Upper bound and remaining-error decomposition.

Cost-sensitive action learning

For each warning state, the target action is the eligible recovery with the lowest observed or modelled cost under matched replay or repeated simulation. The cost function is frozen before test execution. Collision receives the largest penalty, followed by mission abort, failed recovery, excessive delay, path overhead and unnecessary intervention. Never train the selector on actions that would violate the eligibility guards.

SAFETY ORDER  The predictor may recommend an action, but the guard layer has final authority. If no action is eligible, the system stops and requests assistance. The study does not remove Nav2 collision checking.

11. Evaluation metrics

Metric

Definition

Reporting unit

Event recall

Fraction of failure events receiving at least one useful warning in [t_f - H, t_f - delta].

By family, severity and map.

False alerts per mission

Alerts in eligible negative regions divided by clean or non-event missions.

Mean, distribution and upper interval.

AUPRC and AUROC

Window-level discrimination reported as supporting metrics only.

Episode-grouped bootstrap.

Warning lead time

t_f minus the first useful warning time for detected events.

Median, quartiles and tail.

Time-dependent Brier score

Squared error for event-within-H risk at each decision.

Mean with grouped interval.

ECE and reliability curve

Agreement between predicted risk and observed event frequency.

Before and after calibration.

Alert burden

Total alerts, time under alert and repeated alerts per mission.

Clean and faulted missions.

Recovery success

Warning-triggered action followed by stable navigation and no terminal failure within the evaluation horizon.

By action and family.

Mission completion

Goal reached within route budget after any recovery.

Primary downstream outcome.

Recovery overhead

Added time, path length, energy proxy and intervention count.

Compared with paired baseline.

Inference latency

Feature preparation plus model and policy time.

Median, p95 and maximum.

Metric cautions

Do not present window accuracy as the main result. Long negative regions can make it misleading.

Do not grant full credit for an alert after the useful intervention window.

A detector that alarms continuously has high recall and is operationally useless; always pair recall with alert burden.

Report undetected failures when calculating lead time, rather than excluding them silently.

Use the same validation-fixed threshold for all confirmatory test conditions unless threshold adaptation is itself a declared method.

12. Statistical analysis plan

Primary predictor comparison

Compare P3 against P1 on the same held-out episodes at a threshold selected to meet the false-alarm budget on validation data. Estimate the paired difference in event recall using a hierarchical bootstrap that samples maps, then routes, then episodes. Report the point estimate, 95% confidence interval and full denominator.

Recovery comparison

Compare R3 against R0 and R2 using paired map-route-seed-fault episodes. Fit a mixed-effects logistic model for mission completion with policy as a fixed effect and map and route as random intercepts. If model assumptions or convergence fail, use the prespecified hierarchical paired bootstrap.

Secondary analyses

Event recall and false-alert burden by failure family, severity and map.

Lead-time survival-style curve showing the proportion of events warned by each time before failure.

Calibration curves and Brier decomposition for validation versus held-out maps.

Seven leave-one-family-out estimates with no pooled claim unless heterogeneity is reported.

Ablation deltas using identical test episodes and the primary frozen threshold rule.

Recovery action confusion, eligibility violations, regret relative to oracle and action-specific success.

Multiplicity and claim control

H1 and H6 are the two confirmatory claims. H2 to H5 are prespecified supporting hypotheses. All other subgroup, architecture and combined-fault results are exploratory and labeled accordingly. Report effect sizes and intervals; do not turn every metric into an independent significance claim.

Exclusions

Exclusion

Rule

Infrastructure failure

Exclude only when recording, simulator or experiment-control failure prevents the episode from representing any system variant.

Ineligible injection

Exclude when the manifest eligibility condition was not satisfied and the injection never began.

No induced terminal event

Retain as a non-event faulted episode.

Post-hoc visual dislike

Never an exclusion reason.

Recovery guard rejection

Retain and score as an abstention or unavailable action.

13. Data pipeline and schema

Immutable episode record

Field group

Required fields

Identity

run_id, parent_research1_run_id, timestamp, git_commit, container_digest and protocol_version.

Environment

map_id, route_id, start_pose, goal_pose, seed and simulator build.

Fault

family, severity, planned_onset, actual_onset, duration and eligibility result in label-only storage.

System

predictor_id, checkpoint_hash, feature_schema_hash, calibration_id, threshold_id and recovery_policy_id.

Outcome

first event class, t_f, success, collision, timeout, abort, recovery action and recovery result.

Integrity

bag path, message counts, missing-topic flags, extraction version and checksum.

Feature-table contract

One row per decision time with run_id and monotonic decision_index.

Every feature has value, age and missingness semantics documented in a versioned schema.

Normalisation statistics are fitted on training episodes and stored with the model.

Label columns and ground-truth fields are physically separated from deployable features.

A leakage audit fails the build if forbidden names or future timestamps enter the model matrix.

Raw bags are never overwritten after extraction; derived tables include input checksums.

Recommended recording strategy

Record the declared topic allowlist with rosbag2 and MCAP, using simulation time consistently. Write compact episode and event tables beside the bag. Snapshot mode can be used for debugging rare natural failures, but the confirmatory campaign should follow one frozen recording policy. Monitor recorder message loss and mark incomplete episodes before analysis.


Page 

RESEARCH PROTOCOL  |  AUTONOMOUS ROBOT RELIABILITY

14. Repository and reproducibility architecture

Use one repository or a clearly versioned Research 2 package inside the Research 1 monorepo. Shared simulation assets should not be copied into divergent forks.

Path

Purpose

configs/

Feature schemas, fault manifests, model settings, thresholds, costs and recovery guards.

ros_ws/src/failure_monitor/

Online synchronization, feature extraction, inference and alert publication.

ros_ws/src/fault_injector/

Deterministic fault plugins and eligibility checks.

ros_ws/src/recovery_manager/

Action guards, selector and Nav2 interfaces.

data/manifests/

Episode splits, bag checksums, exclusions and dataset versions.

src/labels/

Event detection and causal warning-window construction.

src/features/

Offline extraction, resampling, masks and normalization.

src/models/

Rules, anomaly baselines, TCN, recurrent and optional transformer models.

src/evaluation/

Event metrics, calibration, bootstrap and recovery analysis.

tests/

Unit, integration, replay, determinism, leakage and guard tests.

reports/

Generated tables, figures, model cards and manuscript sources.

Required automated tests

Test

Pass condition

Timestamp causality

No input feature uses a message timestamp after its decision time.

Split integrity

No run_id, map-route pair or derived window crosses a forbidden split.

Leakage denylist

Fault commands, ground truth and outcome fields cannot enter the deployable matrix.

Label fixtures

Hand-constructed event timelines produce exact positive, negative and excluded windows.

Replay equivalence

Offline and online feature pipelines agree within frozen tolerance.

Fault determinism

Same manifest and seed reproduce onset and injected sequence.

Missing-data behavior

Dropped topics create masks and bounded values, not silent forward-filled evidence.

Recovery guards

Every unsafe or ineligible action is rejected in test scenarios.

Metric fixtures

Synthetic predictions reproduce hand-calculated event recall, false-alert and lead-time results.

Clean rebuild

One documented command recreates the environment, tests and a smoke experiment.

15. Ten-week execution plan

Week

Primary objective

Evidence required before advancing

1

Freeze event definitions, feature schema and leakage denylist.

Reviewed protocol, ten annotated bags and passing label fixtures.

2

Build reproducible offline extraction and episode split tooling.

Replay equivalence on ten bags and split-integrity report.

3

Implement threshold rules and event-level evaluator.

Baseline result table plus false-alarm and lead-time plots.

4

Generate balanced pilot and complete dataset card.

600 episodes, prevalence report, runtime estimate and exclusion audit.

5

Train reconstruction, TCN and recurrent baselines.

Frozen training logs, validation comparison and latency profile.

6

Calibrate risk, freeze threshold policy and run ablations.

Reliability plots and signed model-selection record.

7

Run held-out-map and leave-one-family-out evaluation.

Immutable predictions and regenerated metric tables.

8

Implement recovery manager, eligibility guards and policy baselines.

All guard tests pass and ten witnessed recovery cases succeed safely.

9

Run paired recovery campaign and analyse action costs.

Completion, collision, overhead and regret tables regenerate from scripts.

10

Write preprint, release artefacts and create portfolio video.

Independent rerun reproduces the principal result.

Weekly operating rhythm

Monday: freeze the weekly claim and acceptance test.

Tuesday to Thursday: implement and run experiments with daily lab notes.

Friday: rerun the week result from a clean environment and update the dataset or model card.

Weekend: literature synthesis, error analysis, figures and manuscript drafting.

Never postpone label and failure-case inspection; review sampled timelines every week.


Page 

RESEARCH PROTOCOL  |  AUTONOMOUS ROBOT RELIABILITY

16. Acceptance gates

Gate

Pass condition

If it fails

G0: Shared platform

Research 1 produces deterministic episodes with complete logs.

Pause Research 2 and repair shared infrastructure.

G1: Labels

Twenty hand-audited episodes match automatic event and window labels exactly.

Do not train any model.

G2: Leakage

Denylist, timestamp and split tests all pass.

Invalidate derived data and rebuild it.

G3: Baseline

Threshold rules run online and evaluator reports event metrics.

Simplify features and debug timing.

G4: Pilot

At least 600 valid episodes reveal workable event prevalence and lead times.

Redesign injection onset, route set or event rules transparently.

G5: Predictor freeze

Model, calibration and alert policy are timestamped before held-out evaluation.

Treat later changes as a new protocol version.

G6: Generalisation

Held-out-map and unseen-family predictions are immutable and complete.

Report the negative result; do not tune on test failures.

G7: Recovery safety

All guards pass and no action can bypass collision checking.

Do not run closed-loop recovery.

G8: Release

Independent rerun regenerates the principal table from raw artefacts.

Treat the research artefact as incomplete.

17. Risks and mitigations

Risk

Early warning

Mitigation

Severe label leakage

Near-perfect validation score or abrupt risk jump at injection onset.

Audit every feature, delay all fault metadata and run a fault-ID probe.

Random windows cross splits

Training and test contain the same run_id or route.

Split episodes first and fail CI on overlap.

Injected faults are too easy

The model recognises artificial artifacts rather than impending failure.

Vary mechanisms, retain natural failures and use unseen-family evaluation.

Most injections do not cause failures

Low positive-event count despite many episodes.

Adjust eligible onset and severity using pilot data only; retain non-events.

Failure happens too quickly

Median precursor is inside the one-second guard.

Treat the family as not forecastable at that horizon and report it honestly.

Alarm chatter

High recall with unacceptable alerts per mission.

Calibrate, add persistence and optimise under a fixed alarm budget.

Learned selector exploits unsafe actions

Apparent completion gain with collisions or guard violations.

Hard action eligibility, collision monitor and dominant safety cost.

Dataset and compute expand uncontrollably

Long extraction or campaign estimates after pilot.

Use staged gates, compact primary models and one frozen primary severity.

No external methodological review

Leakage or invalid independence goes unnoticed.

Schedule two reviews focused on labels, splits and statistical unit.

ACADEMIC ETHICS  Keep all negative results, failed recoveries and exclusions. Cite reused Research 1 infrastructure and external code. Document AI assistance under the rules of any venue or institution. Never imply publication, peer review or real-world safety validation that has not occurred.

18. Paper and portfolio packaging

Suggested manuscript outline

Abstract: problem, causal benchmark, primary predictor result and recovery impact.

Introduction: why detecting a terminal error is different from providing useful warning.

Related work: robot introspection, fault diagnosis, time-series anomaly detection and autonomous recovery.

Benchmark: platform, failure definitions, injections, causal labels, splits and leakage controls.

Methods: features, threshold rules, temporal predictors, calibration and decision policy.

Recovery system: eligible actions, cost function, selector and safety guards.

Experiments: held-out maps, unseen families, ablations, latency and paired recovery campaign.

Results: event recall, false alarms, lead time, calibration, mission completion and action cost.

Failure analysis: missed warnings, false alarms, wrong recoveries and unforecastable events.

Limitations and ethics: injected-fault realism, simulation gap and absence of formal guarantees.

Reproducibility statement: code, raw logs, manifests, checkpoints, splits and commands.

Minimum figures

Architecture and causal labeling timeline.

Three annotated risk traces showing useful warning, false alarm and missed failure.

Event recall versus false alerts per mission.

Lead-time curve by failure family.

Reliability diagram before and after calibration.

Held-out-map and unseen-family result matrix.

Feature-group ablation plot.

Mission completion and recovery overhead comparison.

Recovery action confusion and representative failure cases.

Portfolio presentation

Lead with one visual sentence: the robot forecast a failure several seconds early, then recovered, or the method failed to generalise and the study explains why. Show synchronized video, telemetry risk, warning time and recovery action. Link the preprint, code, dataset card, one-command reproduction and limitations. Present Research 2 as a direct extension of Research 1, not as a disconnected second demo.

19. Supervisor review checklist

Are failure events operationally defined without subjective post-hoc labeling?

Can any deployable feature reveal the injected fault or eventual terminal result?

Are episode, map and route boundaries respected before window extraction?

Is event recall reported with an explicit false-alarm budget?

Is the warning early enough for the declared recovery action?

Does the unseen-family test truly exclude that family from feature and threshold tuning?

Are natural system failures separated from injected-fault results?

Are recovery eligibility and safety guards independent of the learned selector?

Are all exclusions, failed recoveries and negative results retained?

Can an independent person regenerate the primary tables from the raw bags?


Page 

RESEARCH PROTOCOL  |  AUTONOMOUS ROBOT RELIABILITY

20. Immediate setup checklist

Before creating a dataset

Copy the Research 1 split manifest and mark the held-out maps read-only.

Create configs/failure_events.yaml with thresholds, durations and event precedence.

Create configs/feature_schema.yaml with units, rate, age and missingness semantics.

Create configs/leakage_denylist.yaml with every label-only field and topic.

Record and checksum ten representative raw bags.

Write hand annotations for clean, collision, localisation, stuck and timeout cases.

First working baseline

Extract odometry, command velocity, AMCL covariance, scan validity, goal distance and planner status.

Construct labels with W=5 seconds, H=10 seconds, delta=1 second and G=20 seconds.

Implement four simple rules: covariance high, progress low, scan dropout high and oscillation high.

Select no threshold on the test set. Use development bags, then validate once.

Generate event recall, false alerts per mission and lead-time plots.

Replay one warning through a guarded stop-and-replan recovery.

Research controls

Add CI tests for split integrity, causal timestamps and the leakage denylist.

Start a dataset card and model card before training.

Log every protocol change with its effect on previously generated data.

Schedule the first methods review before the balanced pilot.

Do not inspect held-out risk traces until predictor and policy freeze.

START TODAY  Your first deliverable is not a transformer. It is a plot of a real navigation episode showing exactly when the precursor window begins, what information is available at each decision, and why the label is causally valid.

21. Reading order and references

Read the official logging and recovery interfaces first, then the robot-introspection papers, then the temporal and anomaly-detection baselines. The list below is the working reference set for Protocol 1.0.

ROS 2. Rosbag2 repository and usage. Timestamped recording, replay, MCAP storage, snapshot mode and lost-message statistics. https://github.com/ros2/rosbag2

ROS 2 Jazzy. diagnostic_updater API. Structured diagnostic tasks and system-health reporting. https://docs.ros.org/en/jazzy/p/diagnostic_updater/generated/index.html

Nav2. Behavior Server. Recovery and behavior plugins including spin, backup, drive on heading and wait. https://docs.nav2.org/configuration/packages/configuring-behavior-server.html

Nav2. Detailed Behavior Tree Walkthrough. System-level recovery flow and default recovery actions. https://docs.nav2.org/behavior_trees/overview/detailed_behavior_tree_walkthrough

Nav2. Simple Commander API. Programmatic navigation, cancellation and recovery actions. https://docs.nav2.org/commander_api/index.html

Daftry, S., Zeng, S., Bagnell, J. A., and Hebert, M. 2016. Introspective Perception: Learning to Predict Failures in Vision Systems. https://arxiv.org/abs/1607.08665

Saxena, D. M., Kurtz, V., and Hebert, M. 2017. Learning Robust Failure Response for Autonomous Vision Based Flight. https://publications.ri.cmu.edu/storage/publications/2017/08/07989684.pdf

Saxena, D. M. 2017. Supervised Learning of Corrective Maneuvers for Vision-Based Autonomous Flight. CMU-RI-TR-17-55. https://www.ri.cmu.edu/publications/supervised-learning-of-corrective-maneuvers-for-vision-based-autonomous-flight/

Audibert, J., Michiardi, P., Guyard, F., Marti, S., and Zuluaga, M. A. 2020. USAD: Unsupervised Anomaly Detection on Multivariate Time Series. KDD. https://www.eurecom.fr/publication/6271

Su, Y. et al. 2019. Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network. KDD. https://doi.org/10.1145/3292500.3330672

Garg, A. et al. 2022. An Evaluation of Anomaly Detection and Diagnosis in Multivariate Time Series. https://arxiv.org/abs/2109.11428

Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. 2017. On Calibration of Modern Neural Networks. ICML. https://proceedings.mlr.press/v70/guo17a.html

Bai, S., Kolter, J. Z., and Koltun, V. 2018. An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. https://arxiv.org/abs/1803.01271

Oxford Robotics Institute. Goal-Oriented Long-Lived Systems Lab. Planning, autonomy and decision-making under uncertainty. https://ori.ox.ac.uk/groups/goals

Carnegie Mellon Robotics Institute. Towards Scalable Visual Navigation of Unmanned Aerial Vehicles in the Wild. Introspection and navigation reliability. https://www.ri.cmu.edu/pub_files/2016/4/main2-daftry.pdf

Protocol change control

Version

Date

Status

Required update

1.0

24 Aug 2026

Issued

Initial start-ready protocol.

1.1

After Week 2

Planned

Freeze event definitions, features, splits and leakage controls.

1.2

After Week 4

Planned

Record pilot prevalence, runtime and final campaign size.

1.3

Before Week 7

Planned

Freeze model, calibration, alert policy and confirmatory analysis.

2.0

After study

Planned

Record deviations, final results and archived artefact identifiers.

FINAL PRINCIPLE  A credible failure predictor must be causal, early and useful. If it predicts only after the robot is already failing, depends on leaked fault metadata or alarms constantly, the correct conclusion is that the method did not solve the research problem.

Page 
