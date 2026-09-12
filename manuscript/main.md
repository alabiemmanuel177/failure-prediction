---
title: "Early Warning Without Demonstrated Recovery Benefit: Failure Prediction and Guarded Recovery for Mobile Robot Navigation"
author: "Emmanuel Alabi Olasubomi"
date: "12 September 2026"
---

# Abstract

Predicting a failure is not the same as preventing one. We study whether recent
multimodal robot telemetry can forecast the first mission-ending navigation event early
enough to support a bounded recovery, and whether acting on that forecast improves the
mission. The benchmark uses strictly past-only five-second windows, a ten-second warning
horizon, a one-second too-late interval, complete-episode splits, held-out maps, and
leave-one-failure-family-out evaluation. The primary endpoint is event recall at a
validation-fixed budget of at most 0.10 false alerts per clean mission; the downstream
endpoint is paired mission completion under an independently guarded recovery. On 3
held-out maps (157 terminal events, 1002 episodes) the causal TCN reached event recall
22.9% against 3.2% for the threshold rules at the frozen validation threshold (0.183
false alerts per clean mission), a paired difference of +19.7 percentage points (95%
hierarchical bootstrap interval -0.0 to +44.7). The interval includes zero and the
preregistered criterion is not met. Warning was nonetheless timely where it occurred:
median useful lead time was 3.8 s over 36 detected events, with 121 undetected. Mission
timeouts (6 episodes) were analyzed separately, as preregistered; the TCN warned before
0.0% of them. Leave-one-family-out recall exceeded the baseline in four of seven
families against a preregistered five. In the paired recovery campaign (504
map/route/seed/fault pairs), mission completion under the cost-sensitive guarded policy
R3 was 84.3% versus 86.1% under default Nav2 recovery R0 (difference -1.8 points, 95%
interval -7.7 to +3.2; collision-rate difference +0.0 points). Guard rejections: 0;
guard violations: 0. Both confirmatory hypotheses, H1 and H6, are unsupported. The
result demonstrates a signal-to-decision gap: failures were visible several seconds in
advance, and neither beating a transparent baseline at a fixed alarm budget nor
converting the warning into a better mission outcome was established. We release the
protocol, manifest-scoped analysis, integrity audit, trained checkpoints and core
evidence archive (DOI 10.5281/zenodo.22723258) to make this negative result
reproducible.

## 1. Introduction

A navigation system can fail even when each individual sensor and autonomy component
appears locally plausible. Sensor degradation, localization drift, wheel slip, dynamic
blockage and planner oscillation can develop over time before the navigation action
terminates. A useful monitor must therefore do more than identify an already-failed
state: it must issue a sufficiently early warning without alarming continuously, and
that warning must improve a downstream mission decision.

The literature supplies strong components for the first half of that requirement.
Proactive anomaly detection anticipates navigation failures from fused sensing [1, 2, 3],
introspective perception predicts when a component is unreliable [4, 5, 6], and
reconstruction-based detectors provide normal-only multivariate baselines [7, 8]. The
difficult question is empirical, and it concerns the second half: does an early warning,
delivered under a fixed alarm budget, change what the robot achieves?

We evaluate that link in a frozen ROS 2/Gazebo benchmark built on the platform of a
preceding study. The design separates prediction from action, holds the rest of the
navigation stack fixed, pairs map, route, seed and fault across policies, and tests six
preregistered hypotheses against a threshold and checkpoint frozen before the protected
split was touched. The contribution is methodological and empirical:

1. auditable first-terminal-event labels that exclude future, injected-fault and
   post-outcome information;
2. a comparison of transparent rules, anomaly baselines and causal temporal predictors
   at one validation-fixed operational alarm budget;
3. a paired test of whether warnings improve mission completion when recovery actions
   remain subordinate to an independent eligibility guard.

The work does not claim formal safety certification or universal anomaly detection.

## 2. Related work

The closest direct comparators are proactive anomaly detectors for robot navigation that
fuse camera, LiDAR and planned-path information [1, 2], including occlusion-aware
recurrent variants for field robots [3]. Introspective perception motivates the health
and uncertainty features used here, but predicts component reliability rather than a
terminal mission outcome [4, 5], and earlier corrective-maneuver work learns recovery
from predicted failure rather than evaluating the warning itself [6]. Reconstruction
methods such as USAD [7] and OmniAnomaly [8] supply normal-only multivariate baselines;
their scores are evaluated here through the same alarm policy as the supervised models,
so the comparison is of detectors, not of thresholds.

Temporal convolutional networks provide the primary learned architecture because they
are causal, parallelizable and a strong generic sequence baseline [9]. Recurrent models
are included at comparable parameter scale. Calibration is treated as operational rather
than cosmetic, because a risk threshold triggers an intervention with asymmetric cost
[10]. Event-level evaluation follows early-event prediction work by measuring useful
warnings and notification burden rather than treating overlapping windows as independent
observations [11, 12].

Recovery evidence is kept separate from warning evidence. Recent planning work
incorporates predicted dead ends, local minima and motion into navigation [13, 14, 16],
and regret-based analysis argues that prediction errors should be scored by their
downstream consequence rather than in isolation [15]. That argument motivates the design
here: a successful offline prediction does not establish improved autonomy. We therefore
compare policies on paired map, route, seed and fault episodes using the Nav2 behavior
server and commander interfaces [17, 18], retain guard rejections, and report collisions
and intervention overhead alongside mission completion. The full source-by-source
synthesis and search record are maintained in `literature/`.

## 3. Methods

### 3.1 Platform and study unit

Experiments use Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic and Nav2 on the content-locked
Research 1 platform. Research 2 is an overlay that records immutable MCAP bags,
checksums, topic-health sidecars, event streams and episode summaries. The independent
unit is a complete navigation episode. No window from an episode can cross a data split.
Episode summaries and event sidecars are serialized before exclusive atomic publication,
flushed to stable storage and never overwritten. A separate metadata-only integrity gate
rejects every zero-length summary, MCAP or bag-metadata payload unless the affected run
is explicitly retained, excluded and linked to one approved replacement.

Recording was reduced prospectively in two versioned stages after storage audits. The
first 54 pilot episodes retain full images. Attempts 55–97 use `compact_v1`, which omits
camera/depth/segmentation bags but retains semantic confidence images. From attempt 98,
`compact_v2` records synchronized causal perception scalars instead of those semantic
images. The primary cross-profile perception contract is restricted to confidence mean,
uncertainty mean and inference latency; source representation and profile identifiers
are not model inputs. Earlier bags remain immutable and use the corresponding legacy
offline adapter.

### 3.2 Failure events and causal labels

The primary event is the first prespecified terminal event: collision, navigation abort,
false arrival, localization loss, immobilization, unsafe perception, or mission timeout.
Operational thresholds are frozen before training and ground truth is used only for
offline labels where required. Naturally occurring failures in no-injection runs follow
the same rules but form a separate audit.

The researcher manually inspected a frozen 20-episode audit sample without model
outputs and agreed with all 20 automatic causal signatures. No independent second
reviewer was available. Protocol Amendment PA-2026-09-03-01 admits this primary review
for model development; the study therefore makes no inter-rater-reliability claim.

At decision time *t*, the input contains the preceding five seconds. A decision is
positive when *t* lies from ten to one seconds before terminal time. Decisions inside
the final second, after termination, or after the event receive no early-warning credit.
Eligible negatives remain at least twenty seconds from every event and injection onset.
All times use recorded message availability, not a future header timestamp.

### 3.3 Fault benchmark

Seven deterministic families are studied: camera occlusion, LiDAR dropout, wheel slip,
localization perturbation, dynamic blockage, planner oscillation and semantic
corruption. Each fault has a seeded onset, eligibility rule, duration and severity
manifest. Environment faults are placed relative to the active global path. An injected
fault that does not cause a terminal event is retained as a non-event rather than
discarded. Combined faults are exploratory and excluded from confirmatory claims.

### 3.4 Telemetry and leakage prevention

Candidate inputs cover commanded and measured motion, localization covariance and TF
health, range validity and clearance trends, perception confidence, planner/controller
state, system health and goal-relative progress. Every value has explicit age and
missingness semantics. Normalisation is fit on development episodes only.

Fault commands, fault parameters, injection time, simulator ground truth, terminal
results, recovery outcomes and future-derived features are label-only. Automated tests
reject a deployable matrix containing forbidden names, timestamps later than the
decision, split overlap or silent unbounded forward fill. Offline and online feature
implementations must agree within a frozen tolerance.

### 3.5 Dataset and protected evaluation

The completed balanced pilot contains 648 development-only episodes over six maps, two
routes per map, six replicates, two clean controls and seven medium-severity faults. It
contains 125 terminal events and 523 non-events over 41,701.374 seconds. The retained
bags occupy 53.20 GiB: 54 use `full_v1`, 42 use `compact_v1`, and 552 use `compact_v2`.
The hash-addressed episode inventory and every source checksum are frozen before
admitted feature extraction.

A post-pilot planning audit observed 5–10 terminal events in four injected families.
Before model fitting it therefore preregistered 1,212 additional development episodes,
rounded to complete 12-route blocks, to raise every injected family to approximately 30
expected terminal events at its pilot prevalence. This floor determines data collection
only and is not a confirmatory endpoint.

An attempt invalidated by infrastructure before goal dispatch contributes no scientific
episode. A completed faulted attempt whose declared treatment was not recorded is also
invalid rather than recoded as a control. Post-validation loss of the summary or bag
payload invalidates historical ledger success because the episode can no longer be
independently replayed. Each case remains in the attempt inventory and may receive at
most one distinct, manifest-linked replacement using the same map, route, system,
condition and seed. The original and replacement are both reported, while the design
cell contributes exactly once. Two pre-goal infrastructure attempts and two exact linked
replacements occurred in development; validation exclusions and replacements are
reported from its final immutable inventory rather than inferred from ledger status.

Development data support fitting and debugging. A separately preregistered 324-episode
raw campaign spans three validation maps, six routes, six replicates, two controls and
seven fault families. Separate validation maps select window
configuration, architecture, calibration, persistence, cooldown and threshold. Held-out
maps are inspected once after the model and analysis policy are frozen. Seven
leave-one-family-out folds remove one fault family from all fitting and tuning. All
uncertainty resamples maps, then routes, then episodes.

Research 1 episode aggregates are not time series and therefore cannot be converted to
decision windows. A protected-outcome-free structural audit found 825 retained
development bags and 364 retained validation bags with the core temporal topic contract.
Only the development bags are eligible for fitting, and then only as a separately
adapted and human-audited natural/control source. Research 1 validation bags remain
selection-only, while Research 1 test bags are forbidden for fitting and tuning. With
the 1,860 planned Research 2 development episodes, the maximum pre-supplement fitting
pool is 2,685; a route-balanced supplement of at least 324 development episodes is
therefore required if all 825 Research 1 development bags are admitted.

### 3.6 Predictors and baselines

P0 provides no warning. P1 contains transparent validation-tuned threshold rules. P2 is
a normal-only reconstruction model. P3, the primary predictor, is a causal dilated
temporal convolutional network with residual blocks and dropout. P4 is a parameter-
matched GRU or LSTM; a compact transformer is optional only if data volume and latency
justify it. Inputs include synchronized features, missingness masks and age channels.

The primary head estimates the probability of a terminal event within ten seconds. An
optional secondary head predicts failure family. Model selection uses development and
validation data only. Mandatory ablations remove localization, planner/controller,
perception-confidence and raw-sensor groups; replace history with a single timestamp;
and remove calibration or alert persistence.

### 3.7 Calibration and alarm policy

Calibration candidates are fit on validation episodes only and assessed with Brier
score, expected calibration error and reliability curves. At 2 Hz, an alert requires
two of the last three calibrated scores to exceed the threshold and enters a ten-second
cooldown. The threshold maximises validation event recall subject to at most 0.10 false
alerts per clean mission. Deterministic ties prefer lower burden, longer median useful
lead time and then the higher threshold. The selected calibration and threshold are
frozen before protected evaluation.

### 3.8 Recovery

R0 is the default Nav2 recovery tree without the predictor. R1 uses the warning followed
by conservative stop and replan. R2 maps diagnosed signal groups to guarded actions. R3
selects the lowest predicted-cost eligible action. Available actions are controlled
stop, relocalization when a concrete procedure exists, clear/replan, bounded backup,
spin/rescan, wait and request assistance.

The independent guard has final authority. Backup requires verified rear clearance;
rotation requires swept-volume clearance; wait and replanning require no immediate
collision risk; and repeated recovery exhausts an action budget. If no automated action
is eligible, the system requests assistance. The selector cannot disable Nav2 collision
checking or train on ineligible actions.

Before learned recovery, the discrete guard was exhaustively evaluated over 3,456 robot
states and 51,840 decisions from R1, R2 and R3 policy variants. No ineligible action was
admitted. This is engineering evidence for the guard implementation, not evidence that
prediction-triggered recovery improves mission completion; live recovery remains locked
until the post-model paired campaign.

### 3.9 Outcomes and statistical analysis

The primary prediction outcome is the difference between P3 and P1 event recall at the
frozen alarm budget. A hierarchical paired bootstrap samples maps, routes and episodes
and reports a 95% interval. Warning lead time includes the full failure denominator;
undetected failures are never silently omitted. Supporting outcomes include false alerts
per mission, alert burden, AUPRC, AUROC, Brier score, ECE and inference latency.

The primary systems outcome compares paired mission completion under R3 versus R0 and
R2. The planned model uses policy as a fixed effect and map and route as random
intercepts, with a hierarchical paired bootstrap if convergence or assumptions fail.
Collision rate, guard violations, recovery success, time, path overhead and intervention
count are mandatory companion outcomes. H1 and H6 are confirmatory; subgroup and
combined-fault analyses are exploratory.

## 4. Results

Results were generated on 2026-09-12 from immutable artifacts (see reports/tables and reports/figures with their sha256 sidecars).

On 3 held-out maps (157 terminal events, 1002 episodes) the causal TCN reached event recall 22.9% against 3.2% for the threshold rules at the frozen validation threshold (0.183 false alerts per clean mission), a paired difference of +19.7 percentage points (95% hierarchical bootstrap interval -0.0 to +44.7; preregistered criterion not met); median useful lead time 3.8 s over 36 detected events with 121 undetected. Mission timeouts (6 episodes) were analyzed separately, as preregistered: the TCN warned before 0.0% of them.

Leave-one-family-out recall for the TCN was camera_occlusion 31.2% (16 events); dynamic_blockage 12.9% (31 events); lidar_dropout 0.0% (15 events); localisation_perturbation 34.8% (23 events); planner_oscillation 0.0% (28 events); semantic_corruption 0.0% (8 events); wheel_slip 10.0% (20 events). No pooled unseen-family claim is made; heterogeneity across folds is reported as observed.

In the paired recovery campaign (504 map/route/seed/fault pairs), mission completion under the cost-sensitive guarded policy R3 was 84.3% versus 86.1% under default Nav2 recovery R0 (difference -1.8 points, 95% interval -7.7 to +3.2; collision-rate difference +0.0 points; estimator mixed_effects_logistic_variational_bayes_statsmodels). Guard rejections: 0; guard violations: 0. Hypothesis H6 is not supported.

### 4.1 Figures

Every figure carries a `.json` provenance sidecar naming the checkpoint hash it was
produced from.

**Figure 1** (`fig01_architecture_causal_timeline`) — The prediction pipeline and its
causal timeline: past-only five-second windows, the ten-second warning horizon, and the
one-second too-late interval that separates a useful warning from a late one.

**Figure 2** (`fig02_risk_trace_useful_warning`, `fig02_risk_trace_false_alarm`,
`fig02_risk_trace_missed_failure`) — Annotated risk traces for the three outcomes that
matter operationally, shown at equal weight: a useful warning, a false alarm, and a
missed failure. The false alarm and the miss are shown because a figure set containing
only successes would misdescribe a system whose primary hypothesis was unsupported.

**Figure 3** (`fig03_recall_vs_false_alerts`) — Event recall against false-alert burden.
The operating point is the validation-fixed budget of at most 0.10 false alerts per clean
mission; the realized held-out rate was 0.183.

**Figure 4** (`fig04_lead_time_by_family`) — Distribution of useful lead time by failure
family over the 36 detected events, against the 3 s bar set in advance.

**Figure 5** (`fig05_reliability_diagrams`) — Reliability before and after calibration on
validation data, the evidence behind H3.

**Figure 6** (`fig06_generalisation_matrix`) — Leave-one-family-out recall, predictor
against held-out family. The heterogeneity across folds is the substance of H5: four of
seven families exceeded the baseline against a preregistered five, and three folds
returned zero recall.

**Figure 7** (`fig07_feature_group_ablation`) — Recall decline by ablated feature group.
No numeric threshold was prespecified for H4, so this figure is reported rather than
tested.

**Figure 8** (`fig08_recovery_outcomes`) — Paired mission outcomes under the guarded
policy R3 against default Nav2 recovery R0 across 504 matched pairs.

**Figure 9** (`fig09_action_confusion`) — Recovery-action confusion for the selector,
with guard rejections and violations both zero.
## 5. Discussion

Interpretation is restricted to the observed failure families, maps, robot platform and warning horizon. The confirmatory claims H1 and H6 are reported exactly as estimated above, including negative components, without protected-set tuning.
## 6. Limitations and ethics

The benchmark is simulated, uses one robot family and deliberately controlled faults,
and provides no formal collision-avoidance guarantee. Injected signatures may differ
from physical degradation. Timeout can have diffuse precursors. Independent guards and
existing collision checking remain active, but these controls are not certification.

All invalid episodes, negative results, exclusions, missed warnings, false alarms,
failed recoveries and guard rejections are retained. Reused infrastructure from the
preceding study and external software are credited.

Generative AI assisted with code review, experiment orchestration, documentation
drafting, figure and release-asset generation, and consistency checks. The human
researcher approved the protocol and its amendments, controlled the workstation, and
remains responsible for the study, its claims, the data release, authorship and any
submission. AI-generated text and code were checked against committed source files and
machine-readable results. No AI system is listed as an author.

Label review was performed by one human reviewer. AI comparison was retained only as
supporting evidence and was not counted as an independent human rating. Consequently,
the work cannot estimate inter-rater agreement; this is disclosed as a methodological
limitation rather than filled with a synthetic reviewer identity.

## 7. Reproducibility statement

Every deviation from the preregistered protocol, every amendment, every execution incident and every disclosure is listed in `docs/deviations-and-disclosures.md` and recorded in the chained research log; the paired recovery campaign was executed twice and only the second execution, in which the recovery treatment was actually delivered, enters the H6 analysis (amendment PA-2026-09-10-01).

The core evidence archive is deposited at DOI 10.5281/zenodo.22723258 under CC BY 4.0:
2,225 files containing every report and manifest, the split assignments, extraction and
label versions, feature schema, normalization bundle, all trained checkpoints with their
training records, calibration and threshold records, generated tables and figures, and
the model and dataset cards. Per-window prediction tables were excluded as regenerable,
and the derived model-input sequences and raw recordings are registered by manifest
rather than deposited; the archive therefore supports verifying every reported number
and inspecting any trained model, but not regenerating the predictions from scratch. An independent clean rerun (commit 7ae157c19052, 2026-09-12T08:07:47Z) regenerated the predictions from the frozen checkpoint and reproduced the released tables within tolerance: tab01_predictor_summary identical_bytes, tab02_recall_by_family identical_bytes, tab03_unseen_family identical_bytes, tab04_ablations identical_bytes, tab05_recovery_outcomes identical_bytes, tab06_action_confusion identical_bytes, tab07_latency identical_bytes.

Long-running collection is observed by a read-only 30-minute monitor that reconciles
parent and replacement ledgers, advisory lock ownership and free space. It cannot start,
stop or retry an episode. Scientific completion is determined only from immutable
campaign reports and hash-addressed inventories, not from the monitor snapshot.

## References

[1] T. Ji, A. N. Sivakumar, G. Chowdhary and K. Driggs-Campbell. Proactive Anomaly Detection for Robot Navigation with Multi-Sensor Fusion. arXiv:2204.01146, 2022.

[2] T. Ji, S. T. Vuppala, G. Chowdhary and K. Driggs-Campbell. Multi-Modal Anomaly Detection for Unstructured and Uncertain Environments. Conference on Robot Learning, 2021.

[3] A. Schreiber et al. An Attentional Recurrent Neural Network for Occlusion-Aware Proactive Anomaly Detection in Field Robot Navigation. arXiv, 2023.

[4] S. Daftry et al. Introspective Perception: Learning to Predict Failures in Vision Systems. arXiv, 2016.

[5] S. Rabiee et al. Introspective Perception for Mobile Robots. Artificial Intelligence, 2023.

[6] D. Saxena. Supervised Learning of Corrective Maneuvers for Vision-Based Autonomous Flight. 2017.

[7] J. Audibert et al. USAD: UnSupervised Anomaly Detection on Multivariate Time Series. ACM SIGKDD, 2020.

[8] Y. Su et al. Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network. ACM SIGKDD, 2019.

[9] S. Bai, J. Z. Kolter and V. Koltun. An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. arXiv, 2018.

[10] C. Guo et al. On Calibration of Modern Neural Networks. ICML, 2017.

[11] H. Yeche et al. Temporal Label Smoothing for Early Event Prediction. ICML, 2023.

[12] H. Yeche et al. Dynamic Survival Analysis for Early Event Prediction. Conference on Health, Inference, and Learning, 2024.

[13] A. Rajagopal et al. DR. Nav: Semantic-Geometric Representations for Proactive Dead-End Recovery and Navigation. arXiv, 2025.

[14] Y. Xue et al. Proactive Local-Minima-Free Robot Navigation: Blending Motion Prediction with Safe Control. arXiv, 2026.

[15] K. Nakamura et al. Not All Errors Are Made Equal: A Regret Metric for Detecting System-level Trajectory Prediction Failures. Conference on Robot Learning, 2025.

[16] A. Mohammad et al. A GP-based Robust Motion Planning Framework for Agile Autonomous Robot Navigation and Recovery in Unknown Environments. arXiv, 2024.

[17] Nav2 Project. Behavior Server.

[18] Nav2 Project. Simple Commander API.

[19] ROS 2. rosbag2.

Bibliographic metadata is maintained in `literature/references.bib`, with the
source-by-source appraisal and search record in `literature/evidence-matrix.md` and
`literature/search-log.md`. Venue-formatted references will be regenerated from the
BibTeX at submission.
