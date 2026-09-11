# Early Failure Prediction and Recovery for Mobile Robot Navigation

Emmanuel Alabi Olasubomi

Manuscript status: methods-complete working draft; results are intentionally withheld
until the preregistered gates and protected evaluation pass.

## Abstract

Mobile navigation failures are often detected only after progress has stopped, a
planner has aborted, or a collision has become unavoidable. We study whether recent
multimodal robot telemetry can forecast the first mission-ending event early enough to
support a bounded recovery. The benchmark uses strictly past-only five-second windows,
a ten-second warning horizon, a one-second too-late interval, complete-episode splits,
held-out maps, and leave-one-failure-family-out evaluation. The primary endpoint is
event recall at a validation-fixed budget of at most 0.10 false alerts per clean
mission. The downstream endpoint is paired mission completion under independently
guarded recovery. **RESULT_PENDING:** predictor effect, warning lead time, calibration,
unseen-family generalisation, recovery effect and uncertainty intervals will be inserted
only from immutable generated artifacts.

## 1. Introduction

A navigation system can fail even when each individual sensor and autonomy component
appears locally plausible. Sensor degradation, localisation drift, wheel slip, dynamic
blockage and planner oscillation can develop over time before the navigation action
terminates. A useful monitor must therefore do more than identify an already-failed
state: it must issue a sufficiently early warning without alarming continuously, and
that warning must improve a downstream mission decision.

Prior robot-introspection work established that component failures can be predicted
from observations, while proactive navigation studies showed that multimodal signals or
planner-risk estimates can anticipate selected failures. Recent recovery-aware planning
also integrates predicted dead ends or obstacle motion into navigation. The remaining
experimental question is whether one causal event-level protocol can combine controlled
alarm burden, protected map generalisation, unseen failure families and paired guarded
recovery across heterogeneous system failures.

This study makes three scoped contributions. First, it defines auditable first-terminal-
event labels and explicitly excludes future, injected-fault and post-outcome information.
Second, it compares transparent rules, anomaly baselines and causal temporal predictors
at one validation-fixed operational alarm budget. Third, it tests whether warnings
improve mission completion when recovery actions remain subordinate to an independent
eligibility guard. The work does not claim formal safety certification or universal
anomaly detection.

## 2. Related work

The closest direct comparator is proactive anomaly detection for robot navigation using
camera, LiDAR and planned-path information. Introspective perception motivates health
and uncertainty features but predicts component reliability rather than necessarily a
terminal mission outcome. Reconstruction methods such as USAD and OmniAnomaly provide
normal-only multivariate baselines; their scores are evaluated here through the same
alarm policy as supervised models.

Temporal convolutional networks provide the primary learned architecture because they
are causal, parallelisable and a strong generic sequence baseline. Recurrent models are
included at a comparable parameter scale. Calibration is operational because a risk
threshold triggers an intervention with asymmetric cost. Event-level evaluation follows
early-event work by measuring useful warnings and notification burden rather than
treating overlapping windows as independent observations.

Recovery evidence is separated from warning evidence. Nav2 provides bounded behaviors,
but a successful offline prediction does not establish improved autonomy. We therefore
compare policies on paired map, route, seed and fault episodes, retain guard rejections,
and report collisions and intervention overhead alongside mission completion. The full
source-by-source synthesis and search record are maintained in `literature/`.

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
false arrival, localisation loss, immobilisation, unsafe perception, or mission timeout.
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
localisation perturbation, dynamic blockage, planner oscillation and semantic
corruption. Each fault has a seeded onset, eligibility rule, duration and severity
manifest. Environment faults are placed relative to the active global path. An injected
fault that does not cause a terminal event is retained as a non-event rather than
discarded. Combined faults are exploratory and excluded from confirmatory claims.

### 3.4 Telemetry and leakage prevention

Candidate inputs cover commanded and measured motion, localisation covariance and TF
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
validation data only. Mandatory ablations remove localisation, planner/controller,
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
stop, relocalisation when a concrete procedure exists, clear/replan, bounded backup,
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

**RESULT_PENDING.** This section must be generated from immutable pilot, validation,
held-out-map, unseen-family and paired-recovery artifacts. It must include denominators,
negative results, exclusions, intervals and all guard rejections.

Required figures are the architecture/causal timeline, annotated risk traces, recall
versus false-alert burden, lead-time curves, reliability diagrams, held-out and unseen-
family matrices, feature ablations, recovery outcomes and recovery-action errors.

## 5. Discussion

**RESULT_PENDING.** Interpretation will be constrained to the observed failure families,
maps, robot platform and warning horizon. A failure to outperform rules, transfer to
unseen families, meet the alarm budget or improve paired completion will be reported as
the study result rather than repaired through protected-set tuning.

## 6. Limitations and ethics

The benchmark is simulated, uses one robot family and deliberately controlled faults,
and provides no formal collision-avoidance guarantee. Injected signatures may differ
from physical degradation. Timeout can have diffuse precursors. Independent guards and
existing collision checking remain active, but these controls are not certification.

All invalid episodes, negative results, exclusions, missed warnings, false alarms,
failed recoveries and guard rejections are retained. Reused Research 1 infrastructure
and external software are credited. AI assistance must be disclosed according to the
chosen institution and venue policy.

Label review was performed by one human reviewer. AI comparison was retained only as
supporting evidence and was not counted as an independent human rating. Consequently,
the work cannot estimate inter-rater agreement; this is disclosed as a methodological
limitation rather than filled with a synthetic reviewer identity.

## 7. Reproducibility statement

The release will contain raw-bag checksums, manifests, split assignments, extraction and
label versions, feature schema, normalisation bundle, model checkpoints, calibration and
threshold records, immutable predictions, generated tables and figures, model and
dataset cards, and a one-command verification path. **RELEASE_PENDING:** independent
reproduction has not yet been performed.

Long-running collection is observed by a read-only 30-minute monitor that reconciles
parent and replacement ledgers, advisory lock ownership and free space. It cannot start,
stop or retry an episode. Scientific completion is determined only from immutable
campaign reports and hash-addressed inventories, not from the monitor snapshot.

## References

The working bibliography and verified evidence matrix are maintained in
`literature/README.md`, `literature/evidence-matrix.md` and `literature/search-log.md`.
Venue-formatted references will be generated only after bibliographic metadata review.
