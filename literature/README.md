# Literature review: early failure prediction and recovery for mobile robot navigation

Status: protocol-complete structured scoping review, frozen 1 September 2026
Protocol: 1.0

## Review question

What evidence and methods support a mobile robot using only past and present multimodal telemetry to forecast a mission-ending navigation failure early enough to take a guarded recovery action, especially on maps and failure families excluded from training?

## Scope and search method

This first pass uses four connected bodies of work:

1. robot introspection and proactive navigation anomaly detection;
2. multivariate time-series anomaly detection and causal temporal models;
3. early-event evaluation and probability calibration;
4. robot recovery and ROS 2/Nav2 execution interfaces.

Sources were prioritized when they were primary papers, official project pages, or official ROS/Nav2 documentation. Searches used combinations of *robot navigation*, *failure prediction*, *proactive anomaly detection*, *introspection*, *multi-sensor fusion*, *early event prediction*, *lead time*, and *recovery*. Backward and targeted forward chaining from PAAD and introspective-perception work were completed for Protocol 1.0. This is intentionally a structured scoping review, not a systematic review or meta-analysis; heterogeneous tasks and outcome definitions do not support a pooled effect estimate.

The auditable search record is in [search-log.md](search-log.md), and the extracted
study-level comparison is in [evidence-matrix.md](evidence-matrix.md). Checked working
BibTeX for the manuscript's core sources is in [references.bib](references.bib).

## Synthesis

### 1. The closest precedent predicts navigation failures proactively

Ji et al.'s PAAD is the nearest direct comparator. It estimates future navigation-failure probabilities from current camera/LiDAR observations and a planned path. Its field experiments support two central premises of this protocol: planned autonomy state can be predictive, and multimodal fusion can remain useful under occlusion. PAAD should therefore be implemented or cited as a conceptual comparator where its data and assumptions permit.

The important unresolved gap is evaluation and systems closure. This study must determine whether a warning remains useful when:

- all windows from an episode remain in one split;
- maps and complete failure families are excluded from fitting and tuning;
- alarms are evaluated per event and per mission at a validation-fixed budget;
- late warnings receive no primary credit;
- calibration, persistence, and cooldown are treated as part of the deployed policy; and
- the warning initiates an eligible recovery whose mission outcome and cost are measured.

This gap is narrower and more defensible than a claim of general anomaly detection.

### 2. Robot introspection motivates internal-health features

Introspective-perception research treats predicted perception error as an input to higher-level estimation, planning, or supervision decisions. Daftry et al. established the idea of predicting vision-system failures from appearance. Rabiee and Biswas later presented a broader theory and real-robot evidence for introspective perception in visual SLAM and stereo depth estimation. These works support including perception confidence, pose health, and temporal disagreement, but they also expose a boundary: predicting a component error is not identical to forecasting a terminal navigation event.

Ji et al.'s earlier multi-modal anomaly-detection work emphasizes that heterogeneous modalities can improve robustness in unstructured environments. This supports feature-group ablations rather than assuming every sensor is beneficial.

### 3. Temporal models are baselines, not the scientific contribution

Bai, Kolter, and Koltun found that a simple temporal convolutional architecture can be a strong default compared with canonical recurrent models across sequence tasks. This supports the protocol's causal TCN as the primary learned model and a GRU/LSTM as a controlled comparator.

USAD and OmniAnomaly are useful reconstruction or normal-only baselines for multivariate telemetry. Their anomaly scores do not, by themselves, answer whether an alert forecasts a terminal event within a useful horizon. They must be passed through the same validation-only threshold, persistence, cooldown, and event evaluator as supervised models.

### 4. Early-event evaluation must include timing and burden

Early-event prediction literature shows why ordinary window accuracy and even window AUPRC are incomplete. Yèche et al. frame early-event prediction around low false-alarm rates and temporal label structure. Event-level alarm frameworks distinguish actionable, early, late, and missed warnings and measure notification burden per independent record. For this robot study, the equivalent independent record is the navigation episode.

Accordingly, the primary analysis should report event recall at no more than 0.10 false alerts per clean mission, lead time from the first useful alert, and the number/time of alerts per mission. Window AUPRC and AUROC remain supporting discrimination measures.

Temporal label smoothing and dynamic-survival objectives are promising secondary methods if the fixed binary-window formulation performs poorly near horizon boundaries. They should not be introduced after test inspection.

### 5. Calibration is operational, not cosmetic

Guo et al. show that modern neural networks can be miscalibrated and that post-hoc temperature scaling is often effective. Because a recovery decision attaches cost to a predicted probability, held-out calibration is part of the policy. Calibration candidates must be fit on validation episodes only and compared with Brier score, ECE, and reliability curves. The chosen calibration method and alarm threshold must be frozen together.

### 6. Recovery evidence requires closed-loop comparison

Learning a failure detector does not establish that it improves autonomy. Recovery research ranges from predefined guarded behaviors to learned corrective maneuvers. For this study, Nav2's Behavior Server and behavior trees provide a deliberately constrained action set—stop, wait, backup, spin/rescan, clear costmaps, and replan—while the guard layer retains authority.

The cleanest causal systems test is paired simulation: identical map, route, seed, and fault under default Nav2 recovery versus predictor-triggered recovery. A fixed conservative recovery isolates warning value; a rule-matched selector then separates diagnosis from learned action selection.

### 7. Recent work narrows, but does not erase, the proposed gap

A 2025–2026 primary-source refresh found two especially relevant navigation systems.
DR. Nav integrates RGB-LiDAR dead-end likelihood and recovery-point estimates into a
semantic cost map, while Xue et al. couple predicted obstacle motion to barrier-based
safe control. Both strengthen the case that prospective signals should alter navigation
before termination. They target particular geometric/environmental hazards, however,
rather than forecasting first terminal events across sensor, localisation, planning and
motion failure families from past-only system telemetry.

Nakamura et al.'s system-level regret formulation is adjacent evidence that errors should
be weighted by downstream robot consequence, and a 2026 uncertainty-aware place-
recognition study reinforces calibrated multimodal uncertainty under condition shifts.
These additions sharpen the novelty boundary: the contribution is not merely connecting
a predictor to recovery, but doing so under episode-protected causal labels, a fixed
alarm budget, unseen failure families and independent safety guards.

Schreiber et al.'s 2023 ROAR follow-on is particularly relevant to the planned camera-
occlusion analysis. It carries temporal state forward and explicitly models sensor
occlusion to reduce false positives during brief occlusions. This makes a single-
timestamp ablation and alert-burden reporting essential: a model must not equate an
injected sensor fault with an impending mission failure when the robot can continue.

## Provisional research gap

Existing work supports proactive, multimodal prediction of robot navigation anomalies, learned introspection, and recovery behaviors. The literature reviewed so far does not jointly demonstrate all of the following in one benchmark:

1. strictly causal past-only telemetry windows;
2. first-terminal-event labels with an explicit too-late interval;
3. protected evaluation by complete episode, map, route, and unseen failure family;
4. event recall at a validation-fixed per-mission false-alarm budget;
5. calibrated risk with alert persistence and cooldown; and
6. paired evidence that prediction-triggered, guard-constrained recovery improves mission completion without increasing collisions.

That conjunction is the proposed contribution. The claim must be weakened if the full benchmark cannot be executed.

## Evidence-to-design matrix

| Evidence | Design implication | Confirmatory status |
|---|---|---|
| PAAD predicts future failure from planned path and multimodal perception | Include planner state and multimodal features; compare against sensor-only ablations | Core |
| Introspective perception predicts component errors | Include health/uncertainty summaries, but label terminal mission outcomes independently | Core |
| TCNs are strong general sequence baselines | Use a causal TCN as P3 and matched recurrent model as P4 | Core |
| Reconstruction methods detect unusual multivariate sequences | Include a normal-only baseline under the same alarm policy | Supporting |
| Early-event work prioritizes event recall at low false-alarm burden | Evaluate first useful alert per event and false alerts per mission | Core |
| Neural scores may be miscalibrated | Fit calibration on validation only and freeze it with the threshold | Core |
| Guarded recovery actions already exist in Nav2 | Keep learned selection outside the safety guard and collision checking | Core |

## Threats that the literature review must continue to test

- Whether PAAD or a later study already evaluates unseen failure families under protected map splits.
- Whether robot-failure datasets expose event onset and episode grouping sufficiently for direct comparison.
- Whether false alerts per mission or per hour is the more stable budget across routes of different duration.
- Whether timeout should remain a terminal family or be treated as a censored/diffuse outcome.
- Whether a binary within-horizon target or discrete-time survival target gives the fairest early warning.
- Whether simulator fault signatures remain detectable after removing all injector metadata.

## Primary working references

1. Ji, T., Sivakumar, A. N., Chowdhary, G., & Driggs-Campbell, K. (2022). *Proactive Anomaly Detection for Robot Navigation with Multi-Sensor Fusion*. IEEE Robotics and Automation Letters. https://arxiv.org/abs/2204.01146
2. Ji, T., Vuppala, S. T., Chowdhary, G., & Driggs-Campbell, K. (2021). *Multi-Modal Anomaly Detection for Unstructured and Uncertain Environments*. CoRL/PMLR 155. https://proceedings.mlr.press/v155/ji21a.html
3. Daftry, S., Zeng, S., Bagnell, J. A., & Hebert, M. (2016). *Introspective Perception: Learning to Predict Failures in Vision Systems*. https://arxiv.org/abs/1607.08665
4. Rabiee, S., & Biswas, J. (2023). *Introspective perception for mobile robots*. Artificial Intelligence, 324, 103997. https://doi.org/10.1016/j.artint.2023.103997
5. Saxena, D. M., Kurtz, V., & Hebert, M. (2017). *Learning Robust Failure Response for Autonomous Vision Based Flight*. https://publications.ri.cmu.edu/storage/publications/2017/08/07989684.pdf
6. Audibert, J., Michiardi, P., Guyard, F., Marti, S., & Zuluaga, M. A. (2020). *USAD: UnSupervised Anomaly Detection on Multivariate Time Series*. KDD. https://www.eurecom.fr/publication/6271
7. Su, Y. et al. (2019). *Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network*. KDD. https://doi.org/10.1145/3292500.3330672
8. Bai, S., Kolter, J. Z., & Koltun, V. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. https://arxiv.org/abs/1803.01271
9. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*. ICML/PMLR 70. https://proceedings.mlr.press/v70/guo17a.html
10. Yèche, H., Pace, A., Rätsch, G., & Kuznetsova, R. (2023). *Temporal Label Smoothing for Early Event Prediction*. ICML/PMLR 202. https://proceedings.mlr.press/v202/yeche23a.html
11. Yèche, H., Burger, M., Veshchezerova, D., & Rätsch, G. (2024). *Dynamic Survival Analysis for Early Event Prediction*. CHIL/PMLR 248. https://proceedings.mlr.press/v248/yeche24a.html
12. ROS 2. *rosbag2*. https://github.com/ros2/rosbag2
13. Nav2. *Behavior Server*. https://docs.nav2.org/configuration/packages/configuring-behavior-server.html
14. Nav2. *Detailed Behavior Tree Walkthrough*. https://docs.nav2.org/behavior_trees/overview/detailed_behavior_tree_walkthrough
15. Nav2. *Simple Commander API*. https://docs.nav2.org/commander_api/index.html
16. Rajagopal, V. et al. 2025. *DR. Nav: Semantic-Geometric Representations for Proactive Dead-End Recovery and Navigation*. https://arxiv.org/abs/2511.12778
17. Xue, Y., Zhang, Z., Åkesson, K., & Figueroa, N. 2026. *Proactive Local-Minima-Free Robot Navigation: Blending Motion Prediction with Safe Control*. https://arxiv.org/abs/2601.10233
18. Nakamura, K., Tian, T., & Bajcsy, A. 2025. *Not All Errors Are Made Equal: A Regret Metric for Detecting System-level Trajectory Prediction Failures*. https://proceedings.mlr.press/v270/nakamura25a.html
19. *Sequential Probabilistic Descriptor via Uncertainty-Aware Multi-Modal Fusion for Safety-Critical Place Recognition*. 2026. IEEE Robotics and Automation Letters. https://doi.org/10.1109/LRA.2026.3669806
20. Schreiber, A., Ji, T., McPherson, D. L., & Driggs-Campbell, K. 2023. *An Attentional Recurrent Neural Network for Occlusion-Aware Proactive Anomaly Detection in Field Robot Navigation*. https://arxiv.org/abs/2309.16826

## Review closure for Protocol 1.0

The source search, targeted citation chaining, design-variable extraction, applicability
appraisal and core BibTeX set are complete for pre-model work. The review must be
refreshed immediately before submission, and any paper that directly matches the full
protected-map/unseen-family/paired-recovery design must be incorporated. See
`study-appraisal.md` for the structured quality and applicability assessment.
