# Evidence matrix: early robot failure warning and recovery

Updated: 31 August 2026

## Direct robot evidence

| Study | Task and input | Prospective target and split | Alarm/recovery evidence | Relevance and remaining gap |
|---|---|---|---|---|
| [Daftry et al. 2016](https://publications.ri.cmu.edu/introspective-perception-learning-to-predict-failures-in-vision-systems-2) | Vision-based MAV perception; input appearance predicts perception-system reliability | Component-level vision failure, rather than a multimodal terminal navigation event | Motivates remedial action but does not establish this protocol's event budget and paired Nav2 recovery outcome | Establishes learned introspection; supports perception-health features but not a mission-level claim |
| [Saxena, Kurtz & Hebert 2017](https://publications.ri.cmu.edu/learning-robust-failure-response-for-autonomous-vision-based-flight) | Autonomous quadrotor vision failure and experience-based corrective maneuvers | Situational perception failure linked to a recovery maneuver | Demonstrates learned maneuvers can improve recovery in cluttered flight | Closest early detection-to-action precedent, but vision-specific and not evaluated under held-out failure families |
| [Ji et al. 2022, PAAD](https://arxiv.org/abs/2204.01146) | Field ground robot; camera, 2D LiDAR and planned path | Predicts failure over 10 future steps; training and test came from independent days | Online policy used three consecutive scores over 0.5 at 10 Hz and reported detected anomalies and false detections | Closest task comparator. It supports multimodal/planner inputs and persistence, while leaving room for episode/map/family protection, calibrated probabilities, explicit lead time, and autonomous recovery |
| [Schreiber et al. 2023, ROAR](https://arxiv.org/abs/2309.16826) | Field robot sensory inputs, planned controls, recurrent state and explicit sensor-occlusion estimate | Proactive navigation-failure prediction with temporal state; evaluates synthetic total sensor occlusion | Reduces false positives under brief occlusion relative to PAAD in reported examples; no autonomous recovery comparison | Directly motivates temporal history, camera-occlusion non-event negatives and the no-persistence/single-timestamp ablations |
| [Farid et al. 2022](https://www.roboticsproceedings.org/rss18/p042.html) | Histories of high-dimensional observations for drone navigation and manipulation | Failure predictor with PAC-Bayes and class-conditional false-positive/false-negative bounds; simulation and hardware | Demonstrates safety improvement through predicted failures | Strong false-alarm/false-negative framing; this protocol uses empirical hierarchical intervals rather than claiming formal bounds |
| [Mohammad, Higgins & Bezzo 2024](https://arxiv.org/abs/2402.01617) | GP features from a receding-horizon safe-corridor planner | Predicts future planner-backend failure; tests include new simulated environments and simulation-to-real platforms | Thresholded risk stops the robot and selects a low-risk recovery point; simulation and physical case studies show continuation | Strongest proactive prediction-plus-recovery comparator, but targets one planner failure mechanism instead of seven system-level families |
| [Ji et al. 2021](https://proceedings.mlr.press/v155/ji21a.html) | Multimodal robot anomaly detection in unstructured environments | Primarily reactive anomaly detection | Detection evidence, no prospective recovery claim | Justifies multimodal baselines but should not be treated as proof of early warning |
| [Rajagopal et al. 2025, DR. Nav](https://arxiv.org/abs/2511.12778) | RGB-LiDAR semantic-geometric dead-end likelihood and recovery-point mapping | Proactively represents dead-end risk in unmapped dense environments | Recovery-aware cost map steers planning away from anticipated dead ends | Strong prediction-to-planning comparator, but targets dead ends rather than multimodal terminal system failures and does not replace event-budget evaluation |
| [Xue et al. 2026](https://arxiv.org/abs/2601.10233) | Neural multimodal obstacle-motion prediction coupled to learned barrier functions | Predicts dynamic obstacle motion to avoid future collisions/local minima | Closed-loop safe controller evaluated in simulation and real experiments | Supports forecast-informed guard design; it predicts environment motion rather than robot failure from internal telemetry |
| [2026 uncertainty-aware place recognition](https://doi.org/10.1109/LRA.2026.3669806) | Sequential image-LiDAR probabilistic embeddings with calibrated uncertainty | Detects high-uncertainty place-recognition outputs under long-term condition shifts | Downstream rejection/filtering rather than mission recovery | Supports multimodal uncertainty and held-out-condition calibration, but remains a component-level reliability study |

## Temporal modelling, calibration, and evaluation evidence

| Study | Evidence | Protocol decision |
|---|---|---|
| [Bai, Kolter & Koltun 2018](https://arxiv.org/abs/1803.01271) | Generic temporal convolutions can be strong sequence-model baselines relative to canonical recurrent networks | Use a compact causal TCN as the primary learned architecture and a matched GRU/LSTM comparator; architecture is not the contribution |
| [USAD, Audibert et al. 2020](https://www.kdd.org/kdd2020/accepted-papers/view/usad-unsupervised-anomaly-detection-on-multivariate-time-series.html) | Adversarially trained autoencoders provide an unsupervised multivariate anomaly score | Include a normal-only reconstruction baseline under exactly the same alarm policy and event evaluator |
| [OmniAnomaly, Su et al. 2019](https://doi.org/10.1145/3292500.3330672) | Stochastic recurrent reconstruction probabilities model multivariate telemetry anomalies | Include only as a supporting anomaly baseline; reconstruction anomaly is not equivalent to terminal-event risk |
| [Garg et al. 2022](https://doi.org/10.1109/TNNLS.2021.3105827) | Event detection matters more than isolated anomalous points; trivial detectors can exploit unsuitable point metrics; scoring can matter as much as architecture | Make event recall, false alerts per mission, useful lead time, and alert burden primary; retain window AUPRC/AUROC only as supporting metrics |
| [Guo et al. 2017](https://proceedings.mlr.press/v70/guo17a.html) | Modern neural classifiers may be miscalibrated; post-hoc temperature scaling is an effective simple candidate | Fit calibration only on validation episodes and freeze it with the threshold; report Brier score, ECE, and reliability curves |
| [Yèche et al. 2023](https://proceedings.mlr.press/v202/yeche23a.html) | Temporal label structure matters for early-event prediction | Keep temporal smoothing as a prespecified secondary method only if hard within-horizon labels prove unstable |
| [Yèche et al. 2024](https://proceedings.mlr.press/v248/yeche24a.html) | Dynamic survival analysis offers an alternative formulation for early event prediction | Treat survival modelling as a declared secondary analysis, not a post-test rescue |
| [Nakamura, Tian & Bajcsy 2025](https://proceedings.mlr.press/v270/nakamura25a.html) | System-level probabilistic regret distinguishes prediction errors that actually degrade closed-loop robot performance and supports calibration across contexts | Keep mission outcome and recovery regret downstream of the warning score; component error alone is not the target |

## Platform evidence

| Source | Verified implication |
|---|---|
| [ROS 2 rosbag2](https://github.com/ros2/rosbag2) | Bags store timestamped messages; simulation-time recording must receive `/clock` before writing. The installed Jazzy CLI is checked locally because rolling documentation contains options not present in the installed release. |
| [Nav2 Behavior Server](https://docs.nav2.org/) | Recovery execution should use bounded Nav2 behaviors and preserve collision checking. Exact Jazzy interfaces are verified from the installed platform before live use. |

## Defensible novelty boundary

The literature supports each ingredient separately: proactive multimodal warning,
observation-history failure prediction, planner-risk recovery, temporal anomaly models,
and calibrated decision scores. The intended contribution is their controlled
combination in one mobile-robot benchmark:

1. strictly past-only multimodal telemetry;
2. first-terminal-event labels with a no-credit too-late interval;
3. episode/map/route protection and leave-one-failure-family-out evaluation;
4. event recall at a validation-fixed per-mission false-alert budget;
5. explicit missingness, calibration, persistence and cooldown; and
6. paired mission outcomes under an independent recovery guard.

The novelty claim must be narrowed if any of these controls is removed. In particular,
model complexity, random-window accuracy, or a successful recovery video alone is not a
research contribution.
