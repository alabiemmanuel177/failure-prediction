# Provisional machine-assisted comparison of the frozen audit packet

Status: supporting evidence only; not a human-review attestation and not counted by Gate G1.

Recorded: 31 August 2026  
Scope: 20 frozen audit episodes  
Result: provisional agreement on 20/20; no specific causal-signature disagreements.

## Qualifications

- Camera-occlusion and semantic-corruption injections lack a direct camera or
  semantic-health trace in the displayed four-panel timeline. Agreement relies on the
  recorded injection marker, operational evidence, and absence of contradictory
  terminal evidence.
- The medium planner-oscillation episode records a navigation abort at 189.036 seconds,
  long after its injection ended at approximately 35.8 seconds. The navigation abort is
  supported, but this audit does not establish that the temporary injection caused the
  later abort.
- The review UI originally displayed a dash for severity because the campaign summary's
  `label_only` severity was null. The frozen automatic injection annotations contain the
  expected low, medium, or high severity. This was a presentation defect, not a causal
  signature disagreement.

## Episode comparison

| # | Episode | Comparison | Confidence | Evidence summary |
|---:|---|---|---|---|
| 1 | camera_occlusion-high-201 | Agree | Moderate | Eligible high occlusion at 19.800 s; success; no terminal candidate. Direct camera-health telemetry is not displayed. |
| 2 | camera_occlusion-low-201 | Agree | Moderate | Eligible low occlusion at 20.100 s; success; no terminal candidate. Direct camera-health telemetry is not displayed. |
| 3 | camera_occlusion-medium-201 | Agree | Moderate | Eligible medium occlusion at 19.800 s; success; no terminal candidate. Direct camera-health telemetry is not displayed. |
| 4 | control-s0-202 | Agree | High | No injection, event, or candidate; successful termination with complete motion and odometry evidence. |
| 5 | lidar_dropout-high-201 | Agree | High | High dropout at 15.702 s with direct valid-return decline; no terminal event; success. |
| 6 | localisation_perturbation-medium-201 | Agree | High | Perturbation at 17.502 s, covariance increase, and false arrival at 30.657 s supported by the event topic and label-only ground-truth pose. |
| 7 | control-s3-201 | Agree | High | No injection, event, or candidate; success. |
| 8 | semantic_corruption-medium-201 | Agree | Moderate | Eligible semantic corruption at 20.202 s; no event or candidate; success. No semantic-health channel is displayed. |
| 9 | planner_oscillation-medium-201 | Agree, qualified | Moderate–high | Injection begins at 15.702 s and lasts 20.1 s. Navigation abort at 189.036 s is supported, but the delay prevents a causal attribution claim. |
| 10 | lidar_dropout-medium-201 | Agree | High | Medium dropout at 16.200 s with direct valid-return decline; no event; success. |
| 11 | semantic_corruption-low-201 | Agree | Moderate | Eligible corruption at 19.701 s; no event or candidate; success. No semantic-health channel is displayed. |
| 12 | wheel_slip-low-201 | Agree | High | Slip at 15.984 s with command-to-motion separation; no terminal candidate; success. |
| 13 | lidar_dropout-low-201 | Agree | High | Low dropout at 15.702 s with expected valid-return step; no event; success. |
| 14 | dynamic_blockage-low-201 | Agree | Moderate–high | Blockage at 15.702 s for about 4.098 s with clearance and motion changes; no event; success. |
| 15 | planner_oscillation-low-201 | Agree | Moderate | Eligible oscillation at 15.702 s; motion continues; no event or candidate; success. Local-plan choice oscillations are not shown directly. |
| 16 | localisation_perturbation-low-201 | Agree | High | Perturbation at 15.702 s with covariance increase but no threshold-qualified localisation loss; success. |
| 17 | control-s0-201 | Agree | High | No injection, event, or candidate; success. |
| 18 | dynamic_blockage-medium-201 | Agree | Moderate–high | Blockage at 16.101 s for about 10.101 s with clearance and motion changes; no event; success. |
| 19 | control-s3-202 | Agree | High | No injection, event, or candidate; success. |
| 20 | wheel_slip-medium-201 | Agree | High | Slip at 16.056 s with substantial command-to-motion separation; no terminal candidate; success. |

## Admission effect

None. A named person must still inspect and attest to every episode in the review app,
and is not an independent human review. Under PA-2026-09-03-01, the completed primary
human review passes Gate G1; no inter-rater reliability is claimed.
