# Hypothesis table

Frozen threshold: 0.235; model freeze sha256: b80b4e8ba92cef6bc6b9d55e0d9949622e244f2eb710b418cddfb645196117c4.

| ID | Label | Hypothesis | Estimate | 95% CI | Status | Source |
|---|---|---|---|---|---|---|
| H1 | confirmatory | P3 event recall exceeds P1 at the validation-fixed false-alert budget | 0.197 (paired P3 - P1 event recall difference) | [-0.000, 0.447] | not_supported | reports/confirmatory/held_out_map.final.yaml:h1 |
| H2 | supporting | median useful lead time of detected failures >= 3 s | 3.818 (median lead seconds (detected events)) | [1.745, 6.120] | supported | reports/confirmatory/held_out_map.final.yaml:h2 |
| H3 | supporting | calibration reduces Brier score and ECE on validation data | -0.019 (validation Brier delta (after - before)) | n/a | supported | reports/confirmatory/held_out_map.final.yaml:h3 |
| H4 | supporting | removing planner and localisation health features materially reduces early warning | 0.102 (largest recall decline among planner/localisation ablations) | n/a | reported_no_numeric_threshold | reports/confirmatory/held_out_map.final.yaml:h4 (reports/ablations) |
| H5 | supporting | leave-one-family-out P3 recall exceeds P1 for at least five of seven families | 4 (families where P3 > P1 (of seven)) | n/a | not_supported | reports/confirmatory/unseen_family.yaml:h5 |
| H6 | confirmatory | prediction-triggered recovery improves mission completion without more collisions | n/a (paired completion difference (R3 - R0)) | n/a | not_supported | reports/recovery/paired_recovery.yaml:h6 (owned by the recovery work package) |

H1 and H6 are the two confirmatory claims; H2-H5 are prespecified supporting hypotheses; every other item is exploratory and not an independent significance claim

## Exploratory items (flagged; not claims)

| Item | Estimate | 95% CI | Source |
|---|---|---|---|
| p4 versus P1 event recall | 0.166 | [-0.036, 0.458] | reports/confirmatory/held_out_map.final.yaml:exploratory_models |
| p5 versus P1 event recall | 0.236 | [0.043, 0.421] | reports/confirmatory/held_out_map.final.yaml:exploratory_models |
| p3 recall by severity | {'medium': 0.22695035460992907, 'none': 0.25} | n/a | reports/confirmatory/held_out_map.final.yaml:models |
| p1 recall by severity | {'medium': 0.03546099290780142, 'none': 0.0} | n/a | reports/confirmatory/held_out_map.final.yaml:models |
| p4 recall by severity | {'medium': 0.19858156028368795, 'none': 0.1875} | n/a | reports/confirmatory/held_out_map.final.yaml:models |
| p5 recall by severity | {'medium': 0.2624113475177305, 'none': 0.3125} | n/a | reports/confirmatory/held_out_map.final.yaml:models |
| p3 mission-timeout recall (analysed separately) | 0.000 | n/a | reports/confirmatory/held_out_map.final.yaml:timeouts |
| p1 mission-timeout recall (analysed separately) | 0.000 | n/a | reports/confirmatory/held_out_map.final.yaml:timeouts |
| p4 mission-timeout recall (analysed separately) | 0.000 | n/a | reports/confirmatory/held_out_map.final.yaml:timeouts |
| p5 mission-timeout recall (analysed separately) | 0.167 | n/a | reports/confirmatory/held_out_map.final.yaml:timeouts |
| p3_causal_tcn recall on natural (no-injection) failures | 0.364 | n/a | reports/confirmatory/natural_failure_audit.yaml |
