# Validation-only diagnostics (exploratory)

Generated 2026-09-06T22:58:26.992904+00:00. Frozen threshold 0.2351; no protected data used.

## D1 Precursor structure per family (validation events, primary seed)

| Family | Events | Median s from injection end to window start | Window overlaps injection | Median warnable decisions | Median window max risk | Median pre-onset max risk | Window max >= threshold | Detected by frozen policy |
|---|---|---|---|---|---|---|---|---|
| camera_occlusion | 6 | 25.54 | 0.0 | 18.0 | 0.271 | 0.007 | 1.0 | 1.0 |
| dynamic_blockage | 10 | 105.25 | 0.0 | 18.0 | 0.194 | 0.034 | 0.3 | 0.3 |
| lidar_dropout | 6 | 43.05 | 0.0 | 18.0 | 0.099 | 0.032 | 0.0 | 0.0 |
| localisation_perturbation | 8 | 51.76 | 0.375 | 18.0 | 0.061 | 0.035 | 0.0 | 0.0 |
| none | 11 | None | None | 18 | 0.224 | None | 0.455 | 0.364 |
| planner_oscillation | 11 | 60.7 | 0.0 | 18 | 0.237 | 0.035 | 0.545 | 0.455 |
| semantic_corruption | 5 | 30.81 | 0.0 | 18 | 0.272 | 0.011 | 0.6 | 0.6 |
| wheel_slip | 5 | 23.9 | 0.0 | 18 | 0.088 | 0.029 | 0.0 | 0.0 |

## D2 Threshold-selection stability

200 hierarchical resamples of the validation routes/episodes; threshold re-selected each time with the frozen rule.

- Selected threshold p05/p50/p95: 0.1278 / 0.251 / 0.3141 (frozen 0.2351); never-alarm chosen in 0.02 of replicates
- Resampled thresholds applied to the full validation set: false alerts per clean mission p05/p50/p95 = 0.0 / 0.0833 / 0.2639; exceed the 0.10 budget in 0.36 of replicates; recall p05/p50/p95 = 0.1129 / 0.2581 / 0.5968
- Frozen threshold on resamples: false alerts per clean mission p05/p50/p95 = 0.0 / 0.0946 / 0.2239; exceeds budget in 0.46

## D3 Selection optimism (leave one validation map out)

| Left-out map | Selected seed | Recall on selection maps | Recall on left-out map | FA/clean on left-out map |
|---|---|---|---|---|
| val_00 | 20260904 | 0.500 | 0.060 | 0.000 |
| val_01 | 20260904 | 0.534 | 0.750 | 0.375 |
| val_02 | 20260904 | 0.204 | 0.625 | 0.000 |

Mean recall: 0.413 on selection maps versus 0.478 on the left-out map (optimism -0.066); mean FA/clean on left-out maps 0.125. In-sample frozen reference: recall 0.339 at 0.097 FA/clean; held-out maps gave 0.229 at 0.183.
