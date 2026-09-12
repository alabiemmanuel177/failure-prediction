# tab01_predictor_summary

Generated 2026-09-12T07:59:19Z from immutable artifacts.

| split | model_id | episodes | events | detected_events | event_recall | false_alerts_per_clean_mission | false_alerts_per_mission | median_useful_lead_seconds | brier_raw | ece_raw | brier_calibrated | ece_calibrated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| validation | p3_causal_tcn | 324 | 62 | 21 | 0.339 | 0.097 | 0.207 | 5.701 | 0.048 | 0.045 | 0.030 | 0.005 |
| held_out_map_test | p1_threshold_rules | 1002 | 157 | 5 | 0.032 | 0.468 | 0.332 | 7.105 | 0.041 | 0.041 | 0.041 | 0.041 |
| held_out_map_test | p3_causal_tcn | 1002 | 157 | 36 | 0.229 | 0.183 | 0.230 | 3.818 | 0.034 | 0.041 | 0.016 | 0.008 |
| held_out_map_test | p4_gru | 1002 | 157 | 31 | 0.197 | 0.063 | 0.083 | 3.811 | 0.034 | 0.038 | 0.016 | 0.006 |
| held_out_map_test | p5_compact_transformer | 1002 | 157 | 42 | 0.268 | 0.254 | 0.264 | 5.842 | 0.038 | 0.042 | 0.016 | 0.009 |
