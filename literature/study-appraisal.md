# Study applicability appraisal

Frozen for Protocol 1.0: 1 September 2026

This appraisal addresses applicability to the Research 2 claim. It is not a pooled
risk-of-bias score: the studies use different robots, targets, horizons and outcomes.
`Yes` means the paper directly supplies the property, `partial` means an adjacent or
narrower form, `no` means it does not, and `unclear` means the accessible primary report
does not establish it.

| Study | Prospective robot failure target | Temporal/past-only input clear | Independent environment or episode split | Operational false-alarm evidence | Explicit warning timing | Closed-loop recovery | Hardware evidence | Applicability |
|---|---|---|---|---|---|---|---|---|
| Daftry et al. 2016 | Partial: component failure | Partial | Partial | No | No | No | Yes | Supports introspective features, not terminal-event evaluation |
| Saxena et al. 2017 | Yes, vision-flight failure | Partial | Partial | No | Partial | Yes | Yes | Strong detection-to-correction precedent; vision-specific |
| Ji et al. 2022, PAAD | Yes | Yes | Partial: independent days | Yes | Partial: future steps | No: intervention opportunity only | Yes | Closest multimodal navigation-warning comparator |
| Schreiber et al. 2023, ROAR | Yes | Yes | Partial | Yes, including occlusion false positives | Partial | No | Yes | Closest temporal/occlusion comparator |
| Farid et al. 2022/2023 | Partial: autonomy prediction failure | Yes | Partial | Yes, bounded rates | No mission lead time | Partial: safety monitor | Simulation/hardware across related work | Strong consequence-aware monitoring evidence |
| Mohammad et al. 2024 | Yes, planner failure | Yes | Yes: test worlds and sim-to-real | Partial: thresholded risk | Yes: future corridor | Yes | Yes | Strongest proactive prediction-plus-recovery comparator; one failure mechanism |
| Rabiee & Biswas 2021/2023 | Partial: perception/SLAM reliability | Yes | Yes | Partial | No mission lead time | Partial: estimator integration | Yes | Supports localisation/perception health and downstream consequence |
| Rajagopal et al. 2025 | Partial: dead-end risk | Yes | Yes | Unclear | Yes | Yes, recovery-aware planning | Reported robot navigation evidence | Adjacent proactive environmental-hazard method |
| Xue et al. 2026 | Partial: future local minima/collision | Yes | Yes | Unclear | Yes | Yes, safe control | Yes | Supports forecast-informed guard design, not internal failure telemetry |

## Appraisal conclusion

The closest work establishes that proactive multimodal warning, temporal occlusion
reasoning, planner-failure prediction and autonomous recovery are individually credible.
No reviewed study establishes the complete conjunction of first-terminal-event labels,
episode/map protection, leave-one-failure-family-out evaluation, validation-fixed false
alerts per mission, calibrated useful lead time and paired guard-constrained recovery.
That conjunction remains the scoped contribution and must be tested rather than claimed.
