# Literature search log

Protocol version: 1.0  
Search updated: 31 August 2026
Review type: structured scoping review; not a systematic-review claim

## Eligibility

Include primary research or official platform documentation that directly informs at
least one of: prospective robot failure prediction, multimodal robot introspection,
multivariate temporal anomaly detection, early-event evaluation, probability
calibration, or autonomous recovery. Prefer work with a stated future horizon, false
alarm reporting, independent splits, or closed-loop outcomes.

Exclude surveys as evidence for model performance, component fault diagnosis with no
prospective warning task, papers that evaluate only randomly shuffled time points, and
work for which the failure target or temporal direction cannot be determined. Surveys
may be used only for citation discovery.

## Searches recorded

| Date | Source | Query family | Purpose |
|---|---|---|---|
| 2026-08-24 | arXiv, PMLR, CMU RI, official ROS/Nav2 sites | robot navigation + failure prediction/proactive anomaly/introspection/recovery | Initial protocol scoping |
| 2026-08-28 | arXiv, RSS proceedings, IEEE/ACM landing pages, PMLR, CMU RI | history of sensor observations + robot failure; proactive navigation anomaly + multisensor; planner failure + recovery | Direct robot comparator update |
| 2026-08-28 | KDD/ACM, PMLR | multivariate time-series anomaly + event evaluation; TCN; calibration | Baseline and metric justification |
| 2026-08-28 | ROS 2 and Nav2 official repositories/documentation | rosbag2 loss statistics; behavior server; behavior trees | Implementation-interface verification |
| 2026-08-31 | arXiv, PMLR, IEEE Xplore, OpenReview | robot navigation + proactive failure prediction/recovery, restricted to 2025–2026 | Recent-primary-source novelty refresh |

Search results were screened by title and abstract, then full text when needed to
extract prediction horizon, split unit, alarm policy, and recovery evidence. No numeric
result is imported into the Research 2 protocol unless verified in the primary source.

## Citation-chaining priorities

1. Forward citations to PAAD that evaluate independent maps, episodes, or fault families.
2. Backward and forward citations to Farid et al. for class-conditional failure bounds.
3. Work comparing prospective terminal-event prediction against reconstruction anomaly scores.
4. Robot studies reporting both warning lead time and intervention burden.
5. Closed-loop recovery studies with paired seeds and an independent action-safety layer.

The 31 August refresh retained DR. Nav (dead-end risk and recovery-aware planning),
Xue et al. (prediction-coupled safe control), Nakamura et al. (system-level calibrated
failure regret), and a 2026 uncertainty-aware place-recognition study as adjacent work.
None replaces the direct PAAD comparator or demonstrates this protocol's full joint
combination of seven failure families, causal episode splits, alarm burden, useful lead
time and paired guarded recovery.

On 1 September 2026, targeted forward chaining from PAAD added Schreiber et al.'s ROAR
as a direct occlusion-aware follow-on. Searches also screened task-relevant prediction-
failure monitors, proactive planner-failure recovery, OOD fallback planning and generic
proactive time-series forecasting. The latter three categories were retained only when
they changed a protocol decision; generic forecasting papers without robot navigation
evidence were excluded from the direct-comparator set.

## Frozen review limitations

- This is a targeted structured scoping review; exhaustive database result counts and a
  PRISMA-style duplicate inventory were not claimed or produced.
- IEEE full-text access is incomplete for some records; claims are limited to accessible
  primary abstracts or author manuscripts.
- The search is English-language and method-focused.
- Heterogeneous engineering studies were assessed with a design-applicability matrix
  rather than a clinical risk-of-bias instrument; see `study-appraisal.md`.
- Literature findings cannot substitute for the study's protected-map or unseen-family tests.
