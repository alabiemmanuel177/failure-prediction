# Dataset card: Research 2 navigation failure forecasting

Status: development and validation collection complete and derived; targeted collection
and the preregistered 504-episode supplement in progress; training admission passed
under Protocol Amendment PA-2026-09-03-01; no model frozen. Independent review remains optional and no inter-rater reliability is
claimed.

## Intended use

This dataset supports causal prediction of the first mission-ending navigation event
within a fixed future horizon from past-only mobile-robot telemetry. It may also support
event-level threshold baselines, calibration studies, feature-group ablations and
guarded recovery evaluation. It is not evidence of real-world safety certification.

## Independent unit and splits

The independent unit is a complete navigation episode. Windows from an episode cannot
cross partitions. Development and validation assignments are frozen before extraction;
protected map identities remain unavailable until the complete prediction and analysis
policy is frozen. Leave-one-failure-family-out folds exclude that family from fitting,
calibration and threshold selection.

## Collection design

The preregistered balanced pilot contains 648 development-only episodes across six maps,
two routes per map, six replicates, two clean controls and seven deterministic fault
families at medium severity. Environment faults are placed at 55% of the active global
path. Every attempted episode and invalid artifact is retained. Raw development
collection is governed by the extraction gate. Any feature extraction admitted to
model fitting, model training, calibration or threshold selection remains blocked until
supervisor threshold approval and exact independent review agreement on the frozen
20-episode label audit.

The recording profile is provenance, not a model feature. `full_v1`, `compact_v1`, and
`compact_v2` share the primary numeric feature contract; the newest profile replaces
semantic images with synchronized online scalar reductions. Two pre-goal zero-bag
infrastructure invalids are retained and each is linked to one same-cell, same-seed
replacement under separate campaigns. Final counts distinguish attempts, invalid
attempts, replacements and independent scientific episodes.

Summary and event-sidecar publication is exclusive, atomic and `fsync`-backed. A
separate raw-payload audit fails if any zero-length summary, MCAP or bag metadata is not
declared as an excluded infrastructure-loss run. A historically successful ledger row
does not override missing replay evidence. Treatment-delivery failures and payload loss
use separately preregistered replacement policies; each scientific design cell still
contributes at most once.

The balanced development inventory contains 648 scientific episodes, 125 terminal
events (19.29%) and 523 non-events. It retains 59 false arrivals, 53 navigation aborts
and 13 mission timeouts. Each development map has 108 episodes; each of the 12 routes
has 54. Each injected family has 72 episodes and the two clean-control cells contribute
144 episodes.

The retained development bags total 57,119,425,868 bytes (53.20 GiB) and 41,701.374
seconds. Profile counts are 54 `full_v1`, 42 `compact_v1`, and 552 `compact_v2`. The
hash-addressed dataset identifier is `balanced_pilot_v1-development-648`; its immutable
648-row inventory is `data/manifests/balanced_pilot_v1.episodes.jsonl` with SHA-256
`4b0812acef2da291ecc83113e1e251ecb1dab1b4c4c88e0b46159242d0b1100a`.

A separate 324-episode validation campaign is complete across three validation maps,
two routes per map, six replicates, two clean controls and seven fault families. Its
immutable episode inventory and retained payloads validate. Model selection, calibration
and threshold selection remain forbidden until the human training-admission gate passes.

The post-pilot planning audit found only 5–10 terminal events in four injected families
and preregistered 1,212 route-balanced development additions. The planning rule raises
each injected family to approximately 30 expected terminal events at its observed pilot
prevalence; planner oscillation already exceeds that floor. The split-corrected audit
identifies 825 structurally eligible Research 1 development bags and 364 validation
bags. Validation remains selection-only. If all 825 development bags later pass human
admission and causal adaptation, the maximum fitting pool before supplementation is
2,685 episodes. A route-balanced supplement of at least 324 development episodes is
then required to reach the 3,000-episode protocol target. Neither the 30-event planning
floor nor structural bag eligibility is a performance claim.

## Labels and leakage controls

Primary labels use only the first terminal event. A five-second history ending at the
decision time is positive from ten to one seconds before that event; later decisions
receive no early-warning credit. Eligible negatives remain at least twenty seconds from
event and injection onset. Simulator ground truth, fault commands and parameters,
injection time, terminal results, recovery outcomes and future samples are physically or
logically label-only and denied to deployable features.

## Known limitations

- The benchmark uses one simulated differential-drive robot family and controlled faults.
- Some injected faults may not produce a terminal event; these remain valid non-events.
- Mission timeout can have diffuse precursors and is reported separately.
- Simulation signatures may not transfer to physical hardware.
- Natural failures are audited separately and never silently pooled with injected events.

## Fields still to populate after admitted extraction and the study

- Decision-time feature missingness and age distributions.
- Useful-warning support by family after causal windows are admitted.
- Natural-failure inventory and any protocol deviations.
- Exact software/container versions, license, access restrictions and retention policy.

Generate the episode-level starting inventory with:

```bash
python3 scripts/validate_development_dataset_inventory.py
```

The command recomputes every source and episode-artifact hash and fails on any mismatch.
Run `make raw-payload-check` first to reconcile zero-length artifact evidence against
the frozen replacement declarations.
