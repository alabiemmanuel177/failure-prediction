# Early Failure Prediction and Recovery for Mobile Robot Navigation

Research 2 is a causal early-warning and guarded-recovery study built as an overlay on
the pinned Research 1 platform in `/home/eao/risk-calibrated-nav`.

## Current state

The structured literature review, failure taxonomy, event contract, causal label and
feature pipelines, transparent threshold baseline, event-level evaluator, replay-only
recovery guards, recommendation-only ROS recovery manager, Research 1 integration,
retained MCAP recording, and health
reconciliation are implemented. All seven fault families are frozen after a 21/21
family/severity treatment-integrity gate. Research 2 is isolated on ROS domain 52.

Research 1 completed G7 on 30 August 2026. Research 2 retained the definitive 25-attempt
development integrity campaign and prepared the frozen 20-episode audit sample. All 20
episodes have automatic annotations, causal labels, and audit-only features. Feature
extraction for model fitting and model development remain fail-closed until supervisor
threshold sign-off and two distinct people independently review those annotations with
exact causal-field agreement.
The 648-episode balanced pilot is preregistered in
`data/manifests/balanced_pilot_v1.yaml`; immutable development-only raw collection is
allowed under the extraction gate and proceeds independently of the human training
gate. Protected-map collection remains prohibited until the complete model and
analysis freeze.

The development campaign is complete at 648/648 artifact-usable scientific episodes:
125 terminal events and 523 non-events across the frozen six-map design. Two pre-goal
zero-bag infrastructure invalids are retained and each is replaced exactly once by a
same-cell, same-seed episode under a linked replacement campaign. The immutable episode
inventory and checksums are in `data/manifests/balanced_pilot_v1.dataset.yaml`. The
development collection occupies 53.20 GiB and contains 54 `full_v1`, 42 `compact_v1`,
and 552 `compact_v2` bags. Validation-only raw collection is complete at 324/324 and its
hash-addressed inventory is `data/manifests/balanced_validation_v1.dataset.yaml`; it
cannot select models, calibration, or thresholds before the human training-admission
gate passes.

Research 1 reuse is staged through the outcome-blind structural catalog at
`data/manifests/research1_temporal_catalog_v2.jsonl`: 825 development and 364
validation bags meet the five-topic temporal contract, zero protected-test rows are
exported, and non-required topic names are omitted. This catalog is structural evidence
only and is not admitted to any model pipeline before the human and causal-adapter gates.

The split-corrected post-pilot sizing decision is frozen in
`reports/pilot/post_pilot_campaign_size_v2.yaml`; the superseded v1 report remains
archived with its correction record in
`reports/pilot/training_volume_split_correction_v1.yaml`. It preregisters 1,212
additional development episodes for the six families below a 30-event planning floor.
The complete 12-route design is `data/manifests/targeted_development_v1.yaml`; collection
started only after validation finalization and continues only while the prospective
100 GiB storage reserve passes. Only the 825 structurally
eligible Research 1 **development** bags may contribute to fitting; the 364 Research 1
validation bags remain selection-only. If all 825 pass the human gate and causal adapter
audit, the maximum fitting pool before a supplement is 2,685 episodes, leaving a
route-balanced minimum supplement of 324 development episodes to reach the 3,000-episode
protocol target. The 30-event floor is campaign planning—not a confirmatory claim.

### Model-development stage (opened 3 September 2026)

Training admission passed under Protocol Amendment PA-2026-09-03-01. The offline
causal pipeline now runs over whole inventories (`scripts/extract_dataset_sequences.py`)
and publishes hash-addressed derived artifacts under `data/derived/<dataset_id>/`:
fitting sequences (eligible decisions only) and all-decisions artifacts (every 2 Hz
decision, so the alarm policy replays the deployed stream). The development (648) and
validation (324) inventories are derived. A feature-definition correction made before
any model fit (goal distance now uses the map-frame AMCL pose instead of odom-frame
coordinates) is recorded in the research log; superseded derived artifacts are kept
under `data/derived/_superseded_20260903_odom_frame_goal/`.

The Research 1 causal-adapter audit (`reports/integrity/research1_causal_adapter_audit_v1.yaml`)
admits 647 of 825 structurally eligible development bags; 646 of them end in a terminal
event because Research 1 retained failure bags, so under Protocol Amendment PA-2026-09-03-03 they are
excluded from the frozen fitting pool (2,364 Research 2 development episodes) and kept
as a separately reported natural-failure and domain-shift audit set. The route-balanced development supplement
is preregistered as 504 episodes in `data/manifests/development_supplement_v1.yaml`
and starts automatically after the targeted campaign is finalised
(`scripts/await_campaign_and_derive.py`).

Predictors P1–P6, calibration, threshold, comparison, ablation, freeze, confirmatory,
recovery and release tooling are implemented under the contract in
[docs/model-development-contracts.md](docs/model-development-contracts.md) and driven by
`scripts/run_model_development.py`. A preliminary validation-only pass on the 648/324
pool (`reports/model_selection/preliminary_v1/`) is recorded as a rehearsal: nothing is
frozen, and the final pass reruns on the complete fitting pool. Model fitting uses
`.venv/bin/python` (PyTorch ROCm on the workstation's Radeon AI PRO R9700). Protocol
Amendment PA-2026-09-03-02 fixes six routes per held-out map and raises seeds so the
confirmatory, severity-stress and paired-recovery campaigns keep their preregistered
minimum sizes.

See [integration/STATUS.md](integration/STATUS.md) for the current gate status and
[docs/logging-labeling-taxonomy.md](docs/logging-labeling-taxonomy.md) for the annotation
workflow. The pre-pilot [dataset card](docs/dataset-card.md) records intended use,
leakage controls and fields that must be populated from immutable pilot artifacts.

## Verify

```bash
make verify-premodel
make ros-build
```

The pre-model verification target tests the code, Research 1 content boundary,
extraction configuration, pilot expansion and tamper-evident research log. The
extraction readiness command passes. `python3 scripts/check_manual_audit_gate.py`
intentionally fails until reviewed copies are placed in `data/annotations/reviewed/`.
Protected routes remain deliberately unassigned and must not be inspected before model,
calibration, alarm policy, and analysis freeze.

## Review the frozen audit packet

Launch the localhost-only evidence review interface:

```bash
make review-audit
```

The interface records append-safe primary and optional second-reviewer attestations for
all 20 episodes. It never displays model output or protected-map evidence. Under
Protocol Amendment PA-2026-09-03-01, the completed 20/20 researcher review and
prospective researcher threshold approval admit model development. No inter-rater
reliability is claimed. Any later second reviewer must still be a different real person.

Development or validation collection advances in exact-once 18-episode waves with
immutable per-wave and cumulative reports:

```bash
python3 scripts/run_balanced_pilot_continuous.py \
  --manifest data/manifests/balanced_validation_v1.yaml
```

The controller stops immediately on an invalid artifact, an active Research 1 campaign,
or the manifest's 100 GiB free-space reserve. It does not relax the human training gate
or assign protected routes.

Campaign health can be reconciled without interacting with ROS or changing experimental
state using `make monitor-validation`. The active workstation also has a transient
user-level timer running that check every 30 minutes. Its report is
`reports/status/balanced_validation_monitor.yaml`; see
[docs/campaign-monitoring.md](docs/campaign-monitoring.md) for its fail-closed semantics.
`make raw-payload-check` separately verifies that every zero-length raw artifact belongs
to an explicitly declared, excluded, and replaced infrastructure-loss run.

The manuscript methods shell is [manuscript/main.md](manuscript/main.md). It contains
explicit pending markers instead of invented findings. `make completion-audit` is the
fail-closed full-study check and will pass only after human review, pilot, model freeze,
protected/unseen-family evaluation, paired recovery, final cards and independent
reproduction evidence all exist. The audit also reruns all three dataset hash validators,
the raw-payload audit, the recovery-guard verifier, the Research 1 boundary and the
tamper-evident research-log check; artifact presence alone cannot satisfy completion.

## Results and disclosures

- Final hypothesis table: `reports/confirmatory/final/hypotheses.md` (H1-H6 against the preregistered criteria).
- Deviations, amendments, incidents and disclosures: `docs/deviations-and-disclosures.md`.
- Independent clean-room reproduction record: `reports/reproduction/independent_rerun.yaml`; release record: `reports/reproduction/release.yaml`.
