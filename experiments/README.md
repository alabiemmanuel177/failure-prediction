# Experiment preparation

This directory turns Protocol 1.0 into executable controls. Research 1 completed G7 on
30 August 2026; Research 2 results still require their own gates and must not inherit
Research 1's scientific conclusions.

## Current readiness

| Item | State | Evidence/action |
|---|---|---|
| Ubuntu 24.04 / ROS 2 Jazzy | Available | `/opt/ros/jazzy` exists in the current environment |
| Gazebo tooling | Available | Jazzy Gazebo vendor binary is installed |
| Research 1 simulator | Available | G0-G7 complete; G6 content boundary passes against release HEAD |
| Navigation variants | Available | S0-S3 and frozen Nav2 configuration are reused through the overlay |
| Episode bags | Partial | Research 1 failures exist; exact Research 2 controls/fault bags await the live campaign |
| Split manifest | Development-ready | Development and validation assigned; protected routes intentionally unassigned |
| Twenty hand annotations | Ready to acquire | Prepared 25-attempt campaign may run; Research 1 service is inactive |
| Event and leakage controls | Drafted | See `configs/` |

## Shared-platform handoff status

The following handoff items are now resolved through `integration/research1.lock.yaml`:

1. ROS 2 workspace, shared launch interface and G6 content hashes;
2. development and validation map/route identifiers;
3. system variants, Nav2 parameters and semantic-health topics;
4. episode summary and host provenance interfaces;
5. topic names and map-frame convention.

Research 2 still supplies its own delayed fault injection, exact event publishing,
successful-bag retention, recorder reconciliation and causal annotations.

Do not put protected-test outcomes in development notebooks or tune thresholds from them.

## First 72-hour execution sequence

### Day 1 — contract freeze

- Review every `TODO_RESEARCH1` value in `configs/failure_events.yaml`.
- Map actual topics to `configs/feature_schema.yaml`.
- Extend `configs/leakage_denylist.yaml` from the simulator/fault-injector source.
- Assign maps and routes in `data/manifests/splits.template.yaml` before extracting windows.
- Have the researcher and supervisor sign off event precedence and the collision/localisation/immobilisation thresholds.

Exit criterion: all thresholds have units and provenance; no placeholder is silently accepted by code.

### Day 2 — ten-bag causal audit

- Check bag metadata, duration, topic counts, clock source, and lost-message events.
- Align streams by message timestamp without consuming a future sample.
- Hand-annotate first terminal event and injection onset in ten episodes.
- Generate a timeline displaying the 5 s history, 10 s horizon, and 1 s too-late guard.
- Compare automated and hand labels exactly.

Exit criterion: 20 audited episodes are ultimately required for Gate G1; the first ten establish the workflow.

### Day 3 — baseline and replay

- Implement four transparent rules: localisation covariance, progress, scan dropout, and oscillation.
- Tune only on development episodes.
- Evaluate first useful warning per event, false alerts per mission, and warning lead time.
- Trigger a guarded stop-and-replan in replay or simulation and preserve the log.

Exit criterion: one command regenerates the first metric table and annotated trace.

## Freeze sequence

1. Freeze event rules and precedence.
2. Freeze feature schema and leakage denylist.
3. Freeze episode/map/route splits.
4. Extract windows.
5. Fit models on development data.
6. Select calibration and alert policy on validation data.
7. Timestamp the model/policy bundle.
8. Run protected tests once.

Any change after step 7 creates a new protocol version; it does not overwrite the prior test.

## Offline pipeline order

For each retained episode after the live gate opens:

1. `extract_episode_annotation.py` creates a review-pending event annotation.
2. Human reviewers complete and adjudicate the annotation.
3. `generate_labels.py` writes causal decision labels.
4. `extract_bag_scalar_telemetry.py` writes observations using recorder receipt time as availability time.
5. `extract_scalar_features.py` performs past-only synchronization with value, age and missingness channels.
6. `derive_window_features.py` adds only trailing-history features.
7. `validate_dataset_splits.py` checks episode/map/route independence.
8. `evaluate_threshold_baseline.py` stays fail-closed until rule and alarm thresholds are frozen from development/validation only.
9. `plot_warning_trace.py` writes a dependency-free SVG timeline.

The synthetic smoke validates software plumbing only. Its metrics must never appear in a paper result table.

## Pre-model commands

```bash
# Verify protocol configuration and report unresolved gates
python3 scripts/check_readiness.py --stage extraction

# Verify the shared platform without requiring an exact operations HEAD
python3 scripts/check_research1_boundary.py

# Inspect the frozen 25-attempt campaign without launching ROS
python3 scripts/run_live_integrity_campaign.py --dry-run

# Regenerate a synthetic engineering-only offline metric artifact
python3 scripts/run_offline_pipeline_smoke.py

# Validate one hand annotation
python3 scripts/validate_annotation.py data/annotations/<run_id>.yaml

# Generate a new causal decision-label table (refuses overwrite)
python3 scripts/generate_labels.py \
  data/annotations/<run_id>.yaml \
  data/derived/labels/<run_id>.csv

# Verify the append-only research audit chain
python3 scripts/research_log.py verify

# Run contract and boundary tests
python3 -m pytest -q
```
