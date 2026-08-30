# Supervisor review: terminal-event thresholds

Protocol version: 1.0  
Prepared: 28 August 2026  
Decision status: pending

## Purpose

Seven numerical values remain deliberately outside `configs/failure_events.yaml`.
Approving them freezes what counts as localisation loss, immobilisation, and unsafe
perception before the first Research 2 dataset is extracted. Values must not be revised
after inspecting protected-map outcomes.

The machine-readable proposals are in
`configs/event_threshold_proposals.yaml`. Approval should record the reviewer, date,
decision for each event, and any replacement value with rationale.

## Proposed decisions

### Localisation loss

- Translation error: **0.50 m**.
- Yaw error: **0.50 rad**.
- Persistence: **2.0 s continuously**.
- Proposed event logic: translation **or** yaw exceeds its threshold continuously for
  the persistence duration.

The translation threshold is twice Research 1's frozen 0.25 m goal tolerance; the yaw
threshold equals its frozen 0.50 rad goal tolerance. Two-second persistence is proposed
to reject single AMCL resets and timestamp-alignment transients. Ground truth is
label-only and cannot enter predictor features.

Decision required: accept, revise prospectively, or reject localisation loss as a
separate terminal class.

### Immobilisation

- Minimum commanded motion: **0.05 m/s equivalent**.
- Maximum progress: **0.50 m**.
- Persistence: **10.0 s continuously**.

The progress and duration match Research 1's frozen Nav2 `SimpleProgressChecker`. The
command threshold matches the Research 2 fault-eligibility guard. The review must decide
whether angular-only commands count and whether progress is measured along the route or
as planar displacement.

Recommended interpretation: commanded linear speed at least 0.05 m/s and planar odometry
displacement below 0.50 m over the preceding 10 s.

### Unsafe perception

- Protected stopping region: **0.42 m beyond the footprint**.

Research 1 freezes forward speed at 0.35 m/s and collision-monitor lookahead at 1.2 s;
their product is 0.42 m. This definition additionally requires a label-only missed
critical obstacle and emergency intervention. The reviewer must decide whether to add a
braking or timestamp-latency margin.

## Approval form

Copy `configs/event_threshold_review.template.yaml` to a dated review file. Complete all
reviewer and decision fields without changing the proposal file. A revision is valid
only before dataset extraction and must state whether already-created development data
must be invalidated.

## Acceptance checks after approval

1. Transfer approved values to `configs/failure_events.yaml`.
2. Add exact boundary fixtures for just-below, equal, just-above, and interrupted persistence.
3. Run `python3 scripts/check_readiness.py --stage draft`.
4. Record the approval and configuration hash in the append-only research log.
5. Do not mark fault configurations frozen until their live integrity traces pass.

