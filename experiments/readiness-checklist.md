# Gate G0–G2 readiness checklist

Date opened: 24 August 2026

## G0 — shared platform

- [x] Host has ROS 2 Jazzy.
- [x] Host has Gazebo tooling.
- [x] Research 1 G6 platform is content-locked at commit `dac1d21d57a6c11077608f03b12c225b217efc0d`.
- [ ] Frozen launch command completes a clean route twice with the same seed.
- [x] Episode summary captures G6 platform commit, repository HEAD and simulator provenance.
- [x] rosbag2 records the Research 2 allowlist in MCAP and retains successful bags.
- [x] Independent topic counts and maximum gaps are checkpointed for MCAP reconciliation.

## G1 — labels

- [ ] Event thresholds have numerical values, units, and provenance.
- [ ] Event precedence is supervisor-reviewed.
- [ ] Ten initial representative bags are available.
- [ ] Twenty episodes have independent hand annotations.
- [ ] Automatic first-event labels match all twenty annotations.
- [ ] Positive, negative, too-late, and post-event window fixtures pass exactly.

## G2 — leakage and independence

- [x] Injector events and ground-truth streams are separated under label-only controls.
- [ ] Every feature timestamp is no later than its decision time.
- [ ] Missing samples produce a mask and age channel.
- [ ] No future interpolation or whole-episode normalization is used.
- [ ] Map and route assignments are frozen before window extraction.
- [ ] No run, map-route pair, or derived window crosses a forbidden split.
- [ ] Normalization statistics use development episodes only.

## Blocker log

| Date | Blocker | Consequence | Resolution owner |
|---|---|---|---|
| 2026-08-24 | Research 1 code, manifests, and bags were absent | Resolved by read-only shared-platform integration on 25 August | Researcher |
| 2026-08-28 | Research 1 protected confirmatory campaign is active | Do not run Research 2 Gazebo campaigns concurrently; static preparation may continue | Researcher |
| 2026-08-30 | Research 1 campaign dependency cleared: G7 passed and service inactive | Research 2 development-only live integrity campaign may execute | Resolved |
| 2026-08-30 | `live_integrity_v1` exposed Nav2 clock mismatch and recorder-health interval mismatch | Quarantine all v1 artifacts; repair and pass a new clean smoke before issuing v2 | Codex |
| 2026-08-28 | Seven terminal-event values require supervisor approval | Dataset extraction remains fail-closed | Supervisor |
