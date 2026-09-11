# Gate G0–G2 readiness checklist

Date opened: 24 August 2026

## G0 — shared platform

- [x] Host has ROS 2 Jazzy.
- [x] Host has Gazebo tooling.
- [x] Research 1 G6 platform is content-locked at commit `dac1d21d57a6c11077608f03b12c225b217efc0d`.
- [x] Frozen launch command completes retained clean controls and faulted routes.
- [x] Episode summary captures G6 platform commit, repository HEAD and simulator provenance.
- [x] rosbag2 records the Research 2 allowlist in MCAP and retains successful bags.
- [x] Independent topic counts and maximum gaps are checkpointed for MCAP reconciliation.

## G1 — labels

- [x] Event thresholds have numerical values, units, and provenance.
- [x] Event precedence and numerical values have prospective researcher approval.
- [x] More than ten representative retained bags are available.
- [x] Twenty episodes have first-reviewer hand annotations with 20/20 agreement.
- [x] Protocol Amendment PA-2026-09-03-01 accepts the primary researcher review for model admission.
- [x] Automatic first-event labels pass exact primary-review admission on all twenty.
- [ ] Optional independent pre-publication review is obtained; no inter-rater claim is made meanwhile.
- [x] Positive, negative, too-late, and post-event window fixtures pass exactly.

## G2 — leakage and independence

- [x] Injector events and ground-truth streams are separated under label-only controls.
- [x] Every feature timestamp is no later than its decision time in automated fixtures.
- [x] Missing samples produce a mask and age channel.
- [x] No future interpolation or whole-episode normalization is used.
- [x] Development and validation map/route assignments are frozen before extraction.
- [x] Split tooling rejects run, map-route, and derived-window overlap.
- [x] Normalization code fits development episodes only.

## Blocker log

| Date | Blocker | Consequence | Resolution owner |
|---|---|---|---|
| 2026-08-24 | Research 1 code, manifests, and bags were absent | Resolved by read-only shared-platform integration on 25 August | Researcher |
| 2026-08-28 | Research 1 protected confirmatory campaign is active | Do not run Research 2 Gazebo campaigns concurrently; static preparation may continue | Researcher |
| 2026-08-30 | Research 1 campaign dependency cleared: G7 passed and service inactive | Research 2 development-only live integrity campaign may execute | Resolved |
| 2026-08-30 | `live_integrity_v1` exposed Nav2 clock mismatch and recorder-health interval mismatch | Quarantine all v1 artifacts; repair and pass a new clean smoke before issuing v2 | Codex |
| 2026-08-28 | Seven terminal-event values require supervisor approval | Dataset extraction remains fail-closed | Supervisor |
| 2026-08-31 | Shared ROS domain admitted a foreign `/clock` during supplements | Isolated Research 2 on domain 52; retained and excluded all pre-injection-invalid attempts | Resolved |
| 2026-08-31 | Twenty machine-extracted annotations required human review | Resolved by exact primary-review agreement under PA-2026-09-03-01 | Resolved |
| 2026-08-31 | First reviewer completed 20/20 with no disagreements | First-review evidence is retained; this does not substitute for a distinct second reviewer | Resolved |
| 2026-09-01 | Balanced development pilot reached 648/648 usable episodes | Development raw collection and hash-addressed inventory are complete | Resolved |
| 2026-09-01 | Validation-only 324-episode raw campaign started | Raw collection may continue; fitting, calibration and threshold selection remain fail-closed | Codex |
| 2026-09-03 | No independent reviewer or supervisor was available | Researcher approved PA-2026-09-03-01; single-review evidence admits model development, independent review remains optional, and no inter-rater claim is permitted | Resolved |
| 2026-09-03 | `goal_distance` compared odom-frame coordinates with the map-frame goal | Corrected to `/amcl_pose` before any model fit; derived artifacts regenerated, superseded copies retained | Resolved |
| 2026-09-03 | Transparent rules cannot detect more than 1/62 validation events under the 0.10 clean-mission budget (preliminary) | Retained as the honest P1 baseline; learned predictors are compared at the same budget | Researcher |
| 2026-09-03 | Research 1 retained bags are outcome-selected (646/647 admitted are terminal events) | Kept as a separate development dataset id; prevalence reported separately; contribution tested on Research 2-only validation before the freeze | Researcher |
| 2026-09-03 | Research 1 held-out maps have six routes, protocol assumes eight | Resolved by PA-2026-09-03-02: six routes per map, seeds raised to keep 1,008 / 1,512 / 1,008 minimums | Resolved |
| 2026-09-03 | User processes killed and host rebooted mid-campaign | Ledger clean at 966/1,212; runner, watcher and monitor resumed detached from login sessions; recorded in the research log | Resolved |
| 2026-09-03 | Outcome-selected Research 1 episodes reduce primary-model validation recall (0.290 to 0.161, single seed) | Resolved by PA-2026-09-03-03: excluded from the fitting pool (2,364 Research 2 episodes), retained as an audit set | Resolved |
