# Shared-platform integration status

Last verified: 30 August 2026

## Boundary

Research 2 reuses the Research 1 G6 scientific platform at the content boundary recorded
in `integration/research1.lock.yaml`. Descendant operations-only commits are accepted only
when every locked Git object remains exact. It writes no files to Research 1.
Development and validation maps are permitted. The Research 1 `test` split is rejected
by the Research 2 runner before simulator launch.

## Verified

- Research 1 and Research 2 overlays source together through `scripts/env_research2.sh`.
- The `failure_experiment` ROS 2 package builds with `colcon`.
- Raw camera, scan, and odometry streams are routed through `/research2/raw/*` and
  restored onto the standard topics consumed by Research 1/Nav2.
- Clean proxy smoke on `dev_00` sustained approximately 9–10 Hz camera/scan and 27.7 Hz odometry.
- A complete clean `S0/dev_00_r0` episode succeeded with a retained MCAP.
- A LiDAR-dropout smoke produced a clean prefix, eligible onset, exact 10% invalid-beam
  treatment, injection end, and terminal event. Raw LiDAR remained unchanged.
- The artifact validator passed the clean and LiDAR smokes.
- Successful and failed bags follow the same retention path; no success deletion exists
  in the Research 2 recorder.
- Topic-health counts are checkpointed every second and finalized by the episode runner.
- Research 1 G0-G7 passed, including the complete 3,648-episode pilot, 6,840-episode
  confirmatory archive and independent release reproduction. Its campaign service is inactive.
- The G6 content boundary passes at Research 1 release head
  `a6d5bf49f549c88317a4db5b6488ce451aeae88a`; post-G6 changes do
  not alter shared simulator, navigation, logging, perception, map or route objects.
- Development and validation map/route identifiers are assigned before extraction;
  protected Research 2 routes remain deliberately unassigned.
- A 25-attempt development-only campaign is frozen for four retained controls and all
  seven fault families at three severities. Its runner refuses execution while the
  Research 1 protected campaign is active.
- The offline pipeline now covers MCAP-to-scalar adapters, receipt-time causal
  resampling, explicit age/missingness channels, trailing-window features, development-
  only normalization, leakage denial, transparent rules, M-of-N/cooldown alarms,
  event-level metrics, dependency-free SVG traces and replay-only recovery guards.
- A synthetic engineering smoke regenerates an end-to-end metric artifact and is marked
  structurally as non-research evidence.

Smoke outputs live under `/tmp` and are not research data.

The first `live_integrity_v1` execution on 30 August is quarantined as engineering
evidence: one pre-simulation import failure and 24 artifact-invalid false arrivals
revealed missing Nav2 wall-to-simulation-time activation and misaligned recorder-health
start intervals. None of its episodes may enter training, validation or the manual-label
agreement set.

## Fault qualification matrix

| Family | Deterministic implementation | Unit tested | Live integrity smoke | Frozen for dataset |
|---|---:|---:|---:|---:|
| Camera occlusion | Yes | Yes | Pending S3 | No |
| LiDAR dropout | Yes | Yes | Passed S0 | No |
| Wheel slip / odometry bias | Yes | Yes | Pending | No |
| Localisation perturbation | Yes | Boundary tested | Pending | No |
| Dynamic blockage | Yes, delayed entity spawn | Config tested | Pending | No |
| Planner oscillation | Yes, delayed symmetric entities | Config tested | Pending | No |
| Semantic corruption | Yes, dual risk-grid proxy | Yes | Pending S3 | No |

Candidate severities are deliberately not frozen. Each family needs one witnessed live
integrity trace at all three severities, followed by a signed configuration decision.

## Remaining before dataset extraction

1. Execute the prepared seven-family live integrity matrix. The Research 1 resource
   exclusion cleared on 30 August 2026.
2. Supervisor-review the seven proposed numerical terminal-event values in
   `configs/event_threshold_proposals.yaml`.
3. Freeze fault parameters or revise Protocol 1.1 transparently.
4. Run and independently annotate 20 development/validation episodes.
5. Demonstrate exact automatic-versus-human event/window agreement.
6. Preserve the frozen development and validation route assignments during extraction.

Model development and protected-map execution remain blocked until these gates pass.
