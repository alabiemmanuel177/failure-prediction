# Early Failure Prediction and Recovery for Mobile Robot Navigation

Research 2 is a causal early-warning and guarded-recovery study built as an overlay on
the pinned Research 1 platform in `/home/eao/risk-calibrated-nav`.

## Current state

The structured literature review, failure taxonomy, event contract, causal label
generator, causal scalar-feature extractor, transparent threshold baseline, event-level
evaluator, dependency-free SVG trace generator, replay-only recovery guards, Research 1
integration, retained MCAP recording, health reconciliation, and seven candidate fault
mechanisms are implemented. Clean and LiDAR-dropout development-map smokes pass. No
model training or protected-map evaluation is authorized yet.

Research 1 completed G7 on 30 August 2026. Its host-resource dependency is cleared and
the locked shared-platform boundary still passes. Research 2 may now acquire its
development-only fault-integrity and manual-audit episodes; causal-label certification
still awaits explicit review of seven operational event thresholds.

See [integration/STATUS.md](integration/STATUS.md) for the current gate status and
[docs/logging-labeling-taxonomy.md](docs/logging-labeling-taxonomy.md) for the annotation
workflow.

## Verify

```bash
source scripts/env_research2.sh
cd ros_ws && colcon build --symlink-install --packages-select failure_experiment && cd ..
python3 -m pytest -q
python3 scripts/check_readiness.py --stage extraction
```

The extraction readiness command is expected to fail until candidate fault settings and
seven supervisor-reviewed operational threshold values are frozen. Development and
validation splits are already assigned; protected routes remain deliberately unassigned.
