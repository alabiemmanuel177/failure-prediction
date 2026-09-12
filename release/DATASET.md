# Research 2 core evidence: early failure prediction and guarded recovery

Core evidence for a preregistered ROS 2 / Gazebo / Nav2 study of whether mobile
robot navigation failure can be predicted early enough to act on.

Archive: `research2-failure-prediction-core-v1.tar.zst` (2,225 files, 83.1 MB
uncompressed). Verify against `CORE_SHA256SUMS` before extracting.

## Headline result

Six hypotheses were evaluated against a decision threshold of `0.235` and a model
checkpoint frozen before the protected split was touched.

**Both confirmatory hypotheses were not supported.** The learned predictor could
not be shown to beat a transparent threshold baseline at the validation-fixed
false-alert budget (H1: 0.197, 95% CI [-0.000, 0.447]), and prediction-triggered
recovery did not improve mission completion (H6).

What held: median useful lead time of detected failures was 3.818 s, 95% CI
[1.745, 6.120], against a 3 s bar set in advance (H2); calibration reduced Brier
score on validation (H3). Generalisation did not: the predictor beat the baseline
in four of seven leave-one-family-out folds against a preregistered five (H5).

The full table, with the source file for every row, is at
`reports/confirmatory/final/hypotheses.md`.

## What is in this archive

| Path | Contents |
|---|---|
| `reports/` | Confirmatory results, calibration, thresholds, ablations, integrity audits, figures with provenance sidecars, recovery, pilot, manual-audit and reproduction records |
| `models/` | All trained checkpoints: the frozen final model, per-seed variants, ablations, leave-one-family-out models, and the recovery selector, each with its normalisation and training record |
| `data/manifests/` | Every immutable campaign manifest and dataset inventory |
| `configs/` | Frozen configurations, including the model freeze and calibration policy |
| `docs/` | Dataset card, model card, deviations and disclosures, reviewer packet, protocol |

## What is deliberately not in this archive

- **Per-window prediction tables** (~3 GB of CSV). Regenerable by running the
  included checkpoints over the derived sequences.
- **Derived model-input sequences** (~13 GB) and **raw recordings** (~175 GB).
  Registered by manifest and mirrored, but not deposited.

**Read this before planning work against the archive.** The checkpoints are
included so the models themselves can be inspected, compared and re-run. But
re-running inference end to end additionally requires the derived sequences,
which are not in this deposit. Without them, this archive supports full
verification of every reported number and inspection of every trained model; it
does not by itself support regenerating the predictions from scratch.

## Verifying

```bash
sha256sum -c CORE_SHA256SUMS
tar --use-compress-program=unzstd -xf research2-failure-prediction-core-v1.tar.zst
```

Every figure carries a `.json` provenance sidecar naming the checkpoint hash it
was produced from. Three failed reproduction attempts are retained in
`reports/reproduction/` alongside the successful one, on purpose.

## Scope and limits

Simulation only. This is not evidence of real-world safety, and no claim of
safety certification is made or implied. No inter-rater reliability is claimed
for the label audit: twenty episodes carry a completed single-reviewer review,
and a second reviewer must be a different person.
