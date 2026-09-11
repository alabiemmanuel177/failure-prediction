# Model card: Research 2 early-warning predictor

Status: pre-training contract; no fitted model or performance claim exists yet.

## Intended use

Estimate the probability that the first terminal mobile-navigation event will occur
within ten seconds, using only telemetry available at the current decision time. The
score feeds a separately frozen alarm policy and an independently guarded recovery
system. It is not a safety certificate or a replacement for Nav2 collision checking.

## Required model record

- Predictor ID, architecture ID, checkpoint hash and source commit.
- Feature-schema, split-manifest and normalisation hashes.
- Training episodes, class/event weighting and random seeds.
- Window length, horizon, optimizer, loss, stopping rule and parameter count.
- Calibration method/parameters and validation-only threshold record.
- Median, p95 and maximum feature-plus-inference latency.

## Required evaluation

- Event recall at the frozen false-alert budget with complete denominators.
- False alerts per clean and all non-event missions; alert burden and lead time.
- AUPRC/AUROC as supporting window metrics only.
- Brier score, ECE and reliability curves before and after calibration.
- Held-out maps, seven unseen-family folds and mandatory feature/policy ablations.
- Paired recovery completion, collision, overhead, guard rejection and regret results.

## Prohibited evidence

No injected-fault commands, parameters, onset timestamps, simulator ground truth,
terminal action results, recovery results, future samples or protected-set tuning may
enter fitting, calibration or deployable features. Random window splitting and window
accuracy cannot support the primary claim.

## Limitations to update after evaluation

Populate missed-event and false-alarm failure modes, family/map heterogeneity,
unforecastable horizons, calibration drift, missing-data sensitivity, latency limits and
simulation-to-real constraints. Negative results remain part of the card.
