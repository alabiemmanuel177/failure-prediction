# Reviewer packet — Research 2

Early Failure Prediction and Recovery for Mobile Robot Navigation.

Prepared 12 September 2026, against release commit `00d0653`. Everything below is
in this repository at that commit unless stated otherwise. If a claim in the
manuscript is not traceable to one of these files, treat that as a finding.

---

## What is being asked

An independent read of the evidence, not a proofread. The three questions that
matter most:

1. **Are the two confirmatory conclusions correct as stated?** Both are reported
   as not supported. The risk here is the opposite of the usual one — a null
   result stated more strongly than the interval justifies.
2. **Is anything exploratory being read as confirmatory?** Several model variants
   were run and one has an interval excluding zero. It is marked exploratory. Say
   so if the framing anywhere lets it drift.
3. **Do the deviations change any conclusion?** They are disclosed; the question
   is whether disclosure is sufficient or whether a conclusion should weaken.

## Start here

| Order | Artefact | What it settles |
|---|---|---|
| 1 | [`reports/confirmatory/final/hypotheses.md`](../../reports/confirmatory/final/hypotheses.md) | The final hypothesis table: all six, with estimates, intervals, status and the source file for each row |
| 2 | [`docs/deviations-and-disclosures.md`](../deviations-and-disclosures.md) | Every departure from the preregistered plan, and what was done about it |
| 3 | [`reports/reproduction/release.yaml`](../../reports/reproduction/release.yaml) | The release record: checksums over manifests, configs, checkpoints, reports and documentation |
| 4 | [`reports/reproduction/independent_rerun.yaml`](../../reports/reproduction/independent_rerun.yaml) | The clean-worktree rerun that reproduced the release |

## The frozen objects

Any result should resolve to these and nothing else.

- **Decision threshold:** `0.235`, fixed on validation before the protected split was touched.
- **Model checkpoint:** sha256 `b80b4e8ba92cef6bc6b9d55e0d9949622e244f2eb710b418cddfb645196117c4`.
- **Protected split assignment:** [`reports/confirmatory/protected_split_assignment.yaml`](../../reports/confirmatory/protected_split_assignment.yaml).

Both the release record and the rerun record state protected-split use explicitly
rather than leaving it implied. That is deliberate — please check it is accurate.

## Hypotheses, in one place

| | Hypothesis | Type | Outcome |
|---|---|---|---|
| H1 | Learned predictor beats the transparent baseline at the validation-fixed false-alert budget | confirmatory | not supported |
| H2 | Median useful lead time of detected failures >= 3 s | supporting | supported |
| H3 | Calibration reduces Brier score and ECE on validation | supporting | supported |
| H4 | Removing planner and localisation health features degrades early warning | supporting | reported; no numeric threshold was prespecified |
| H5 | Leave-one-family-out recall beats baseline in >= 5 of 7 families | supporting | not supported |
| H6 | Prediction-triggered recovery improves mission completion without more collisions | confirmatory | not supported |

H4 is the one to scrutinise for framing: it has no prespecified numeric bar, so
it cannot pass or fail, only be reported.

## Supporting context

- [`docs/dataset-card.md`](../dataset-card.md) — intended use, independent unit, splits, collection design, retention of invalid attempts
- [`docs/model-card.md`](../model-card.md) — the model and its limits
- [`reports/confirmatory/natural_failure_audit.yaml`](../../reports/confirmatory/natural_failure_audit.yaml) — performance on natural, non-injected failures
- [`reports/confirmatory/unseen_family.yaml`](../../reports/confirmatory/unseen_family.yaml) — the leave-one-family-out evidence behind H5
- Three failed reproduction attempts are retained beside the successful one, in `reports/reproduction/`. They are kept on purpose; a reproduction record showing only the attempt that worked is not one.

## Label audit

Twenty episodes carry automatic annotations, causal labels and a completed
researcher review under `data/annotations/reviewed/`. **No inter-rater
reliability is claimed** — only one reviewer has completed it. A second reviewer
must be a different person. The review application records attestations
append-safe; it never displays model output or protected-map evidence.

The app runs as a user service on port 3002, published over Tailscale Funnel.

## How to report

Findings by severity — anything that changes a conclusion, anything that changes
a number, anything else. Please state explicitly whether, in your view, the two
confirmatory nulls are correctly stated. A recommendation is welcome but not the
point; the disposition will record your comments individually either way.
