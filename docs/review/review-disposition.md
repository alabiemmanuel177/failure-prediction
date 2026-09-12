# Review disposition

Recorded 12 September 2026.

## Independent second review: not obtained

A reviewer packet was prepared (`docs/review/reviewer-packet.md`) and the evidence
review application was made available. **No independent second reviewer completed a
review.** The study is released without one, deliberately and on the record, rather than
waiting indefinitely or substituting something that is not an independent review.

Two consequences follow, both already disclosed in the manuscript rather than introduced
here:

1. **No inter-rater reliability is claimed or estimable.** The twenty-episode label
   audit carries a completed review by one human reviewer. AI comparison was retained as
   supporting evidence only and was never counted as an independent human rating. The
   limitation is stated in the manuscript's limitations section and in the dataset card.
2. **No external technical review informs the reported conclusions.** The hypothesis
   outcomes, intervals and disclosures stand on the frozen protocol, the completion
   audit and the independent clean-worktree rerun — all machine-checkable — rather than
   on a reviewer's judgement.

## What was verified instead

- All six hypotheses evaluated against a threshold (`0.235`) and checkpoint frozen
  before the protected split was touched.
- Completion audit passes.
- Independent clean-worktree rerun regenerated predictions from the frozen checkpoint
  and reproduced every released table byte-identically.
- Three failed rerun attempts retained alongside the successful one.
- Core evidence deposited at DOI 10.5281/zenodo.22723258.

## If a reviewer is found later

The packet remains valid and points at the same frozen artefacts. Any review received
after this date will be recorded here, with its disposition, and will not be
retrospectively presented as having informed the release.
