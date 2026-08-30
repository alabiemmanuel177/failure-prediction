# Logging, labeling, and failure-taxonomy manual

## Separation of concepts

A **fault family** is an intervention mechanism, such as LiDAR dropout. A **terminal outcome** is what ended navigation, such as collision or timeout. They must never share one label column: a LiDAR dropout may cause no failure, immobilisation, abort, or collision.

Each episode therefore records:

- planned and actual injection information in `label_only`;
- zero or more observed events with timestamps and evidence;
- the first terminal event used for the primary forecasting label;
- nonterminal observations and recovery outcomes separately;
- infrastructure exclusions without deleting the raw record.

## Annotation workflow

1. Copy `data/annotations/episode.template.yaml`; never edit the template for an episode.
2. Enter times in simulation seconds using the same clock as the bag.
3. Record every observable event, even though only the first terminal event is primary.
4. Cite confirmation topics and add a short evidence note.
5. Use `confirmed` only when the operational rule is met; use `ambiguous` and retain the episode otherwise.
6. Have a second reviewer adjudicate the initial 20 episodes.
7. Generate labels to a new path; the command refuses to overwrite an existing label file.

Research 2 bags can be converted into a review-pending annotation without copying
timestamps manually:

```bash
source scripts/env_research2.sh
python3 scripts/extract_episode_annotation.py \
  data/raw/summaries/<run_id>.yaml \
  data/annotations/<run_id>.yaml
python3 scripts/validate_annotation.py data/annotations/<run_id>.yaml
```

Automatic extraction is not adjudication. A person must inspect the timeline and set
`review.adjudication_status` before the episode counts toward Gate G1.

## Window semantics

For default `W=5`, `H=10`, `delta=1`, and `G=20`:

- a decision at `t` consumes only `[t-5, t]`;
- it is positive when `t` is in `[t_f-10, t_f-1]`;
- it is too late when `t` is in `(t_f-1, t_f]`;
- it is post-event when `t > t_f`;
- otherwise it is negative only if at least 20 seconds from every event and eligible injection onset;
- all remaining decisions are excluded from primary training/evaluation.

Positive assignment takes precedence over the injection guard. Otherwise a fault onset inside the true precursor interval would erase the very positive windows the study is designed to learn. Injection metadata itself remains forbidden input.

## Logging layers

| Layer | Content | Mutability |
|---|---|---|
| Raw MCAP | Allowlisted ROS messages and clock | Immutable after finalization |
| Episode manifest | Provenance, injection, system variant, outcome, checksums | Finalized once; amendments are new records |
| Annotation | Human event times, evidence, review status | Versioned with adjudication history |
| Label table | One row per decision time and causal eligibility | Regenerated, input-checksummed derived data |
| Research log | Decisions, deviations, exclusions, issues, reviews | Append-only hash chain |

## Pre-model acceptance condition

Model development may begin only when:

- exact simulator topics and numerical event thresholds replace all Research 1 placeholders;
- 20 hand-audited episode annotations exactly match automatic labels;
- raw-bag checksums and message counts are present;
- split and leakage tests pass;
- at least one clean, one positive, one too-late, one guard-excluded, and one post-event window fixture is verified manually.
