# Follow-up 3: precision-first rally filtering

## Bottom line

The frozen automatic rule could not isolate a non-empty zero-error subset.

Only 1 of the current pipeline's 311 predicted rally spans was a complete
record at the project's ±5 base-30-frame tolerance. Even the strictest rule
kept six erroneous records alongside that one correct record.

In every leave-one-fixture-out split, each non-empty rule still made errors on
the two development fixtures. The held-out procedure therefore selected no
rule and retained 0 of 311 records. This is a negative result for this simple,
auditable route. It is not a claim that no future filter could work.

## What was tested

The input was the frozen Issue 103 output for `sset_01`, `sset_15` and
`sset_21`. These contain 109, 123 and 79 predicted rally spans respectively.
The three annotator-result SHA-256 hashes were checked against their fixed
values before use. The court, shuttle and exclusion sidecar hashes were
recorded in the truth-free feature table.

A predicted span first had to map to exactly one covered ground-truth rally.
It then counted as complete only when all five annotation fields were correct:

- exact contact count;
- every contact matched within the accepted timing tolerance;
- player attribution;
- server;
- point outcome.

Landing and hit height were not part of this first completeness test. The
project currently treats them as experimental outputs.

The rule had four increasingly strict levels:

1. The current pipeline had resolved the server, final actor and point
   outcome. It also had at least one accepted contact. Every accepted contact
   had explicit wrist support, a visible shuttle, no recurrence warning and no
   exclusion at that frame.
2. At least 80% of the rally span had a detected court and passed the existing
   court-view vote.
3. The shuttle tracker reported a visible shuttle on at least 80% of the span.
4. The point outcome came from the next server and independently agreed with
   the geometric outcome check.

This is a monotone ladder: each level adds conditions without removing earlier
ones. The 80% threshold was fixed before scoring. There was no classifier or
threshold sweep.

## Leakage boundary

The automatic feature table was built before the scorer opened any labels. It
contains no ground-truth fields and declares that fact in its schema. Its
SHA-256 is
`ffbb61f73434632d2c81ffb930fac7ce0b430b511488f4cc55998953e7e1499e`.

For each held-out fixture, the other two fixtures were development data. A
rule was eligible only when it retained at least one development record and
made no development errors. Among eligible rules, the scorer would choose the
one retaining the most records. A tie would go to the stricter rule.

No rule was eligible in any split. The scorer did not substitute a hand-picked
rule or inspect the held-out fixture to make one eligible.

## Result

The strictest rule gives the clearest view of why selection failed:

| Fixture | Kept by strictest rule | Complete records | Records with errors |
|---|---:|---:|---:|
| `sset_01` | 4 | 0 | 4 |
| `sset_15` | 1 | 1 | 0 |
| `sset_21` | 2 | 0 | 2 |
| **Total** | **7** | **1** | **6** |

The four `sset_01` records all had the wrong contact count. Three also missed
the timing requirement. Two had wrong player attribution and two had the wrong
server.

Of the two `sset_21` records, one was not a complete ground-truth rally span.
The other had incorrect timing, player attribution, server and point outcome.

The single correct `sset_15` record does not support a transferable rule. Any
development pair containing `sset_01` or `sset_21` still had errors at the same
strict level.

The exact development results were:

| Held out | Development fixtures | Strictest rule kept | Development errors |
|---|---|---:|---:|
| `sset_01` | `sset_15`, `sset_21` | 3 | 2 |
| `sset_15` | `sset_01`, `sset_21` | 6 | 6 |
| `sset_21` | `sset_01`, `sset_15` | 5 | 4 |

Because every split selected no rule, the final held-out result was 0 retained
and 311 rejected. Precision is undefined when nothing is retained; it is not
100%.

At ±10 and ±15 base-30 frames, the unfiltered pipeline had three complete
records instead of one. The rule-selection decision stayed fixed from the
primary ±5 result, so the sensitivity runs also retained nothing.

## Intern advisory signal

No new VLM inference ran for this follow-up.

The earlier Intern scene call was not added after the result was known. Its
retained row-level artefact combines predictions with human scene labels, and
the original parse-only attempts are not retained in this repository package.
It also describes whole scene targets rather than the contact and outcome
evidence tested here. Retrofitting that signal after seeing the held-out result
would break the frozen-rule boundary.

This limit narrows the conclusion: the deterministic ladder failed. The result
does not rule out a separately planned, truth-blind VLM advisory filter.

## Decision

Do not use this rule to claim a precision-first dataset subset.

The court, track and outcome checks measure whether supporting evidence exists.
On these fixtures, they did not predict whether the complete annotation record
was correct. Follow-up 4 can still test compact evidence as input to the serve
task; this result does not change the clean-interface choice of Intern from
Follow-up 2.

## Limits

This test covers three labelled fixture videos and the frozen Issue 103
pipeline output. It does not estimate performance across the full ShuttleSet
collection.

The primary ±5 base-30 tolerance becomes four native frames for the two 25 fps
fixtures and five frames for the 30 fps fixture. The ±10 and ±15 results are
sensitivity checks only.

The rule uses existing automatic signals. It does not repair missing or wrong
contacts, rally boundaries, player attribution, server calls or outcomes.

## Evidence

- [`../../experiments/precision_first_trials.py`](../../experiments/precision_first_trials.py)
  contains the truth-blind feature builder, frozen rule ladder, held-out
  selection and scorer.
- [`evidence/3_precision_first_features.json.gz`](evidence/3_precision_first_features.json.gz)
  contains all 311 automatic feature rows and the source hashes. Its compressed
  SHA-256 is
  `982a0a912a21a59612bbfb9d701613606be5c768f3ae7105e03d1e849fbdcb4a`.
- [`evidence/3_precision_first_inputs.tar.gz`](evidence/3_precision_first_inputs.tar.gz)
  contains the portable frozen annotation, court and shuttle inputs used to
  build those rows. Its SHA-256 is
  `a079f40e6fbf15963a0dec1e9a7d191396dd885962de44ff7dcb07d9bf723af1`.
- [`3_precision_first_dataset.json.gz`](3_precision_first_dataset.json.gz)
  contains every rule score, held-out fold, failed field and sensitivity
  result. Its compressed SHA-256 is
  `497ed821bc667026d9d1aadf77859f88050c552b41989008bc960061d56a549a`.

To reproduce the score from the repository root:

```bash
run_dir=$(mktemp -d) &&
tar -xzf scratch/vlm_pr80_eval/followups/results/evidence/3_precision_first_inputs.tar.gz \
  -C "$run_dir" &&
gzip -cd scratch/vlm_pr80_eval/followups/results/evidence/3_precision_first_features.json.gz \
  > "$run_dir/features.json" &&
PYTHONPATH=src:scratch/vlm_pr80_eval \
  python -m experiments.precision_first_trials score \
  --features "$run_dir/features.json" \
  --artifacts-root "$run_dir" \
  --output "$run_dir/score.json"
```
