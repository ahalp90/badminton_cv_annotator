# Issue 90 held-out serve-rule evaluation

## Result

Do not port PR #88 into the annotator unchanged.

PR #88 tied PR #82 on aggregate server accuracy. It improved visible-start and
joint accuracy, but the result was not consistent across both held-out videos.
It regressed server and visible-start accuracy on `sset_22`. This fails the
issue's requirement to improve without regressions.

## Primary comparison

The primary population contains 85 held-out rallies with a one-to-one mapping
between one ShuttleSet rally and one detected rally span.

| Measure | PR #82 | PR #88 | Change |
| --- | ---: | ---: | ---: |
| Server side | 61/85 (71.8%) | 61/85 (71.8%) | 0 |
| Visible start | 31/85 (36.5%) | 41/85 (48.2%) | +10 |
| Both correct | 27/85 (31.8%) | 38/85 (44.7%) | +11 |

PR #88 changed 16 server answers. Eight were fixes and eight were damages. The
exact two-sided McNemar p-value for those paired server changes is 1.0.

## Per-video results

| Video | Population | Server PR82 -> PR88 | Start PR82 -> PR88 | Joint PR82 -> PR88 |
| --- | ---: | ---: | ---: | ---: |
| `sset_20` | 53 | 40 -> 43 | 19 -> 30 | 17 -> 28 |
| `sset_22` | 32 | 21 -> 18 | 12 -> 11 | 10 -> 10 |

`sset_20` had seven server fixes and four damages. `sset_22` had one fix and
four damages.

## End-to-end coverage

The two videos contain 144 ground-truth rallies and 153 detected spans. The
annotator fully covered 92 rallies. Seven of those were in spans that overlapped
another rally, leaving 85 strict one-to-one rallies for the primary comparison.
There were 33 split rallies, 19 missed rallies, and 32 spans with no
ground-truth stroke.

`sset_22` was the weaker segmentation case. Only 32 of its 70 ground-truth
rallies formed strict one-to-one spans, while 33 were split. This limits how
broadly the rule result can be read. It does not remove the measured regression
on the valid one-to-one population.

## Leakage boundary

Predictions for all 153 detected spans were written before ShuttleSet labels
were opened. The frozen prediction file has SHA-256:

`6b946a6422681936472019f32965a2a9ba32a4d88ef8a15dc0e015ec72873e09`

The scorer verified that checksum before loading labels. It then used the
predeclared 10-frame base-30fps temporal tolerance and a strict one-to-one rule
that also excludes partial rally overlaps. No threshold, rule, or prediction
was changed after labels were loaded.

## Decision

Issue #90 is complete as an evaluation. Its promotion gate did not pass. Keep
PR #88 as research evidence and leave production behavior unchanged.
