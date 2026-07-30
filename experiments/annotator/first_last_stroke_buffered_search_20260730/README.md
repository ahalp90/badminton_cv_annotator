# First/last stroke buffered search

## Bottom line

The 90-frame first/last-stroke buffer was useful as a diagnostic, but it
should not become a matching rule. It revealed 19 additional correct
candidate-target associations for each court mode. The same buffers also
contained far more candidates that were wrong for the target.

The evidence supports removing this diagnostic from routine end-to-end runs
once the result and its reproduction path have been retained.

## What was checked

The search asked whether a contact candidate existed near the first or last
labelled stroke of each ground-truth rally. Its window could extend by up to
90 base-30 frames, subject to the neighbouring-rally and video bounds.

The post-hoc analyser compared the selected candidate with every labelled
stroke using the existing strict timing tolerance. It also checked whether a
correct candidate fell outside the predicted rally span. The analysis used
the three normal stride-8 ShuttleSet cases from the fixed CUDA measurement at
commit `189c5af`.

## Result

| Selected candidate | Static court | Live court detection |
|---|---:|---:|
| Correct first or last stroke | 317/564 | 313/564 |
| Different labelled stroke | 51/564 | 50/564 |
| No labelled stroke within strict tolerance | 178/564 | 173/564 |
| No candidate | 18/564 | 28/564 |

Among buffers containing a candidate, 229 static and 223 live-detection
selections were wrong for the target. The different-stroke rows are not
necessarily model false positives: they match a real labelled stroke, but not
the first or last stroke being assessed. The no-match rows are stronger false
positive candidates, subject to ground-truth omissions and the strict timing
tolerance.

Strict scoring did not credit 20 of the correct target associations for each
court mode. Nineteen belonged to split-rally cases, which strict scoring
deliberately excludes because there is no single associated predicted span.
The twentieth reused a candidate that strict scoring correctly assigned to a
closer neighbouring stroke. The buffered search therefore exposed 19
additional associations, not 20 new inference detections.

Every correct candidate in a covered rally was already inside its associated
predicted span. Extending an otherwise valid rally boundary would have
recovered none of them.

## Utility assessment

The diagnostic answered its question: ordinary boundary truncation is not
hiding useful first or last stroke candidates in covered rallies. Accepting
any candidate in the wider buffer would introduce substantially more
ambiguity than useful recovery.

Keep the analyser and this evidence pack so the conclusion remains
reproducible. Do not use the buffered search as a contact matcher.

TODO: remove `wide_edge_contact_rows` and routine `wide_edge_contacts.csv`
generation after updating their direct tests and any report or schema
consumers.

## Evidence

- `first_last_stroke_buffered_search_20260730-173434.csv` contains one
  post-hoc classification row per assessed first/last-stroke buffer.
- `scripts/analyse_first_last_stroke_buffered_search.py` regenerates a
  timestamped CSV from a completed measurement directory without rerunning
  inference.
