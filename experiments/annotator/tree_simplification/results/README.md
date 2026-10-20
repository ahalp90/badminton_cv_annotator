# Development selection evidence

These compressed JSON files retain the numerical evidence used to choose the
smaller annotator. They contain validation results from eight videos, with 668
labelled rallies and 5,696 contacts per run. The five seed offsets reuse those
videos; they are repeated fits, not independent evaluation populations.

- `five-seed-evidence.json.gz`: matched old XGBoost and 7/15-leaf selectors,
  both depths, plus every outcome from the six-setting initial-seed screen.
- `retained-setting-evidence.json.gz`: both retained 31-leaf configurations
  (200 and 400 rounds), both depths and all five seeds. Counts are retained per
  video, with pooled metrics, paired differences and completion receipts.
- `previous-hgb-validation.json.gz`: the best earlier HGB grid's validation
  means, including its source branch, commit and result path. These aggregate
  means provide a descriptive comparison, rather than paired seed differences.
- `selected-production.json.gz` and `selected-review-rows.json.gz`: the selected
  seed-zero bundle's exact production replay and review-ranking comparison.
- `selected-settings.json.gz`: the export's source run, depth and exact training
  and pool settings, including the unchanged contact configuration.
- `selected-timing.json.gz`: both model orders on all eight validation videos,
  using two CPU threads and checking every output stream.
- `timing-script.json.gz`: the checksum of the timing script, which ran beside
  the experiment checkout before being added as `benchmark.py`.
- `selected-profile.json.gz`: a single-video profile locating the expanded
  proposal and feature construction cost. Profiling adds timing overhead.

The evidence records the original source paths on Carmack and in the local
collection. Large prepared inputs and fitted models remain outside Git. The
experiment checkout was `ddd2a7f2`; production export uses `e2b948d2`.
The selection retains the original seed-zero fit rather than choosing a seed
from its validation score.

Attributed F1 is twice the correctly attributed matched contacts divided by
predicted plus labelled contacts. Pool the counts across videos before computing
each seed's F1, then average those five scores. A paired difference always uses
the same seed, tolerance and validation videos. The retained comparison records
which changes also preserve the candidate pools and target policy.

All results are development selection evidence. The HGB-to-XGBoost and
old-to-reduced-stack comparisons change whole systems; they do not isolate
the effects of the backend, proposal coverage or removal of individual models.
