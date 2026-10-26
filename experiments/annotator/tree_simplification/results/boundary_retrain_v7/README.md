# Annotator boundary-policy refit

This earlier comparison selected the **2 seconds before / 3 seconds after**
training policy on the eight validation videos. The [current default](../discovery_context/README.md)
comes from the subsequent comparison without a maximum evaluation buffer.
All three trees were refitted
on the original 32 training videos with the existing hyperparameters and five
seed offsets. Hyperparameter retuning after these changes remains a TODO.

Selection maximised mean fully correct labelled-rally coverage before confidence
filtering, using the same 2/3-second evaluation rule for all three training
policies. Ties prefer 2/3, then 3/3, then 5/5. Test results did not choose the
training policy.

| Training allowance, before / after | Fully correct validation rallies, mean of five seeds | Share of 668 labelled rallies |
| --- | ---: | ---: |
| 2 / 3 seconds | 216.8 | 32.46% |
| 3 / 3 seconds | 208.6 | 31.23% |
| 5 / 5 seconds | 209.8 | 31.41% |

The allowances are maximum context, so closer valid boundaries can pass.
Discovery output retains at most two seconds before its first kept contact and
three seconds after its last, with the existing small padding and neighbour
checks. Supplied rally bounds stay fixed.

- [`selection.json.gz`](selection.json.gz) contains the selection rule and counts
  for every seed and training policy.
- [`validation-cutoffs.json.gz`](validation-cutoffs.json.gz) records the three
  confidence operating points selected from all 3,346 distinct validation scores.
  The 99% boundary target retains only 7–12 validation rallies per seed.
- [`2_3/models/`](2_3/models/) contains the selected policy's five raw-baseline
  bundles and five body-window-36 comparison bundles. The default is raw baseline,
  seed offset 0. Each bundle contains the contact, selector and confidence trees.
- [`2_3/fit-settings.json.gz`](2_3/fit-settings.json.gz) records the actual fit
  settings, training/validation videos, contact sampling orders and source revision.
  Each tree uses two CPU cores; exported models predict on the CPU.
- [`2_3/models_40/`](2_3/models_40/) contains the five contact-size comparison
  bundles. Their contact models use all 40 original videos, while the selector
  and confidence models retain their selected 32-video fits. Each bundle's
  `contact-fit.json.gz` records the fit settings and 32-video parity check.

The body-window-36 setting is a fixed comparison from the earlier validation
search. This refit does not repeat the hyperparameter or preprocessing search.

Use the [benchmark retraining and evaluation commands](../../../monotrack_benchmark/writeup_scripts/README.md)
to reproduce the three-policy comparison. The earlier bundles and experimental
results remain under the parent results directory for their historical reports.

## Contact training coverage

The [saved-input measurement](contact_training_coverage.json.gz) checks the
32 training videos for the selected raw-baseline contact model. Of 27,571 loaded
GT contacts, 26,893 (97.5%) have a candidate frame within the positive training
radius of one frame at 30 fps. The 678 others have no feature/search row within
that radius; no additional contacts are lost to the candidate-region filter.
The 80,651 positive frame rows exactly match the saved contact-fit count.

Recheck the saved prepared inputs without fitting a model, from the repo root:

```bash
PYTHONPATH=src:. python experiments/annotator/tree_simplification/analysis/contact_training_coverage.py \
  --training NEW_RUN/training --workers 8 | gzip > contact_training_coverage.json.gz
```

The [contact-training explanation](../../../../../docs/annotator/how_it_works.md#how-contact-training-uses-ground-truth)
describes the search regions, labels and negative sampling.
