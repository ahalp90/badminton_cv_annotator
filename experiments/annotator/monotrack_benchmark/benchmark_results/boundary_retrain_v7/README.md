# Boundary-policy benchmark results

The [v7 report](../../annotator_monotrack_own_weights_compared_writeup_v7.md) uses
training policy **2 seconds before / 3 seconds after**, selected by mean fully
correct validation-rally coverage under the common 2/3-second evaluation rule.
Test outcomes did not choose the training policy or confidence thresholds.

## Training and evaluation boundary allowances

Each cell gives **validation / test** coverage of fully correct discovered
rallies, averaged over five seeds, without confidence filtering. Allowances are
seconds before the first labelled contact / after the last. Each row uses
separately retrained selector and confidence models, sharing the same contact
fits. Each column applies one evaluation rule to the saved predictions.
Discovery output context stays capped at 2 seconds before / 3 seconds after
selected contacts in every cell; the table varies training labels and scoring
allowances.

| Training allowance | Evaluate at 2/3 s | Evaluate at 3/3 s | Evaluate at 5/5 s |
| --- | ---: | ---: | ---: |
| **2/3 s** | **32.46% / 42.83%** | 39.94% / 48.70% | 44.10% / 54.37% |
| 3/3 s | 31.23% / 42.27% | 40.00% / 48.94% | 43.95% / 54.52% |
| 5/5 s | 31.41% / 42.35% | 40.30% / 49.13% | 44.43% / 54.90% |

Compare down a column to assess training choices. Under the strict 2/3 rule,
wider training allowances did not improve coverage; 2/3 was selected on
validation. Training at 5/5 scores slightly higher under the looser rules, but
the seed ranges overlap. Moving across a row relaxes the scoring rule without
changing predictions, so those larger gains are not improvements in the model.
The [CSV](policy_comparison.csv.gz) includes test seed ranges and boundary-only
scores. Validation and test have different flaw-label exclusions; see
[Appendix G](../../annotator_monotrack_own_weights_compared_writeup_v7.md#appendix-g-rallies-with-flawed-labels).

## Saved results

- `primary/`: headline 2/3 training and 2/3 evaluation. Includes five-seed
  summaries, paired video-bootstrap intervals, per-video counts, found-rally
  judgements and full validation/test confidence curves.
- `policy_selection.json.gz`: selection rule and all five validation counts
  for each training policy.
- `validation_comparison.json.gz`: all nine training/evaluation combinations
  on validation, including full unique-score threshold searches.
- `comparison/`: test summaries and validation-selected operating points for
  all nine training/evaluation combinations.
- [`policy_comparison.csv.gz`](policy_comparison.csv.gz): validation and test
  coverage for all nine combinations. Compare training choices under the same
  evaluation rule.
- `training_targets/`: differences in the actual selector targets consumed by
  the three fits, after confirming proposal rows and features align.
- `exclusion_effect/`: sensitivity to the treatment of the 524 excluded test
  rallies. These checks do not change the headline exclusion rule.

The three primary confidence thresholds are approximately 0.279 (closest to
half retained), 0.249 (best retrieval F1) and 0.786 (99% mean boundary target).
Use the exact values in `primary/confidence_cutoffs.json.gz` when scoring.
The boundary target retains only 7–12 validation rallies per seed.

The [model record](../../../tree_simplification/results/boundary_retrain_v7/README.md)
contains the ten selected 32-video bundles, fit settings and five additional
40-video contact-model comparison bundles. The latter retain the selected
32-video selector and confidence models. A new hyperparameter search remains
a TODO.

Reproduce inference and scoring with the [write-up scripts](../../writeup_scripts/README.md).
Use a fresh `NEW_RUN` directory for generated predictions. The policy records'
`NEW_RUN/annotator_runs/<policy>` paths refer to that generated data directory.
Large per-video prediction files are not duplicated in this compact results
folder. MonoTrack, court inputs and heuristic-only predictions retain the
existing benchmark locations.

Tables and JSON here are gzip-compressed. To redraw from this folder:

```bash
mkdir -p FIGURE_DATA
cp primary/*.gz FIGURE_DATA/
gzip -dk FIGURE_DATA/contact_summary.json.gz FIGURE_DATA/per_video_*.csv.gz
python ../../writeup_scripts/make_figures.py --data FIGURE_DATA --out FIGURES
```

Figures use the exact stored thresholds; displayed thresholds are rounded.
