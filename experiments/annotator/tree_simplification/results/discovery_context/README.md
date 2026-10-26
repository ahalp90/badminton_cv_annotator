# Selected discovery model

The default annotator uses the **no-max raw-baseline bundle, seed offset 0**.
Training and discovery retain outward context without a fixed time limit.
The validation comparison uses all 668 rallies from eight videos, five seeds
and ±10-frame contact tolerance. Discovery assessment uses the first and last
accepted contacts, with no maximum outward context.

| Training and discovery output | Fully correct / 668 | Share | Contact F1 | Correct-player F1 | Fully correct / 354 unflagged |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5/5 training; existing 2/3 output cap | 296.8 | 44.43% | 89.881% | 85.320% | 223.6 (63.16%) |
| No maximum context in training or output | 294.0 | 44.01% | 90.136% | 85.336% | 222.6 (62.88%) |

Counts and percentages are five-seed means. The baseline keeps the simpler
uncapped rule: the 2.8-rally validation difference is small and does not
establish statistical superiority.
The unflagged view excludes 314 marked rallies after judgement. All labels
remain in contact matching and neighbouring-rally checks; contact F1 measures
use every contact. Test results did not select the model.

Training rejects merged or incomplete rallies and contact/player errors;
extra outward context alone is allowed. Discovery retains the selected windows,
with small contact-margin extensions subject to neighbour and ownership checks.
Supplied rally boundaries stay fixed. Boundary-only confidence assessment uses
2 seconds before and 3 seconds after the labelled contacts, plus the inward
timing allowance. Dataset export separately adds two seconds before the first
accepted contact and three after the last, allowing overlapping clips.

- [Validation comparison](selection.json.gz) contains every seed and the common rule.
- [No-max bundles](no_max/models/) contain five raw-baseline and five body-window-36
  models. Each bundle holds the contact, selector and review trees.
- [Fit settings](no_max/fit-settings.json.gz) record the saved selector settings,
  reused contact fits and fixed training/validation videos.
- [40-video contact comparison](no_max/models_40/) reuses the fitted 40-video contact
  trees with the selected 32-video selector and review models.
- The [benchmark report](../../../monotrack_benchmark/annotator_monotrack_own_weights_compared_writeup_v7.md)
  gives test results and finite-boundary confidence surveys.

Contact fitting, preprocessing and the five-frame-preferred selector targets
are unchanged. Tree settings remain 180/400/100 maximum rounds and 31/31/7
maximum leaves for contact/selector/review, respectively. Learning rates are
0.06/0.05/0.05, minimum child weight 5 and L2 regularisation 1. The base contact
seed is 20260824; selector and review use 20260905. Benchmark offsets are
0, 100, 200, 300 and 400. These are variations in fitting, not independent
samples of videos.

A fresh hyperparameter search is a low-priority follow-up. Another selector
layer remains a separate experiment.
