# Selected discovery model

The default annotator uses the **5/5-trained raw-baseline bundle, seed offset 0**.
It narrowly beat training without maximum context limits on fully correct
validation rallies. Selection used all 668 rallies from the eight validation
videos, five seeds and ±10-frame contact tolerance, with no maximum context
limit in the common scoring rule. Discovery boundaries use the first and last
accepted contacts.

| Training and discovery output | Fully correct / 668 | Share | Contact F1 | Correct-player F1 | Fully correct / 354 unflagged |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5/5 training; existing 2/3 output cap | 296.8 | 44.43% | 89.881% | 85.320% | 223.6 (63.16%) |
| No maximum context in training or output | 294.0 | 44.01% | 90.136% | 85.336% | 222.6 (62.88%) |

Counts and percentages are five-seed means. The primary measure was fully
correct coverage; ties would use correct-player F1, then contact F1. The
2.8-rally difference is small and does not establish statistical superiority.
The unflagged view excludes 314 marked rallies after judgement. All labels
remain in contact matching and neighbouring-rally checks; contact F1 measures
use every contact. Test results did not select the model.

The selected training rule allows up to five seconds before the first labelled
contact and five after the last. Its discovery finishing rule shortens long
windows to at most two seconds before the first selected contact and three
after the last, then applies the existing small edge padding and ownership
checks. Shorter windows stay shorter. Supplied rally boundaries stay fixed.
Dataset export separately anchors clips to the first and last accepted contacts
and adds two seconds before and three after, allowing overlapping clips.

- [Selection results](selection.json.gz) contain every seed and the common rule.
- [5/5 bundles](5_5/models/) contain five raw-baseline and five body-window-36
  models. Each bundle holds the contact, selector and review trees.
- [Fit settings](5_5/fit-settings.json.gz) record the saved selector settings,
  reused contact fits and fixed training/validation videos.
- [40-video contact comparison](5_5/models_40/) reuses the fitted 40-video contact
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
layer remains a separate experiment. The no-max arm was fitted and exported
at revision `65311a3e`; that checkout retains its uncapped finishing rule.
