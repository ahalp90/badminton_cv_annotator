# Discovery context comparison

The **5/5-trained configuration with its existing 2/3 discovery cap** wins the
pre-set validation comparison. It recovers 296.8 fully correct rallies per seed,
compared with 293.0 for training and discovery without maximum context limits.
This small difference selects the default; it does not establish statistical
superiority or separate the effects of training and finishing.

## Selection

Both configurations are judged at ±10 frames under the same rule without
maximum outward context. A correct rally must cover one complete labelled rally
without merging neighbours and contain every contact with the right player,
without extras. Contact F1 uses the whole stream, including outside-rally events.

| Configuration | Fully correct / 668 | Share | Contact F1 | Correct-player F1 | Fully correct / 354 unflagged |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5/5 training; 2/3 discovery cap | 296.8 | 44.43% | 89.881% | 85.320% | 223.6 (63.16%) |
| No maximum context in training or discovery | 293.0 | 43.86% | 90.136% | 85.336% | 221.6 (62.60%) |

Values are means across seeds 0, 100, 200, 300 and 400. Selection maximises fully
correct coverage over all 668 validation rallies; ties use correct-player F1,
then contact F1. The unflagged view removes 314 rallies after judgement, retaining
all labels for matching and neighbour checks and all contacts in F1.
[Per-seed results and rule](policy_selection.json.gz).

Preparation, contact fits and proposal features were reused. The uncapped arm
refitted selector targets, selectors and review models. It used the same
five-frame-preferred targets and contact-ownership checks. Test results did not
choose the configuration. The selected bundles and fit settings are in the
[model record](../../../tree_simplification/results/discovery_context/README.md).

## Selected-model test results

The 46 test videos contain 3,327 scored rallies and 37,184 labelled contacts.
At ±10 frames, the selected model scores 96.714% Contact F1, 94.702%
Correct-player F1 and 60.024% fully correct with supplied bounds. In discovery,
it scores 93.482%, 90.820% and 54.944%, respectively. Rally correctness uses
no maximum outward context limit. The [primary results](primary/) contain all
conditions, both timing tolerances, per-video counts and paired intervals.

## Finite-boundary confidence surveys

These surveys judge **discovery output before dataset export**. The 1/1 rule
allows at most one second before the first labelled contact and one after the
last; 2/3 allows two and three seconds. Both retain the ±10-frame inward
allowance and the same rally/contact/player checks. Dataset clips are separate:
accepted first/last contacts plus 2/3 seconds, clipped only at video edges.

Every unique validation score was searched: 3,339 for 5/5 and 3,333 for no-max.
The objectives are closest to half the pooled validation detections retained,
and highest mean retrieval F1 for fully correct rallies. The third objective,
99% mean boundary accuracy with detections retained in all five seeds, is
unattainable in all four surveys.

The table reports test results at the validation-selected thresholds. Counts
and shares are averaged separately. Boundary and fully-correct percentages use
kept detections as their denominator; coverage uses all 3,327 labelled test
rallies. Thresholds are rounded to six decimal places; each survey's JSON
records the exact value.

| Configuration | Limits, seconds before/after | Objective | Threshold | Test detections kept | Right boundaries | Fully correct | Fully correct coverage |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5/5 + 2/3 output cap | 1/1 | Keep half | 0.343045 | 2,193.2 | 1.15% | 0.57% | 0.38% |
| 5/5 + 2/3 output cap | 1/1 | Best F1 | 0.283042 | 2,519.8 | 1.28% | 0.58% | 0.44% |
| 5/5 + 2/3 output cap | 2/3 | Keep half | 0.343045 | 2,193.2 | 69.37% | 51.30% | 33.81% |
| 5/5 + 2/3 output cap | 2/3 | Best F1 | 0.346607 | 2,173.0 | 69.38% | 51.43% | 33.59% |
| No maximum context | 1/1 | Keep half | 0.325110 | 2,223.0 | 0.85% | 0.51% | 0.34% |
| No maximum context | 1/1 | Best F1 | 0.589647 | 1,279.6 | 0.69% | 0.55% | 0.21% |
| No maximum context | 2/3 | Keep half | 0.325110 | 2,223.0 | 31.39% | 21.89% | 14.63% |
| No maximum context | 2/3 | Best F1 | 0.238601 | 2,699.8 | 33.68% | 22.05% | 17.89% |

The 1/1 rule rejects nearly all discovery windows from both configurations.
The selected model deliberately retains up to 2/3 seconds of context, while
the no-max arm can retain more. These surveys measure window tightness as well
as contact completeness; their scores should be read separately from the
common-rule selection above.

Each [survey directory](surveys/) holds found-rally judgements, all validation
and test threshold curves, and `confidence_cutoffs.json.gz` with selected
thresholds and test results. [Reproduction commands](../../writeup_scripts/README.md)
reuse saved predictions. The uncapped experiment used revision `65311a3e`;
the selected default restores the capped finishing rule.

## Figures

Unpack the primary JSON/CSV files needed by the figure script into a temporary
directory, keeping `rally_courts.csv.gz` compressed. Then run from the repo root:

```bash
python experiments/annotator/monotrack_benchmark/writeup_scripts/make_figures.py \
  --data FIGURE_DATA \
  --confidence-data experiments/annotator/monotrack_benchmark/benchmark_results/discovery_context/surveys/5_5/2_3 \
  --out FIGURES
```

The court inputs and MonoTrack predictions are unchanged. A fresh annotator
hyperparameter search is a low-priority follow-up; selector-depth experiments
are separate from this comparison.
