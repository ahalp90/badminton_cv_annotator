# Discovery context comparison

The **5/5-trained configuration with its existing 2/3 discovery cap** wins the
validation comparison: 296.8 fully correct rallies per seed, against 294.0
for training and discovery without maximum context limits. This is 2.8 more
rallies, 0.42 percentage points or a 0.95% relative improvement. The difference
does not establish statistical superiority or isolate training from finishing.

## Selection

Discovery boundaries are the **first and last accepted predicted contacts**.
Both configurations are judged at ±10 base-30 frames under the same rule
without maximum outward context. A correct rally covers one complete labelled
rally without merging neighbours and contains every contact with the right
player, without extras. Contact F1 uses the whole stream, including
outside-rally events.

| Configuration | Fully correct / 668 | Share | Contact F1 | Correct-player F1 | Fully correct / 354 unflagged |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5/5 training; 2/3 discovery cap | 296.8 | 44.43% | 89.881% | 85.320% | 223.6 (63.16%) |
| No maximum context in training or discovery | 294.0 | 44.01% | 90.136% | 85.336% | 222.6 (62.88%) |

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
With supplied bounds, the selected model scores 96.714% contact F1, 94.702%
correct-player F1 and 60.024% fully correct rallies.
In discovery, the selected model scores 93.482% contact F1, 90.820%
correct-player F1 and 54.944% fully correct rallies.
Rally correctness uses no maximum outward context limit. The [primary results](primary/)
contain all conditions, both timing tolerances, per-video counts and paired intervals.

### Both configurations on validation and test

Both use the 32-video contact fits and work from whole videos, finding their
own rallies. The table uses the common first/last-contact rule, no maximum
outward context and ±10 base-30 frames inward/contact-matching allowance.
Values are five-seed means. F1 includes contacts outside found rallies.

| Split | Configuration | Contact F1 | Correct-player F1 | Fully correct rallies | Share fully correct |
|---|---|---:|---:|---:|---:|
| Validation | 5/5 | 89.881% | 85.320% | 296.8 / 668 | 44.431% |
| Validation | No-max | 90.136% | 85.336% | 294.0 / 668 | 44.012% |
| Test | 5/5 | 93.482% | 90.820% | 1,828.0 / 3,327 | 54.944% |
| Test | No-max | 93.651% | 90.964% | 1,828.4 / 3,327 | 54.956% |

5/5 recovers 2.8 more fully correct validation rallies per seed. On test, the
arms are effectively tied: no-max recovers 0.4 more rallies out of 3,327,
a difference of 0.012 percentage points. No-max improves test contact F1 by
0.168 points and correct-player F1 by 0.144 points. Test boundary coverage is
72.462% for 5/5 and 72.456% for no-max. Test results do not change the
validation-selected default.

The [comparison summary](discovery_comparison.json.gz), [5/5 primary results](primary/)
and [no-max test results](no_max_test/) contain the underlying values and
per-seed outcomes. No-max uses the same `metrics.score_stream_for_item` scorer
at tolerance 10 with `supplied=False` and `RallyBoundaryPolicy(None, None)`;
all five seeds passed the saved full-stream contact-count checks.

## Finite-boundary confidence surveys

Both configurations are surveyed using their five saved prediction runs. The boundary-only measure checks the first and last accepted predicted
contacts against the labelled rally: at most 1/1 or 2/3 seconds outward, with
±10 base-30 frames inward allowance. The contact interval must cover exactly
one complete labelled rally. Interior contact errors and player attribution do
not affect this measure. Empty detections fail it and remain in the denominator.
Fully correct additionally requires all contacts and hitters to be right; its
results are identical under 1/1, 2/3 and the common no-maximum-context rule.
Dataset clips separately add 2/3 seconds to accepted contact endpoints, clipped
only at video edges; clips may overlap.

Every unique validation score was searched: 3,339 for 5/5 and 3,333 for no-max. The objectives are closest
to half the pooled validation detections retained, highest mean retrieval F1
for fully correct rallies, and the lowest threshold reaching 99% mean boundary
accuracy with detections retained in all five seeds. The last target averages
seed percentages; it does not require every seed to reach 99%.

The next table reports 5/5 test results at validation-selected thresholds;
the comparison with no-max follows below. Counts and
shares are averaged separately. Boundary and fully-correct percentages use
kept detections as their denominator; coverage uses all 3,327 labelled test
rallies. Thresholds are rounded to six decimals; JSON records the exact value.

| Limits before/after | Objective | Threshold | Test detections kept | Right boundaries | Fully correct | Fully correct coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1/1 | Keep half | 0.343045 | 2,193.2 | 78.69% | 68.22% | 44.97% |
| 1/1 | Best F1 | 0.422554 | 1,837.4 | 80.79% | 71.95% | 39.73% |
| 1/1 | 99% mean validation boundaries | 0.880410 | 26.2 | 95.88% | 92.54% | 0.73% |
| 2/3 | Keep half | 0.343045 | 2,193.2 | 82.58% | 68.22% | 44.97% |
| 2/3 | Best F1 | 0.422554 | 1,837.4 | 84.48% | 71.95% | 39.73% |
| 2/3 | 99% mean validation boundaries | 0.873122 | 36.4 | 95.73% | 94.24% | 1.02% |

The 5/5 configuration reaches the 99% mean validation target, retaining only
7.0 detections per seed for 1/1 and 9.6 for 2/3. Test accuracy is 95.88% and
95.73%, respectively. No-max reaches neither 99% validation target. These tiny
retained sets are a poor basis for choosing between the configurations.

### Boundary-only comparison

The following thresholds each retain about half of the validation detections
(377 per seed for both arms): 0.343045 for 5/5 and 0.325110 for no-max. They
were then applied unchanged to test. Percentages are mean per-seed accuracy
among retained detections, not coverage of all labelled rallies.

| Limits before/after | Configuration | Validation boundaries right | Test boundaries right | Test detections kept |
|---|---|---:|---:|---:|
| 1/1 | 5/5 | 76.91% | 78.69% | 2,193.2 |
| 1/1 | No-max | 77.25% | 78.31% | 2,223.0 |
| 2/3 | 5/5 | 78.50% | 82.58% | 2,193.2 |
| 2/3 | No-max | 79.10% | 82.28% | 2,223.0 |

For a workflow that finds the rally's outer contacts and leaves the interior
to a person, the two configurations are close. No-max has a small accuracy
advantage at stricter practical thresholds; 5/5 is slightly ahead at the
keep-half point on test. Under 2/3, retaining 10% of validation detections
gives 91.96% test boundary accuracy for no-max versus 90.66% for 5/5, with
about 390 detections kept by each. At 25%, the figures are 87.79% and 87.19%.
These small differences do not establish a clear overall winner.

For a broader comparison, thresholds below were chosen to retain 10%, 25%,
50% or 75% of pooled validation detections. All use the 2/3 boundary rule;
the test counts show the coverage trade-off. The 1/1 results show the same
small no-max advantage at 10% and 25%. The [full retention comparison](boundary_retention.json.gz)
includes both rules, validation/test accuracy, counts and labelled-rally coverage.

| Validation detections retained | 5/5 test boundary accuracy | No-max test boundary accuracy | 5/5 test detections kept | No-max test detections kept |
|---|---:|---:|---:|---:|
| 10% | 90.66% | 91.96% | 395.4 | 389.6 |
| 25% | 87.19% | 87.79% | 1,192.8 | 1,150.2 |
| 50% | 82.58% | 82.28% | 2,193.2 | 2,223.0 |
| 75% | 78.48% | 78.79% | 2,900.6 | 2,889.4 |

Each [survey directory](surveys/) holds found-rally judgements, validation
and test threshold curves, and exact operating points. [Reproduction commands](../../writeup_scripts/README.md)
reuse saved predictions. The uncapped experiment used revision `65311a3e`;
the default retains the validation-selected capped configuration.

## Figures

Unpack the primary JSON/CSV files needed by the figure script into a temporary
directory, keeping `rally_courts.csv.gz` compressed. Then run from the repo root:

```bash
python experiments/annotator/monotrack_benchmark/writeup_scripts/make_figures.py \
  --data FIGURE_DATA \
  --confidence-data experiments/annotator/monotrack_benchmark/benchmark_results/discovery_context/surveys/5_5/2_3 \
  --out FIGURES
```

Court inputs and MonoTrack predictions are unchanged. A fresh annotator
hyperparameter search and selector-depth experiments are separate follow-ups.
