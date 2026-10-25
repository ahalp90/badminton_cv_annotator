# Consumed selector targets

These files compare the labels actually used to fit the rally selectors across
the three boundary policies. Proposal sequences and feature arrays were checked
for equality before comparing labels. Each policy pair covers 20,416,782 rows
across two feature configurations, five seeds and 40 training/validation videos.
Rows are proposal examples, not distinct rallies or contacts.

| Training policies compared | Changed labels | Positive → negative | Negative → positive | Spans losing all positive options | Spans gaining a positive option |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2/3 → 3/3 | 7,109 | 1,250 | 5,775 | 0 | 1,430 |
| 2/3 → 5/5 | 12,037 | 1,697 | 10,109 | 0 | 3,487 |
| 3/3 → 5/5 | 5,110 | 538 | 4,425 | 0 | 2,057 |

The fitted selectors use five-frame-preferred targets: within a span, use
±5-frame labels when at least one judged option satisfies that tolerance;
otherwise use ±10-frame labels. Frame tolerances scale from 30 fps. Widening
the boundary allowance can enable a five-frame-positive option and therefore
switch other options in that span from positive to negative. No span loses all
its positive options in these comparisons. Confidence-model targets and the
headline evaluation use ±10 frames.

`target_differences_per_video.csv.gz` contains all 1,200 video/policy-pair records.
`target_differences.csv.gz` and `target_comparison.json.gz` summarise them by
feature configuration, seed, split and policy pair. Labels are −1 (unjudgeable),
0 (incorrect) and 1 (correct); the files include every transition, including
changes involving unjudgeable rows.
