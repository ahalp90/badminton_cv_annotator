# Whole-rally exclusion results

These results use the annotator refitted after excluding unreliable train and
validation rallies from whole-rally learning and scoring. Contact fitting and
contact scoring retain all master contacts. The existing ShuttleSet22 test
protocol is unchanged: 3,327 scored rallies and 37,184 contacts across 46 videos.

## Validation comparison

Both fits use the same 347 judgeable validation rallies and all 5,696 contacts.
The 321 excluded rallies remain available for contact matching and neighbouring-rally
checks; a prediction merging rallies still counts as wrong. Discovery boundaries
are first and last accepted predicted contacts, with ±10 base-30-frame tolerance
and no maximum outward context. Values are five-seed means.

| Fit | Fully correct / 347 | Share | Contact F1 | Correct-player F1 |
| --- | ---: | ---: | ---: | ---: |
| Previous whole-rally labels | 222.2 | 64.03% | 90.136% | 85.336% |
| Unreliable whole-rally labels excluded | 227.2 | 65.48% | 90.211% | 85.411% |

[Per-seed comparison and rules](policy_selection.json.gz).
The refit corrects unreliable supervision; this small gain is not evidence of
statistical superiority. Test fully correct rallies fall from 60.34% to 55.73%
with supplied bounds and from 54.96% to 51.93% in discovery. The [model record](../../../tree_simplification/results/discovery_context/README.md)
links all fifteen bundles and fit settings. Contact trees, preprocessing and
proposal features were reused; selector and review trees were refitted.

## Test results and confidence

The [primary results](primary/) contain both timing tolerances, per-video counts,
all seven refitted annotator conditions and paired intervals. MonoTrack, trees-off
and court reference results are unchanged. MonoTrack's full 60-run validation
reselection retained zeroed shuttle input, L2=0 and interpolation off, so its
saved test predictions were reused.

The [1/1 survey](surveys/no_max/1_1/) and [2/3 survey](surveys/no_max/2_3/) search
all attainable validation confidence thresholds. The first/last accepted contacts
must cover one complete labelled rally, with at most the specified seconds of
outward context and ±10 base-30 frames inward allowance. Each directory contains
per-detection judgements, complete validation/test curves and validation-selected
cutoffs with test outcomes. The README and report also retain fixed thresholds
0.50, 0.70, 0.75, 0.80 and 0.85. Test outcomes do not choose thresholds.

The [report](../../annotator_monotrack_own_weights_compared_writeup_v7.md)
gives the numerical comparisons and limits. [Reproduction commands](../../writeup_scripts/README.md)
reuse saved predictions. Exclusion counts and reasons are in the
[development population record](../rally_exclusions/label_counts.json.gz).

## Figures

Unpack primary JSON/CSV files into a temporary directory, keeping
`rally_courts.csv.gz` compressed, then run from the repository root:

```bash
python experiments/annotator/monotrack_benchmark/writeup_scripts/make_figures.py \
  --data FIGURE_DATA \
  --confidence-data experiments/annotator/monotrack_benchmark/benchmark_results/discovery_context/surveys/no_max/2_3 \
  --out FIGURES
```
