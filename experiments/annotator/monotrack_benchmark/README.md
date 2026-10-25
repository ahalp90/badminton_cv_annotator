# MonoTrack benchmark

[Read the current report](annotator_monotrack_own_weights_compared_writeup_v7.md)
for the annotator and court-detector comparison with MonoTrack on 46 ShuttleSet22
videos: 3,327 scored rallies and 37,184 labelled contacts. Results use five seeds
and validation-selected settings. The annotator uses the 2/3-second boundary
training policy; the main contact tolerance is ±10 frames, with ±5 in Appendix E.

## Results and models

| Find | Location |
| --- | --- |
| All nine training/evaluation boundary combinations | [Validation and test table](benchmark_results/boundary_retrain_v7/README.md#training-and-evaluation-boundary-allowances) |
| Current scores, confidence curves and training-target comparison | [Boundary-policy results](benchmark_results/boundary_retrain_v7/) |
| Selected annotator models and fit settings | [Model record](../tree_simplification/results/boundary_retrain_v7/README.md) |
| MonoTrack selection, five-seed predictions and checkpoints | [MonoTrack results](benchmark_results/monotrack_five_seed/) |
| Prepared court and rally inputs | [Inputs](benchmark_results/inputs/) |
| Tree-disabled comparison | [Tuned heuristic results](../heuristic_only_tune/KNOWN_RALLIES.md) |

## Rerun

Run from the repository root. To rescore predictions from the full comparison:

```bash
W=experiments/annotator/monotrack_benchmark/writeup_scripts
python "$W/metrics.py" \
  --annotator-runs NEW_RUN/reporting/annotator_runs/2_3 \
  --out REPORT_DATA \
  --boundary-before-seconds 2 --boundary-after-seconds 3
```

This path assumes `boundary_comparison.py --output NEW_RUN/reporting` produced
the predictions. The compact results contain scores and tables; annotator
prediction streams are produced by the
[training and inference commands](writeup_scripts/README.md#batched-inference-and-resume).
The scoring command uses the saved MonoTrack five-seed predictions by default.

- [Prepare inputs and run MonoTrack](benchmark_scripts/README.md).
- [Train HitNet](hitnet_retrain/README.md).
- [Retrain the annotator, compare policies, select confidence cut-offs and draw figures](writeup_scripts/README.md).

The current report links its figures directly. Its updated charts are in
`writeup_figures/v7/`; shared court charts and galleries remain alongside them.

## Earlier comparisons

Earlier reports are kept together: [v1](archive/reports/v1.md),
[v2](archive/reports/v2.md), [v3](archive/reports/v3.md),
[v4](archive/reports/v4.md), [v5](archive/reports/v5.md) and
[v6](archive/reports/v6.md). Each retains its original results and scoring rules.
Their exclusive figures are in [archive/figures/](archive/figures/).

[Historical results](archive/results/) contain the original condition comparison,
the pre-v7 annotator runs and the confidence rerun/curves. The
[reviewed court-fit failure](benchmark_results/review_clips/53_0061.md) refers to
those original runs. Current MonoTrack results and the earlier HitNet training
grid remain in their locations linked above.
