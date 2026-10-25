# MonoTrack benchmark

[Read the current report](annotator_monotrack_own_weights_compared_writeup_v7.md)
for the comparison of our annotator and court detector with MonoTrack on
46 ShuttleSet22 videos, covering 3,327 rallies and 37,184 labelled contacts. It
uses the annotator's three-tree stack. The earlier
[v6](annotator_monotrack_own_weights_compared_writeup_v6.md),
[v5](annotator_monotrack_own_weights_compared_writeup_v5.md), [v4](annotator_monotrack_own_weights_compared_writeup_v4.md),
[v3](annotator_monotrack_own_weights_compared_writeup_v3.md),
[v2](annotator_monotrack_own_weights_compared_writeup_v2.md) and
[first report](annotator_monotrack_own_weights_compared_writeup.md) are retained.

The later [tuned heuristic comparison](../heuristic_only_tune/KNOWN_RALLIES.md)
adds the tree-disabled annotator with the same supplied rally windows, both
sets of known-rally courts and the same scoring at ±10 and ±5 frames.

Both MonoTrack court conditions use zeroed shuttle input, L2 0 and inference
interpolation off, chosen by five-seed mean fully correct validation rallies.
Test results report the mean and min–max across those five seeds. Main results
use ±10 frames. Appendix A gives the validation comparison; Appendix E gives
the ±5-frame test results.

- [Benchmark instructions](benchmark_scripts/README.md): current inference and
  scoring commands.
- [HitNet training](hitnet_retrain/README.md): training inputs, settings and
  exported weights.
- [MonoTrack results](benchmark_results/monotrack_five_seed/): validation
  selection, five-seed test predictions, scores and training checkpoints.
  The retrained annotator's compact results are in
  [`benchmark_results/boundary_retrain_v7/`](benchmark_results/boundary_retrain_v7/).
- [Confidence curves](benchmark_results/boundary_retrain_v7/primary/): kept-rally
  results at every distinct validation score and test results at those thresholds.
- [Figures](writeup_figures/): charts, court overlays and inspected frames. The
  updated comparison charts are in `writeup_figures/v7/`, with unchanged charts
  in `writeup_figures/v5/` and court galleries in `writeup_figures/v3/`.
- [Write-up scripts](writeup_scripts/README.md): the scripts behind the v7
  numbers, tables and figures.
- [Reviewed failure](benchmark_results/review_clips/53_0061.md): the annotated
  clip and its explanation.
