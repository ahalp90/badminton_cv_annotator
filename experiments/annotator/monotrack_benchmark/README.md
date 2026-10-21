# MonoTrack benchmark

[Read the current report](annotator_monotrack_own_weights_compared_writeup_v5.md)
for the comparison of our annotator and court detector with MonoTrack on
46 ShuttleSet22 videos, covering 3,327 rallies and 37,184 labelled contacts. It
uses the annotator's three-tree stack. The earlier
[v4](annotator_monotrack_own_weights_compared_writeup_v4.md),
[v3](annotator_monotrack_own_weights_compared_writeup_v3.md),
[v2](annotator_monotrack_own_weights_compared_writeup_v2.md) and
[first report](annotator_monotrack_own_weights_compared_writeup.md) are retained.

The later [tuned heuristic comparison](../heuristic_only_tune/KNOWN_RALLIES.md)
adds the tree-disabled annotator with the same supplied rally windows, both
sets of known-rally courts and the same scoring at ±10 and ±5 frames.

Both MonoTrack conditions use `hitnet_zeroed.h5`, with InpaintNet-filled shuttle
positions replaced by zeros and shuttle interpolation off at inference. Main
results use ±10 frames. Appendix A compares all four shuttle/interpolation
configurations; Appendix E gives the ±5-frame results.

- [Benchmark instructions](benchmark_scripts/README.md): current inference and
  scoring commands, including the five result directories used by the reports.
- [HitNet training](hitnet_retrain/README.md): training inputs, settings and
  exported weights.
- [Results](benchmark_results/comparison/): predictions and per-condition scores.
  Our annotator's runs for v5 are in
  [`benchmark_results/annotator_runs/`](benchmark_results/annotator_runs/).
- [Figures](writeup_figures/): charts, court overlays and inspected frames. The
  v5 charts are in `writeup_figures/v5/`; its court galleries and the v3 and v4
  figures are in `writeup_figures/v3/`.
- [Write-up scripts](writeup_scripts/README.md): the scripts behind the v5
  numbers, tables and figures.
- [Reviewed failure](benchmark_results/review_clips/53_0061.md): the annotated
  clip and its explanation.
