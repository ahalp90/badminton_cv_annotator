# MonoTrack benchmark

[Read the current report](annotator_monotrack_own_weights_compared_writeup_v2.md)
for the comparison of our annotator and court detector with MonoTrack on
46 ShuttleSet22 videos, covering 3,327 rallies and 37,184 labelled contacts.
The [first report](annotator_monotrack_own_weights_compared_writeup.md) is also
retained.

Both MonoTrack conditions use `hitnet_zeroed.h5`, with InpaintNet-filled shuttle
positions replaced by zeros and shuttle interpolation off at inference. Main
results use ±10 frames. Appendix A compares all four shuttle/interpolation
configurations; Appendix E gives the ±5-frame results.

- [Benchmark instructions](benchmark_scripts/README.md): current inference and
  scoring commands, including the five result directories used by the reports.
- [HitNet training](hitnet_retrain/README.md): training inputs, settings and
  exported weights.
- [Results](benchmark_results/comparison/): predictions and per-condition scores.
- [Figures](writeup_figures/): charts, court overlays and inspected frames.
- [Reviewed failure](benchmark_results/review_clips/53_0061.md): the annotated
  clip and its explanation.
