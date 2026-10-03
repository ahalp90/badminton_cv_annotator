# Court detector: development and experiments

The court detector turns badminton footage into court geometry: four corners that let
the project map image positions onto the court floor. It was built to work beyond the
camera views and venues covered by an existing court model. The difficult part is
deciding which visible lines belong to the playing court, then keeping that decision
reliable through occlusion and camera cuts.

**[The development story](report.md)** explains why a replacement was needed, how the
detector works, what the experiments changed, and where it still fails.

| Topic | Documentation |
| --- | --- |
| Current design choices | [Design guide](../../docs/court_detector/design.md) |
| Setup, operation and outputs | [Operator guide](../../src/court_detector/README.md) |
| Existing 86-video extraction | [Released predictions and loading example](../../data/court_detections/sset_and_sset22/extractions_20261003/README.md) |
| Earlier approaches and development measurements | [Earlier approaches](../../docs/court_detector/earlier_approaches.md) and [development evaluation](../../docs/court_detector/evaluation.md) |
| Released results | [Release evaluation](released_dataset_evaluation/README.md) and [court gallery](released_dataset_evaluation/gallery.md) |

## Supporting experiments and tools

These folders answer specific questions raised in the story:

- [Search trial](search_policy_trial/README.md): would relaxing player checks
  recover useful courts, and what false detections would it introduce?
- [Original extraction](baseline/README.md): what did the same 86 videos look
  like before the court-sharing repair?
- [Fixed-frame checks](saved_views/README.md): how can a detector change be
  compared on the same saved images and observations?
- [Saved inputs](inputs/README.md) and [reproduction commands](reproduce.md):
  how can the release and search-trial measurements be rebuilt?

The earlier line-search prototype and its recorded comparisons live under [the
independent-court experiment](../annotator/independent_court/README.md). The maintained
implementation is in `src/court_detector/`.
