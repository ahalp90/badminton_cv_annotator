# Code map

Start with [`annotate_video()` in annotate_video.py](../../src/annotator/annotate_video.py).
It prepares the inputs, runs heuristic or learned contact inference, and estimates
rally results. The [source directory guide](../../src/annotator/README.md) gives
one starting file for each task.

## Follow one video

1. **Prepare contact inference.**
   [`prepare_annotation_inputs.py`](../../src/annotator/inference/prepare_annotation_inputs.py)
   assembles court geometry, tracks players, measures shuttle-to-wrist distances,
   excludes unusable frames and finds initial rallies and heuristic contacts.
   Training uses the same preparation.
2. **Detect contacts.** The learned route builds motion, wrist and visibility
   measurements in [`build_features.py`](../../src/annotator/contacts/build_features.py).
   [`score_candidates.py`](../../src/annotator/contacts/score_candidates.py) runs the
   fitted contact model, applies its cutoff and removes competing nearby hits.
   The heuristic route uses the rules in
   [`heuristics.py`](../../src/annotator/contacts/heuristics.py).
3. **Refine rallies.**
   [`build_contact_proposals.py`](../../src/annotator/rallies/build_contact_proposals.py)
   constructs alternative contact lists by adding plausible missed hits or removing
   suspected false hits. [`refine_rallies.py`](../../src/annotator/rallies/refine_rallies.py)
   scores those alternatives, selects one per rally and finishes its boundaries.
   A separate review model estimates the probability that each finished rally
   is correct. Lower scores suggest closer review; scores do not change annotations.
4. **Attribute players.**
   [`player_attribution.py`](../../src/annotator/contacts/player_attribution.py)
   measures wrist gaps and identifies individual hitters. Both inference routes
   use its player-alternation rule for the final contact order.
5. **Estimate rally results.**
   [`estimate_results.py`](../../src/annotator/rally_results/estimate_results.py)
   applies the landing and point-winner rules to each finished rally and measures
   hit heights. These are predictions; evaluation compares them with labels.

Preparation uses supplied rally bounds unchanged. With supplied contacts and no
bounds, it finds bounds on the original track. With neither supplied, it finds
bounds after excluding unusable footage. Player tracking uses the original
track; the excluded-frame copy is only for rally/contact detection.

## Courts and players

Player selection and running-position updates are shared with BST-X in
[`shared/player_tracking.py`](../../src/shared/player_tracking.py).
`track_sticky_players` in
[`annotator/player_tracking.py`](../../src/annotator/player_tracking.py) collects
the selected detections, ankle positions and box heights for accepted court
intervals. Contact attribution then measures separate `WristGaps`. Dataset
player features need only the `TrackedPlayers` result.

Court preparation and frame lookup answer different questions:

- [`prepare_courts.py`](../../src/annotator/courts/prepare_courts.py): build and
  validate court-detector results and their player-occupancy masks.
- [`frame_geometry.py`](../../src/annotator/courts/frame_geometry.py): find the
  accepted geometry that applies to a particular frame.

In the detector, [`view_matching.py`](../../src/court_detector/view_matching.py)
compares camera views so the detector can reuse a court calibration.

## Records and model inputs

Shared contact/rally records are in [`types.py`](../../src/annotator/types.py):
`ContactEvent` is one timed hit; `ContactSequence` is one rally interval with an
explicit ordered contact list; `FinishedRally` adds the final striker.
A rally's contact membership is explicit, rather than inferred from final bounds.
Contacts outside rallies remain in the full contact stream.

`PreparedAnnotationInputs` retains prepared arrays and initial detections.
`ContactFeatures` contains measured candidate rows and their search intervals.
`ScoredContacts` retains those rows, model probabilities and selected row indices.
`RallyOptions` contains alternative contact lists and their model features.
`RallyPrediction` retains the selected rallies and their scores.
`VideoAnnotation` returns the finished `AnnotatorResult` and both exclusion
masks. The dataset builder saves these masks. Pass `keep_working_data=True`
to also retain prepared inputs and model scores for inspection.

[`models.py`](../../src/annotator/models.py) loads and saves the three-model bundle.
Feature construction and scoring are separate functions so training can build
inputs without running a fitted model. The heuristic-only route does not build
the learned contact-feature matrix.

## Training, scoring and tools

[`training/workflow.py`](../../src/annotator/training/workflow.py) fits and evaluates
the current learned models. [`training/sequences.py`](../../src/annotator/training/sequences.py)
defines their contact matching and whole-rally correctness.

Use [`experiments/annotator/legacy_evaluation/`](../../experiments/annotator/legacy_evaluation/)
for the earlier three-video fixture harness, static-court preparation and
ground-truth scoring. The historical greedy contact matching
and the current model's chronological optimal matching are different metrics.
Do not combine their results without identifying which metric was used.
Pinned ShuttleSet and commentary benchmark runners are under
[`experiments/annotator/shuttleset_benchmarks/`](../../experiments/annotator/shuttleset_benchmarks/).
Production commentary loading is in
[`dataset_builder/commentary_inputs.py`](../../src/dataset_builder/commentary_inputs.py).

Manual scene and rally-start editors are in [`labelling/`](../../src/labelling/).
They do not yet provide a general editor for rally start/stop, every contact and
its top/bottom hitter. [`video_overlay/`](../../src/video_overlay/) renders saved
shuttle predictions on source footage.

The former track-only batch segmentation command and its overcount-based doubles
filter have been retired. The full annotator still supports heuristic-only
inference. Scraper title-based doubles screening and BST-X player-overcount
outputs remain separate, active features.
