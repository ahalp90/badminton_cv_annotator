# Code map

Start with [`run_video()`](../../src/annotator/run_video.py). It shows how one
video becomes an annotation: prepare evidence and rough rallies, select contacts,
then estimate outcomes. Follow its calls to inspect a particular rule.

Read [How it works](how_it_works.md) alongside the code if the badminton terms or
intermediate results are unfamiliar.

## Follow one video

| Step | Read next | What it decides |
| --- | --- | --- |
| Prepare rough rallies | `prepare_annotation()` in [preparation.py](../../src/annotator/preparation.py) | Build player evidence, exclude unusable footage, find spans and heuristic contacts |
| Score and select contacts | [learned_model_inference.py](../../src/annotator/learned_model_inference.py) | Run the contact model, rally selector and review ranking |
| Build contact evidence | [contacts/features.py](../../src/annotator/contacts/features.py) | Choose search regions and measure shuttle/player signals |
| Select initial contacts | [contacts/model.py](../../src/annotator/contacts/model.py) | Score candidates, apply temporal separation and the cutoff |
| Shortlist rally contacts | [sequence/rally_options.py](../../src/annotator/sequence/rally_options.py) | Choose possible earlier serves and missed contacts |
| Propose rally sequences | [sequence/rally_options.py](../../src/annotator/sequence/rally_options.py) | Prepare contact evidence, combine additions/deletions, order alternatives and build their features |
| Choose and finish rallies | [sequence/rally_model.py](../../src/annotator/sequence/rally_model.py) | Score proposals, select one per rally, finish bounds and sides, then build review evidence |
| Fit player order | [sequence/contacts.py](../../src/annotator/sequence/contacts.py) | Choose the alternating order from player guesses and wrist distances; retain the first and final striker |
| Fit heuristic contact order | [heuristic_inference.py](../../src/annotator/heuristic_inference.py) | Filter heuristic contacts and use the same final player-alternation step as model inference |
| Estimate outcomes | [outcomes/video.py](../../src/annotator/outcomes/video.py) | Derive server, winner, landing and hit height |

Read `rally_options.py` from `build_rally_context()` to `build_rally_options()`:
prepare the initial rallies and shortlisted contacts, then build the alternatives
and their model rows. Both inference and training use these two steps.
The shortlisting rules follow the two builders in the same file, then
`toggle_proposals()` defines contact edits, duplicate removal and ordering.
`shortlisted_by_span` contains possible additions to each rally. Unused-contact
features consider that shortlist and every contact in any proposal, including stream contacts
absorbed when an earlier serve extends a rally.
The detailed feature calculations come afterwards in the same file, including
contact probabilities, spacing, raw player-side agreement and unused candidates.

In `rally_model.py`, `predict_rally_options()` chooses and finishes the rallies.
The scoring and boundary helpers follow in the same file. Both inference routes
use `fit_rally_sides()` in `sequence/contacts.py` for player order; the voting
and tie-breaking rules follow it there. `build_review_features()` uses the
selected proposal’s raw player guesses and final rally bounds, calling
`sequence_summary()` for those values.

All three trees use `positive_probabilities()` in
[contacts/model.py](../../src/annotator/contacts/model.py). It reads the class-1
probability from trees fitted on labels 0 and 1.

Preparation uses supplied rally bounds unchanged. With supplied contacts and no
bounds, it finds bounds on the original track. With neither supplied, it finds
bounds on the track after dead-time masking. Replay-mask estimation also uses
original-track bounds when needed.

`heuristic_only=True` uses the same preparation, then selects rule-based contacts
and fits their player order before outcome estimation. Supplying `spans` lets
callers compare contact detection within known rally bounds.

Training and experiments call `prepare_annotation()` directly to obtain initial
rallies and feature inputs without selecting final contacts or estimating outcomes.
The rally CLI calls `segment_shuttle_track()` in
[rally_segmentation.py](../../src/annotator/rally_segmentation.py) for tracks without
player or court evidence.

## Know which representation you are reading

- `AnnotationEvidence` contains original arrays, resolved settings, masks and initial rally/contact evidence.
- `FinishedRally` pairs a selected contact sequence with its fitted final striker. Both inference routes pass it directly to outcomes.
- `ContactFeatures` contains candidate rows ready for scoring and their search intervals.
- `ScoredContacts` retains the scored feature matrix, candidate probabilities and selected row indices.
- `ContactEvent` is a frame, contact probability and individual court-half guess.
- `ContactSequence` is a rally's bounds and the contacts it owns.
- `RallyContext`: initial rallies, the full contact stream, shortlisted contacts
  by rally and physical feature rows by frame.
- `RallyOptions` groups the ordered alternatives and model-feature rows.
- `SequenceTrainingVideo` adds a video identity, training group and GT labels to that same context.
- `RallyPrediction` retains the selected sequences, full contact stream and scores.
- `AnnotatorResult` is the public result saved by the dataset builder.

The full contact stream includes contacts outside every rally. A sequence's
membership is explicit; final bounds alone do not define that membership.
Use `RunCapture` to retain prepared evidence and model predictions
beside the public result. Prepared evidence includes both exclusion masks.

## Find a particular rule

| Question | Source |
| --- | --- |
| Why does a rally start or end here? | [rally/spans.py](../../src/annotator/rally/spans.py) |
| Which motion peaks become heuristic contacts? | [rally/contacts.py](../../src/annotator/rally/contacts.py) |
| How are players followed across frames? | [rally/evidence.py](../../src/annotator/rally/evidence.py) |
| Where are court halves and shared record types? | [types.py](../../src/annotator/types.py) |
| How are image pixels projected onto the court? | [shared/court.py](../../src/shared/court.py) |
| How is corner uncertainty estimated? | `corner_error_band_from_corners()` in [courts/evidence.py](../../src/annotator/courts/evidence.py) |
| Which court geometry applies to this frame? | [courts/scenes.py](../../src/annotator/courts/scenes.py) |
| Why is footage excluded? | [masks/replay.py](../../src/annotator/masks/replay.py) |
| How does excluded footage read as rest? | `apply_replay_mask()` in [rally_segmentation.py](../../src/annotator/rally_segmentation.py) |
| Where are the shared array calculations? | [calc_helpers.py](../../src/annotator/calc_helpers.py) |
| Which earlier or missed contacts can be proposed? | [sequence/rally_options.py](../../src/annotator/sequence/rally_options.py) |
| Why did final rally bounds move? | [sequence/rally_model.py](../../src/annotator/sequence/rally_model.py) |
| Which player hit in this camera scene? | [contacts/attribution.py](../../src/annotator/contacts/attribution.py) |
| Why did the striker half change? | [sequence/contacts.py](../../src/annotator/sequence/contacts.py) |
| How is a winner or landing judged? | [outcomes/point_winner.py](../../src/annotator/outcomes/point_winner.py) |
| How does a setting scale with frame rate? | [config.py](../../src/annotator/config.py) |

## Training and experiments

[`training/workflow.py`](../../src/annotator/training/workflow.py) is the fitting
entry point. It reads a manifest, calls shared preparation, fits contacts, then
fits rally selection and review. The production proposal and feature builders
are also used during fitting.

[`training/contact.py`](../../src/annotator/training/contact.py) owns contact
sampling and fitting. [`training/rally.py`](../../src/annotator/training/rally.py)
owns grouped selector/review training. [`training/rally_targets.py`](../../src/annotator/training/rally_targets.py)
judges candidate sequences. [`training/sequences.py`](../../src/annotator/training/sequences.py)
holds labelled rally data, contact matching and rally correctness.

Use the [refit guide](retuning.md) for configuration and result persistence.
Heuristic searches live in `experiments/annotator/heuristic_only_tune`; the
current fitted-model comparison lives in `experiments/annotator/tree_simplification`.
Historical reports explain earlier choices without adding branches to inference.

## Callers and saved results

[`cli.py`](../../src/annotator/cli.py) and
[`dataset_builder/vision.py`](../../src/dataset_builder/vision.py) load saved
extraction artefacts. The latter supplies the dataset builder's full annotation
stage. [`artifact_io.py`](../../src/annotator/artifact_io.py) serialises the
public result and masks.

In [`dataset_builder/features.py`](../../src/dataset_builder/features.py),
player tracking and position projection use the same prepared scene courts.

MonoTrack uses
[`benchmark_scripts/annotator_runner.py`](../../experiments/annotator/monotrack_benchmark/benchmark_scripts/annotator_runner.py)
for supplied-boundary and discovery comparisons. Its five-seed orchestration
lives in `writeup_scripts/annotator_runs.py`.

[`models.py`](../../src/annotator/models.py) is the single place for the selected
bundle location and model persistence checks. See [Inputs and outputs](inputs_outputs.md)
for saved schemas and [Maintenance](maintaining.md) for change-sensitive invariants.

Commentary evaluation writes `issue104-commentary-benchmark/5`. GT rally-view
fields use `gt_contact` in their names, including `gt_contact_pairing`,
`aligned_gt_contact_pairing` and `aligned_multi_gt_contact_pairing`. The input
corpus and rally-record schemas are unchanged. Earlier benchmark reports retain
the field names and schema versions used for those runs.
