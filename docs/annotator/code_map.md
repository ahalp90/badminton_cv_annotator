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
| Score and select contacts | [hybrid.py](../../src/annotator/hybrid.py) | Run the contact model, rally selector and review ranking |
| Build contact evidence | [contacts/features.py](../../src/annotator/contacts/features.py) | Choose search regions and measure shuttle/player signals |
| Select initial contacts | [contacts/model.py](../../src/annotator/contacts/model.py) | Score candidates, apply temporal separation and the cutoff |
| Shortlist rally contacts | [sequence/candidates.py](../../src/annotator/sequence/candidates.py) | Gather earlier/later contacts and their physical evidence |
| Propose rally sequences | [sequence/rally_options.py](../../src/annotator/sequence/rally_options.py) | Combine contact additions/deletions, order alternatives and build their features |
| Choose and finish rallies | [sequence/rally_model.py](../../src/annotator/sequence/rally_model.py) | Score proposals, select one per rally, finish bounds and sides, then build review evidence |
| Fit heuristic contact order | [contacts/selection.py](../../src/annotator/contacts/selection.py) | Filter heuristic contacts and use the same final player-alternation step as model inference |
| Estimate outcomes | [outcomes/video.py](../../src/annotator/outcomes/video.py) | Derive server, winner, landing and hit height |

In `hybrid.py`, `build_rally_context()` prepares the initial rallies and
shortlisted contacts for both inference and training. In `rally_model.py`,
`predict_rally_options()` chooses and finishes the rallies. The scoring and
finalisation helpers follow in the same file.

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
- `ScoredContacts` adds candidate probabilities and selected row indices.
- `ContactEvent` is a frame, contact probability and individual court-half guess.
- `ContactSequence` is a rally's bounds and the contacts it owns.
- `RallyCandidates`: shortlisted contacts by rally and their physical evidence.
- `RallyContext` groups the initial rallies, full contact stream and shortlisted evidence.
- `RallyOptions` groups the ordered alternatives and model-feature rows.
- `SequenceTrainingVideo` adds a video identity, training group and human labels to that same context.
- `RallyPrediction` retains the selected sequences, full contact stream and scores.
- `AnnotatorResult` is the public result saved by the dataset builder.

The full contact stream includes contacts outside every rally. A sequence's
membership is explicit; final bounds alone do not define that membership.
`RunCapture` holds optional mask copies, model predictions and landing diagnostics
beside the public result. It receives masks after preparation succeeds.

## Find a particular rule

| Question | Source |
| --- | --- |
| Why does a rally start or end here? | [rally/spans.py](../../src/annotator/rally/spans.py) |
| Which motion peaks become heuristic contacts? | [rally/contacts.py](../../src/annotator/rally/contacts.py) |
| How are players followed across frames? | [rally/evidence.py](../../src/annotator/rally/evidence.py) |
| Which court geometry applies to this frame? | [courts/scenes.py](../../src/annotator/courts/scenes.py) |
| Why is footage excluded? | [masks/dead.py](../../src/annotator/masks/dead.py) and [masks/replay.py](../../src/annotator/masks/replay.py) |
| Which earlier or missed contacts can be proposed? | [sequence/candidates.py](../../src/annotator/sequence/candidates.py) |
| Why did final rally bounds move? | [sequence/rally_model.py](../../src/annotator/sequence/rally_model.py) |
| Why did the striker half change? | [sequence/rally_model.py](../../src/annotator/sequence/rally_model.py) |
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

MonoTrack uses
[`benchmark_scripts/annotator_runner.py`](../../experiments/annotator/monotrack_benchmark/benchmark_scripts/annotator_runner.py)
for supplied-boundary and discovery comparisons. Its five-seed orchestration
lives in `writeup_scripts/annotator_runs.py`.

[`models.py`](../../src/annotator/models.py) is the single place for the selected
bundle location and model persistence checks. See [Inputs and outputs](inputs_outputs.md)
for saved schemas and [Maintenance](maintaining.md) for change-sensitive invariants.
