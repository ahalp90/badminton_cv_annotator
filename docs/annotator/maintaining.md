# Maintainer guide

This guide maps code changes to affected modules, refitting requirements and
relevant tests. The [fixed heuristics](heuristics.md)
and [tree model stack](tree_stack.md) explain the behaviour behind those decisions.

**Contents**

| Planning a change | Annotation stages | Checks and supporting files |
| --- | --- | --- |
| [Main data path](#main-data-path) | [Rally opening and contact search](#rally-opening-and-rough-contact-search) | [Tests by subsystem](#tests-by-subsystem) |
| [Refit requirements](#does-a-code-change-require-a-refit) | [Contact model](#contact-model) | [Debugging wrong contacts](#debugging-order-for-wrong-contacts) |
| [Model compatibility](#model-loading-and-compatibility) | [Sequence refinement](#sequence-refinement) | [Files outside src/annotator](#files-outside-srcannotator) |
| [FPS scaling](#fps-scaling) | [Court and player evidence](#court-and-player-evidence) |  |
| [Shared training and annotation code](#training-and-annotation-share-code) | [Masks and shuttle guards](#exclusion-masks-and-shuttle-guards) |  |
|  | [Outcome rules](#outcome-rules) |  |

## Main data path

Most changes to the annotation runtime pass through this sequence:

```text
saved metadata / shuttle / pose / court
        ↓
run_video.py
        ↓
rally + masks + courts
        ↓
hybrid.py
        ↓
contacts → sequence
        ↓
outcomes
        ↓
persisted annotation files
```

Evaluation, review GUIs and VLM benchmark code sit beside this path and do not run during normal annotation.

## Does a code change require a refit?

| Change | New fit? | Reason |
| --- | --- | --- |
| Documentation, logging or display-only code | No | Model inputs and choices are unchanged |
| Output serialisation only | Usually no | Fitted decisions are unchanged; saved-file compatibility still needs checking |
| Landing, winner or hit-height logic after final contacts | No for contact/sequence models | These rules run after final contact selection |
| Contact score cutoff | Yes for a like-for-like model comparison | It changes the initial contact stream seen by the sequence and review models |
| Rule for guarded candidates (`reject_masked_without_player`) | Yes for a like-for-like model comparison | It changes the candidate rows seen by the sequence and review models; the contact tree's own training rows are unchanged |
| Contact feature values, windows, units or feature order | Yes | The contact tree receives different inputs |
| Candidate-region generation | Yes | Different frames become eligible for contact scoring |
| Rally-span rules that feed model features/options | Yes | Feature rows and sequence alternatives change |
| Sequence candidate generation or legal edits | Yes | The sequence models are choosing from a different set of alternatives |
| Sequence feature definitions | Yes | The fitted chooser inputs change |
| Side-geometry mode (`video`/`scene`) | Yes | Raw court-half evidence changes |
| Substantial shuttle, pose or court extraction change | Usually yes | The fitted models see materially different input values or geometry |
| scikit-learn version change | Yes | The model loader requires an exact fit/runtime version match |

The practical question is whether the change alters a value or choice presented to a fitted tree. If it does, the old model directory was fitted for different inputs.

## Rally opening and rough contact search

Relevant modules:

- `rally/spans.py` — rally opening and closing;
- `rally/serve.py` — serve setup and look-back rules;
- `rally/trajectory.py` — shared shuttle transforms;
- `rally/contacts.py` — heuristic contact impulses and gates;
- `contacts/features.py` — the wider search regions presented to the contact tree;
- `config.py`, `fps_constants.py`, `resolve.py` — rough-rally/mask settings and FPS scaling.

A change here often affects the training examples as well as annotation. The training code calls the same rough-rally and feature builders.

## Contact model

Relevant modules:

- `contacts/features.py` — feature rows and search regions;
- `contacts/model.py` — feature order, probability cutoff, the optional rule for guarded candidates and nearby-contact suppression;
- `training/contact.py` — row selection and contact-tree fitting.

The saved contact tree depends on the exact order in `CONTACT_FEATURE_NAMES`. `metadata.json` records that order, and loading fails if the current code disagrees.

The score cutoff is different: it is stored in `ContactModelConfig` inside `models.joblib`. A different cutoff changes which initial contacts reach sequence refinement, so model comparisons are clearest when each cutoff has its own complete fitted model directory.

`ContactModelConfig.reject_masked_without_player` is also stored in the bundle
and defaults to off. When enabled, it removes unreliable shuttle candidates
that have no selected player nearby. Both annotation (`hybrid.predict_contacts()`)
and sequence training (`training/workflow.py::sequence_training_video()`) apply
the rule through `score_contact_features()`, keeping their behaviour aligned.

The rule requires the shuttle hallucination mask that `run_video()` places on
`ContactEvidence`. A caller enabling it must supply that mask; otherwise scoring
raises an error.

Two details of contact-tree fitting affect reproducibility:

- Distant negatives are sampled with a fixed seed in the order the training videos are supplied.
- The order of the selected rows also affects the fitted tree. `fit_contact_model()` takes an optional `fit_video_order` that reorders the selected rows by video without changing the sampling. `fit_from_manifest()` and the `fit` command do not pass it, so both orders follow the manifest.

## Sequence refinement

Relevant modules:

- `sequence/contacts.py` — contact-event and sequence data structures;
- `sequence/candidates.py` — possible serve and later-contact frames;
- `sequence/edits.py` — allowed keep/add/replace/delete combinations;
- `sequence/features.py` — chooser inputs;
- `sequence/choices.py` — chooser thresholds and margins;
- `sequence/refine.py` — option-pool construction and the three chooser stages;
- `sequence/sides.py` — final alternating court-half assignment;
- `training/sequences.py` — target generation and grouped fitting.

Training and annotation use the same sequence-option builders. A new edit type, candidate rule or feature changes the examples used for fitting as well as the choices made at runtime, so it requires a new fit.

## Court and player evidence

Relevant modules:

- `courts/evidence.py` — court geometry used by annotation;
- `courts/scenes.py` — scene-specific court lookup;
- `rally/evidence.py` — sticky player assignment;
- `hybrid.py::sequence_inputs()` — raw court-half estimate for each candidate contact;
- `models.py::SideGeometry` — video-level versus scene-level net band.

A geometry change may affect more than winner/landing logic. Player-side assignment and wrist-to-shuttle features depend on the court/player evidence, so some court changes also change contact and sequence inputs.

## Exclusion masks and shuttle guards

Relevant modules:

- `masks/replay.py` — replay/off-rally mask;
- `masks/dead.py` — selects the configured mask method;
- `masks/inpaint.py` — shuttle recurrence/guard grading;
- `run_video.py::build_shuttle_hallucination_mask()` — converts guard codes to a boolean rejection mask;
- `dataset_builder/shuttle_evidence.py` — saved shuttle guard files and validation.

The dataset-builder full annotation path currently requires `DeadMaskMode.REPLAY`.

Changing which frames are excluded before feature construction changes the evidence seen by the fitted stages and generally requires a new fit.

The shuttle guard mask is narrower than the frame exclusion mask. It feeds the slow-motion speed estimate and the outcome rules. It reaches contact candidates only through the optional rule above. [Fixed heuristics](heuristics.md#shuttle-guard-grades) has the detail.

## Outcome rules

Relevant modules:

- `outcomes/point_winner.py` — server, landing and winner logic;
- `outcomes/video.py` — assembly of contact, landing, verdict and hit-height outputs.

Changes confined to outcome rules after the final contacts are fixed do not require the contact or sequence models to be retrained. They still need outcome-specific tests or evaluation because the standard refit report measures contacts and rally assembly, not winner or landing accuracy.

## Model loading and compatibility

`models.py` checks:

- both model files exist;
- schema matches;
- scikit-learn version matches exactly;
- contact feature names and order match;
- the serialised object has the expected `AnnotatorModels` type;
- side geometry agrees between `models.joblib` and `metadata.json`.

These checks answer a narrow question: whether the model files are structurally compatible with the current runtime. They do not measure prediction quality on current vision inputs.

## FPS scaling

Frame-count settings are defined against a 30 FPS reference and resolved once for each video.

- `fps_constants.py` holds the base values and scaling rules.
- `resolve.py` produces the final per-video values.
- lower-level code receives those final values; scaling them again would apply the FPS conversion twice.

This keeps a timing window such as six frames at 30 FPS approximately constant in seconds across different source frame rates.

## Training and annotation share code

The training path calls the same builders used during annotation:

- contact features come from `contacts/features.py`;
- `hybrid.sequence_inputs()` builds contact scores and raw court-half information;
- `sequence/refine.py` builds the option pool;
- `finish_sequences()` applies the same chooser and finalisation code used during annotation.

This sharing matters because a separate training-only copy of these rules could silently drift away from runtime behaviour.

## Tests by subsystem

### Model loading and training

```bash
PYTHONPATH=src uv run pytest \
  tests/test_annotator_models.py \
  tests/test_annotator_contact_training.py \
  tests/test_annotator_sequence_training.py \
  tests/test_annotator_confidence_training.py \
  tests/test_annotator_training_workflow.py
```

### Contact features and scoring

```bash
PYTHONPATH=src uv run pytest \
  tests/test_annotator_contact_features.py \
  tests/test_annotator_contact_model.py \
  tests/test_annotator_hybrid.py
```

### Sequence refinement

```bash
PYTHONPATH=src uv run pytest \
  tests/test_annotator_sequence_options.py \
  tests/test_annotator_sequence_choices.py \
  tests/test_annotator_sequence_refine.py \
  tests/test_annotator_sequence_finish.py
```

### End-to-end annotator path

```bash
PYTHONPATH=src uv run pytest \
  tests/test_annotator_run_video.py \
  tests/test_dataset_builder_vision.py \
  tests/test_dataset_builder_cli.py
```

The wider `tests/test_annotator_*.py` set covers the older and more specialised paths as well.

## Debugging order for wrong contacts

When a predicted contact looks wrong, the useful intermediate evidence is usually:

1. `definitive_exclusion_mask` — whether the frame was removed before scoring;
2. shuttle guard codes and `shuttle_quality.json.gz` — whether the shuttle position was graded unreliable and, when the model directory turns on the optional rule, whether the candidate was dropped for having no picked player;
3. court-scene coverage — whether usable geometry existed;
4. contact search regions — whether the frame was ever presented to the contact tree;
5. contact probability and cutoff — whether the tree selected it;
6. sequence option scores — whether sequence refinement removed, added or replaced it.

This sequence follows the same order as the data flow, so it separates “never considered” from “considered and rejected” without jumping straight to the final result.

## Files outside `src/annotator`

A few files outside `src/annotator` are also relevant:

- `src/dataset_builder/vision.py` — saved-input loaders and annotation saving;
- `src/dataset_builder/shuttle_evidence.py` — shuttle guard files;
- `configs/dataset_builder/*.toml` — vision-model paths and extraction settings;
- `tests/test_annotator_*.py` — current behaviour and edge-case coverage;
- [`experiments/annotator/`](../../experiments/annotator/README.md) — development history, measurements and experiment runners.

The [code map](code_map.md) gives a more complete per-module listing.
