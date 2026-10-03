# Code map

This page maps the auto-annotator from its command-line entry point to the
modules that produce an annotation. The [symptom table](#symptom--likely-starting-point)
maps common problems to relevant modules. [Fixed heuristics](heuristics.md) and
[Tree model stack](tree_stack.md) explain the algorithms.

The annotation runtime is only part of `src/annotator`. Evaluation tools, manual review utilities and VLM experiments live in the same package but run separately.

## Production call map

```text
python -m annotator
    │
    ▼
annotator/__main__.py
    │
    ▼
annotator/cli.py
    │  load one video's saved builder stages
    ▼
dataset_builder/vision.py
    │  run_full_annotation_stage()
    ▼
annotator/run_video.py
    │
    ├── config.py + fps_constants.py + resolve.py
    │       resolve per-video settings
    │
    ├── courts/ + rally/ + masks/
    │       build court/player data, exclusion mask,
    │       rough rallies and contact search regions
    │
    ├── hybrid.py
    │     ├── contacts/
    │     │      features → contact probabilities → initial events
    │     └── sequence/
    │            limited repair options → three chooser stages
    │            → rally bounds → alternating court halves
    │
    └── outcomes/
           server / winner / landing / hit height
    │
    ▼
dataset_builder/vision.py
       save annotation result, masks and shuttle-quality summary
```

The fitting path reuses the same feature and sequence-option builders:

```text
python -m annotator.training
    ▼
training/workflow.py
    ├── load saved extraction data
    ├── run rule-based preprocessing and capture ContactEvidence
    ├── contacts/features.py
    ├── training/contact.py
    ├── hybrid.sequence_inputs()
    ├── sequence/refine.py option builders
    ├── training/sequences.py
    └── training/confidence.py
    ▼
models.py → models.joblib + metadata.json
```

## Main orientation files

Seven files give a compact view of the maintained architecture:

1. **`cli.py`** — command-line inputs and saved-stage loading.
2. **`dataset_builder/vision.py::run_full_annotation_stage`** — where saved vision outputs become annotator inputs and where annotation outputs are persisted.
3. **`run_video.py`** — complete one-video flow.
4. **`hybrid.py`** — learned contact and sequence stages without the later outcome code.
5. **`sequence/refine.py`** — sequence alternatives and the three chooser stages.
6. **`training/workflow.py`** — fitting and held-out evaluation.
7. **`models.py`** — model directory contents and compatibility checks.

## Package shape

```text
src/annotator/
├── core orchestration and shared data types
├── contacts/             contact features and contact classifier
├── courts/               court data used by annotation
├── masks/                replay/bad-frame masks and shuttle guards
├── rally/                rough rallies and heuristic contact signals
├── sequence/             learned contact-sequence repair
├── outcomes/             winner, landing and hit height
├── training/             model fitting and held-out evaluation
├── evaluation/           fixed/historical benchmarks and scoring helpers
├── review/               manual truth/review tools
├── validation_overlay/   visual inspection renderer
└── vlm_scene_benchmark/  separate VLM scene research
```

`__init__.py` files are omitted below unless they contain meaningful behaviour.

## Core orchestration and shared data

| Module | Role |
| --- | --- |
| `__main__.py` | Entry point for `python -m annotator`; calls `cli.main()`. |
| `cli.py` | Loads one video's saved metadata, shuttle, pose and court files; checks shuttle guards; loads the model directory; calls the dataset-builder annotation stage. |
| `run_video.py` | Coordinates the full one-video run: rough rallies and masks, fitted contact/sequence models, then outcome fields. |
| `hybrid.py` | Bridge from `ContactEvidence` to contact scoring, sequence refinement and rally review scores. |
| `models.py` | `AnnotatorModels`, `SideGeometry`, model saving/loading and compatibility checks. |
| `config.py` | Settings for rough rallies, masks and other rule-based steps that feed the fitted models. Also contains a few older shared output paths. |
| `fps_constants.py` | 30 FPS reference constants and scaling rules. |
| `resolve.py` | Produces final per-video settings for a given FPS. |
| `types.py` | Shared enums/data structures, shuttle speed helpers and sticky-player array conventions. |
| `shuttle_track.py` | Validation for the `(t, 3)` normalised shuttle track. |
| `video_metadata.py` | Constant-frame-rate metadata and serialisation helpers. |
| `artifact_io.py` | Deterministic compressed JSON/NumPy/text I/O used by annotator experiments and support records. |
| `doubles_flag.py` | Converts per-frame player over-count into a span/video doubles flag. This is adjacent to the contact pipeline. |
| `rally_segmentation.py` | Wrapper that keeps older rally-segmentation imports working while most implementation now lives under `rally/`. |

## `contacts/`: contact scoring

| Module | Role |
| --- | --- |
| `contacts/features.py` | Builds search regions and the 85-value contact feature vector for each searched frame. |
| `contacts/model.py` | Defines feature order, score cutoff, the optional rule for guarded candidates and nearby-contact suppression; runs the contact classifier. |

`CONTACT_FEATURE_NAMES` fixes the feature order used during fitting and annotation. The same order is recorded in model metadata.

## `courts/`: court data used by annotation

The court detector itself lives outside this package. These modules convert saved court results into the geometry the annotator consumes.

| Module | Role |
| --- | --- |
| `courts/evidence.py` | Builds the court data structure used by annotation. |
| `courts/scenes.py` | Stores scene-specific court geometry and looks it up by frame. |
| `courts/views.py` | Matches recurring static camera views so court calibration can be shared across similar views. |

## `masks/`: unusable frames and shuttle positions

| Module | Role |
| --- | --- |
| `masks/dead.py` | Selects the configured frame-exclusion strategy. |
| `masks/replay.py` | Builds replay/off-rally masks from court absence, slow motion and related signals. |
| `masks/composition.py` | Alternative segment-level mask based on broadcast cuts and court-view votes. |
| `masks/inpaint.py` | Grades repeated shuttle-track patterns that look fabricated or otherwise unsafe. |

The dataset-builder full annotation path currently uses `DeadMaskMode.REPLAY`.

## `rally/`: rough rallies and heuristic contact evidence

| Module | Role |
| --- | --- |
| `rally/spans.py` | Opens and closes rough rally spans from shuttle motion, rests and tracking gaps. |
| `rally/serve.py` | Serve-setup evidence and serve-start look-back rules. |
| `rally/contacts.py` | Heuristic contact impulses, wrist gates and suppression. |
| `rally/evidence.py` | Sticky player assignment and court/player measurements used by serve/contact rules. |
| `rally/trajectory.py` | Shuttle smoothing and transforms shared by rally and contact logic. |
| `rally/cli.py` | Older batch command for the rough rally-segmentation path. |

## `sequence/`: contact-sequence repair

Data moves through the sequence package roughly like this:

```text
contacts.py → candidates.py → edits.py → features.py → choices.py → refine.py
```

| Module | Role |
| --- | --- |
| `sequence/contacts.py` | `ContactEvent`, `ContactSequence` and scored-candidate data structures. |
| `sequence/candidates.py` | Shortlists earlier serve frames and possible missed later contacts. |
| `sequence/edits.py` | Builds the limited keep/add/replace/delete alternatives. |
| `sequence/features.py` | Creates inputs for the serve, sequence and insertion models. |
| `sequence/choices.py` | Applies chooser score rules and required margins. |
| `sequence/refine.py` | Builds the option pool, runs the three chooser stages, widens bounds and calls final side assignment. |
| `sequence/boundaries.py` | Expands rally bounds without changing which contacts belong to the rally. |
| `sequence/sides.py` | Chooses the alternating `Top`/`Bot` pattern for each final rally. |
| `sequence/confidence.py` | Builds features and scores the rally review model. |

## `outcomes/`: fields derived after contact selection

| Module | Role |
| --- | --- |
| `outcomes/point_winner.py` | Server, landing and winner rules plus contact-half attribution helpers. |
| `outcomes/video.py` | Builds final contact records, verdicts, landings, hit heights and related diagnostics. |

These modules run after the contact/sequence choice. Changes confined here usually do not require the contact or sequence models to be refitted.

## `training/`: fitting and held-out evaluation

| Module | Role |
| --- | --- |
| `training/__main__.py` | Entry point for `python -m annotator.training`. |
| `training/workflow.py` | Reads manifests/labels, loads saved extraction data, runs grouped fitting and writes validation/test reports. |
| `training/contact.py` | Selects labelled contact rows and fits the contact classifier. |
| `training/sequences.py` | Builds sequence targets and fits grouped sequence models. |
| `training/confidence.py` | Fits the rally review tree from out-of-group completed predictions. |

The training package calls the same feature and option builders used at runtime. This is what keeps fitted inputs consistent with annotation inputs.

## `evaluation/`: fixed and historical scoring tools

These modules support specific benchmark jobs. They are not called by a normal annotation run.

| Module | Role |
| --- | --- |
| `evaluation/batch_report.py` | Plain-text reports for the older rally-segmentation batch command. |
| `evaluation/scoring.py` | Rally/contact scoring helpers and ground-truth structures. |
| `evaluation/gt_scoring.py` | Committed ShuttleSet ground-truth scoring. |
| `evaluation/fixtures.py` | Digest-checked calibration fixtures and geometry records. |
| `evaluation/commentary_benchmark.py` | Scores prepared commentary against ShuttleSet rally populations. |
| `evaluation/commentary_benchmark_inputs.py` | Loads and validates the pinned inputs for that commentary benchmark. |
| `evaluation/shuttleset_benchmark.py` | Scores pinned rally records against ShuttleSet ground truth. |
| `evaluation/shuttleset_features.py` | Feature prototypes and re-exports used by older ShuttleSet work. |
| `evaluation/shuttleset22_features.py` | Older ShuttleSet22 prototype evaluator; its pinned court files predate the current court schema. |

New model fits normally use the evaluation report produced by `training/workflow.py`. The modules above remain relevant when reproducing the specific benchmark they implement.

## `review/`: manual labelling and audit tools

| Module | Role |
| --- | --- |
| `review/broadcast_editor.py` | OpenCV editor for broadcast-scene labels. |
| `review/broadcast_labels.py` | Data format and save/load helpers for broadcast-scene truth. |
| `review/rally_start_editor.py` | UI for reviewing rally-start events. |
| `review/rally_starts.py` | Rally-start truth records and review-session state. |

These tools create or inspect truth data; they do not run during annotation.

## `validation_overlay/`: visual checks

This package renders selected video spans with algorithm output drawn on top.

| Module | Role |
| --- | --- |
| `validation_overlay/core/cli.py` | Shared render command and frame composition. |
| `validation_overlay/core/decode.py` | Exact-frame ffmpeg decoding using saved metadata. |
| `validation_overlay/core/encode.py` | Long-lived ffmpeg encoder. |
| `validation_overlay/core/hud.py` | Scaled on-frame text and marks. |
| `validation_overlay/core/timeline.py` | Reads span CSVs and constructs the render frame plan. |
| `validation_overlay/overlays/shuttle_track.py` | Shuttle-track overlay. |

The separate `validation_overlay/README.md` and `DOCS.md` cover that tool in detail.

## `vlm_scene_benchmark/`: separate scene-classification research

This directory contains the Issue-38 VLM scene benchmark. It is not called by the contact/rally annotator.

| Module | Role |
| --- | --- |
| `vlm_scene_benchmark/contracts.py` | Saved benchmark record formats and reload checks. |
| `vlm_scene_benchmark/prepare.py` | Builds frame-mapped videos and cut manifests. |
| `vlm_scene_benchmark/prompts.py` | Frozen benchmark prompts. |
| `vlm_scene_benchmark/runtime.py` | Backend runtime records and response handling. |
| `vlm_scene_benchmark/run_cli.py` | Runs one benchmark backend. |
| `vlm_scene_benchmark/score_cli.py` | Scores a saved benchmark result. |
| `vlm_scene_benchmark/scoring.py` | Frame/boundary scoring. |
| `vlm_scene_benchmark/gate_cli.py` | Applies the deployment gate. |
| `vlm_scene_benchmark/backends/__init__.py` | Backend selection and shared backend interfaces. |
| `vlm_scene_benchmark/backends/internvideo3.py` | InternVideo3 adapter. |
| `vlm_scene_benchmark/backends/qwen3_vl.py` | Qwen3-VL adapter. |

## Files outside `src/annotator`

| Location | Role |
| --- | --- |
| `src/dataset_builder/vision.py` | Saved-input loaders, annotation integration and output serialisation. |
| `src/dataset_builder/shuttle_evidence.py` | Shuttle inpaint/guard filenames and validation. |
| `configs/dataset_builder/*.toml` | Model paths and upstream extraction settings. |
| `models/annotator/` | Usual location for the selected model directory. |
| `experiments/annotator/` | Retained comparisons and experiment records, including the old-court regression check and the completed new-court refit comparison. |
| `scratch/contact_det*` and related scratch directories | Historical development material and rejected/older variants. |
| `tests/test_annotator_*.py` | Current behaviour and edge cases in executable form. |

## Symptom → likely starting point

| Symptom | Relevant code |
| --- | --- |
| Rally opens or closes at the wrong time | `rally/spans.py`, `rally/serve.py` |
| Replay/dead frames leak into a rally | `masks/replay.py`, `masks/dead.py`, `run_video.py` |
| FPS-dependent timing looks wrong | `fps_constants.py`, `resolve.py` |
| Player assignment jumps | `rally/evidence.py` and its court/pose inputs |
| A real hit is never scored | `contacts/features.py` search regions, then exclusion masks |
| Contact probability looks wrong | `contacts/features.py`, `contacts/model.py` |
| Serve repair looks wrong | `sequence/candidates.py`, `sequence/edits.py`, serve features/targets |
| Later-hit repair looks wrong | `sequence/candidates.py`, `sequence/refine.py`, `training/sequences.py` |
| `Top`/`Bot` assignment looks wrong | `hybrid.py::sequence_inputs`, `sequence/sides.py`, side geometry |
| Rally review score looks wrong | `sequence/confidence.py`, `training/confidence.py` |
| Winner or landing looks wrong | `outcomes/point_winner.py`, `outcomes/video.py` |
| Model directory will not load | `models.py` |
| Fitting appears to leak match groups | `training/workflow.py`, `training/sequences.py` |
| Saved standalone output differs from builder output | `annotator/cli.py`, `dataset_builder/vision.py` |

Historical experiment code can still answer questions about an earlier design or measurement. For current annotation behaviour, the modules above are the direct implementation path.
