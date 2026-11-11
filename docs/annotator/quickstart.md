# Run the auto-annotator

The auto-annotator turns saved shuttle, pose and court data into rally spans,
contact frames (racket hits), player-side assignments and outcome estimates.
Rally spans are the start and end frames of the predicted clips. The standalone
command and dataset builder use the same annotation code. This guide covers
running it; [How the annotator works](how_it_works.md) explains the pipeline.

There are two common cases:

- **Metadata, shuttle, pose and court outputs already exist** — `python -m annotator` runs annotation for one video.
- **Only source footage exists** — the dataset builder runs the vision stages first and then calls the same annotation code.

The default bundle in [`runtime/deployed/annotator`](../../runtime/deployed/annotator/)
contains four XGBoost classifiers: contact scoring, first rally selection,
correction of split pieces, and final review scoring. It uses standing-person
counts and splits first-pass rallies at hit gaps longer than three seconds.
Only split pieces go through the second rally selector. See
[the model settings](tree_stack.md) for feature sizes and tree settings.

**Contents**

| Preparation | Running annotation | Settings and help |
| --- | --- | --- |
| [Environment](#1-environment) | [One-video command](#3-one-video-command) | [Prediction settings](#6-settings-that-can-change-the-result) |
| [Saved input layout](#2-saved-input-layout) | [Output files](#4-output-files) | [Common failures](#7-common-failures) |
|  | [Dataset-builder run](#5-dataset-builder-run) | [Further detail](#8-further-detail) |

## 1. Environment

From the repository root, this installs the dependencies pinned in `uv.lock`:

```bash
uv sync
```

The project needs Python 3.12 or newer and uses the versions pinned in `uv.lock`. The annotator dependencies are part of the base project environment. No experiment extra is needed for a normal annotation run.

For GPU fitting or annotation on a V100, also complete the [V100 setup](retuning.md#v100-setup).

The commands on this page use `uv run`. Inside an already-activated project environment, plain `python` works the same way.

Learned annotation needs a fitted model directory containing:

```text
models.joblib
metadata.json
```

The two files form one model bundle. Before annotation starts, the loader
checks the model format, feature order and rally-proposal settings. Library
versions are recorded for reference.

The model needs shuttle, pose and court data consistent with the data used
during training. A substantial detector change can require a new fit
and evaluation. The [refit guide](retuning.md) explains how to create that model.

## 2. Saved input layout

For video ID `match-name`, `--run-dir` has this structure:

```text
data/dataset-run/
└── stages/
    ├── metadata/
    │   └── match-name/
    │       └── video_metadata.json.gz
    ├── shuttle/
    │   └── match-name/
    │       ├── shuttle_track.npy.xz
    │       ├── shuttle_guard_codes.npy.xz
    │       └── *_inpaint_mask.json.gz
    ├── pose/
    │   └── match-name/
    │       ├── pose_kps.npy.xz
    │       ├── pose_bboxes.npy.xz
    │       ├── pose_scores.npy.xz
    │       ├── pose_kp_scores.npy.xz
    │       └── pose_ndet.npy.xz
    └── court/
        └── match-name/
            ├── court_evidence.json.gz
            └── court_present.npy.xz
```

The source video is not reopened during this run. The command reads the saved stage files.

The shuttle guard codes are recomputed from the saved shuttle track and compared with the stored codes. A mismatch means the shuttle files came from different extraction states, so annotation stops rather than mixing them.

[Inputs and outputs](inputs_outputs.md) lists the exact array shapes and field meanings.

## 3. One-video command

```bash
PYTHONPATH=src uv run python -m annotator \
  --run-dir data/dataset-run \
  --video-id match-name \
  --models runtime/deployed/annotator \
  --output-dir data/annotations/match-name
```

`match-name` is the video's directory name inside each stage of the extraction run.

A successful run prints the number of rallies and the path to the saved result.

The output directory must be new or empty. This prevents a standalone comparison run from overwriting an existing result by accident.

The command runs the same annotation function as the dataset builder, but it does not update an existing dataset run's annotations or manifest. It needs neither the source video nor labels, and it does not run scraping, vision extraction or model fitting.

## 4. Output files

The output directory contains:

```text
annotator_result.json.gz
raw_replay_mask.npy.xz
definitive_exclusion_mask.npy.xz
shuttle_quality.json.gz
```

`annotator_result.json.gz` contains the annotation itself: rally spans, final contacts, court-half assignments, review scores and outcome estimates.

### Main result fields

- Rally spans include the start frame and exclude the end frame: `[start, end)`.
- `rallies[rally_id].sequence.events` contains the final contacts belonging to a rally.
- `contact_events` contains the full final contact stream, including contacts that fall outside all rally spans.
- `Top` and `Bot` refer to physical halves of the visible court: `Top` is the far half and `Bot` is the near half. They are not player names; the labels stay with the court halves when players change ends.
- `rally_confidence` is a score for ordering rallies during review. It does not change the annotation or approve a rally automatically.
- Server, winner, landing and hit-height fields are estimates made after the final contacts are known. They can be empty when the evidence is not good enough.

Use the events in each finished rally for contact membership. Filtering `contact_events` against the rally spans is not a substitute.

## 5. Dataset-builder run

When the dataset builder runs the whole pipeline, its config names the annotator model directory:

```toml
[models]
annotator = "runtime/deployed/annotator"
```

The key sits in the `[models]` table beside the vision models. A relative path resolves from the repository root, and the default is the tracked bundle above.

The supplied trial config already contains this setting. A typical run is:

```bash
PYTHONPATH=src uv run python -m dataset_builder run \
  --config configs/dataset_builder/trial.toml \
  --run-dir data/dataset-run
```

The builder runs the vision stages and then calls the same `run_full_annotation_stage()` used by the standalone command. The trial config also performs acquisition and vision extraction; it does not fit annotator trees.

The builder checks the model files before annotation starts. When resuming a
run, it checks both files again. Replacing either means the old annotation
results must be regenerated with the new model.

## 6. Settings that can change the result

The command-line arguments select the saved extraction, video, model and output directory.
Settings that affect predictions are stored with the model: changing them
usually requires a new fit. The table below identifies those settings for
readers preparing a different model; the [maintainer guide](maintaining.md)
explains which changes require refitting.

| Setting | Current default | Where it lives | What it changes |
| --- | ---: | --- | --- |
| Contact score cutoff | `0.9` | `ContactModelConfig` in `contacts/score_candidates.py` | Cutoff for the initial contact stream; lower-scoring candidates can still be reconsidered by sequence repair |
| Rally/contact rules | `BaseAnnotatorConfig` | saved in the model directory | Rough rallies, masks and contact search regions seen by the fitted stages |
| Shuttle/pose/court extraction | dataset-builder config | `configs/dataset_builder/*.toml` | Upstream evidence supplied to annotation |

Player attribution always uses the current scene's court geometry. Reusing one
homography for every scene gives video-wide geometry through the same path.

The [refit guide](retuning.md#python-fit-settings) lists tree-training parameters and defaults.

Separate model and output directories make side-by-side comparisons easy to trace.

## Heuristic-only annotation

Use `--heuristic-only` to run the benchmark mode from saved inputs:

```bash
PYTHONPATH=src uv run python -m annotator \
  --run-dir data/dataset-run \
  --video-id match-name \
  --heuristic-only \
  --output-dir data/annotations/match-name-heuristic
```

Choose one mode: `--models PATH` or `--heuristic-only`. The heuristic mode uses
current default settings and writes the same output files. For configured
heuristic benchmarks, use the [benchmark runner](../../experiments/annotator/monotrack_benchmark/benchmark_scripts/README.md).

The Python API accepts the prepared arrays and court/landing inputs:

```python
from annotator.config import BaseAnnotatorConfig
from annotator.annotate_video import annotate_video

annotation = annotate_video(
    track, bboxes, scores, kps, ndet,
    fps=fps,
    heuristic_only=True,
    base=BaseAnnotatorConfig(),
    **court_and_mask_inputs,
)
```

`annotation.result` contains heuristic contacts, player assignments and outcome
estimates. The exclusion masks are returned alongside it.
Change `base` for a heuristic experiment and save its configuration with the
result. For feature inputs, call `prepare_annotation_inputs()` directly. See the
[input guide](inputs_outputs.md#direct-python-call) for the required arrays.

## 7. Common failures

**“Annotator model bundle is incomplete”**
The `--models` directory is missing `models.joblib`, `metadata.json`, or both.

**“persisted shuttle guard codes differ from the final track”**
The saved shuttle track and guard codes disagree. The shuttle stage needs to
be regenerated as a set.

**Court evidence has no usable operational inputs**
The court stage contains an incompatible or incomplete court result. Annotation needs the current `court-evidence/0.6` court inputs.

**Output directory must be new or empty**
The standalone runner found existing files at the chosen output path.

## 8. Further detail

`python -m annotator` runs the full annotation chain and supports heuristic-only
inference with `--heuristic-only`.

| Task | Document |
|---|---|
| Understand the full pipeline | [How the annotator works](how_it_works.md) |
| Inspect the fixed rules | [Fixed heuristics](heuristics.md) |
| Understand the fitted classifiers | [Tree model stack](tree_stack.md) |
| Check saved inputs and output fields | [Inputs and outputs](inputs_outputs.md) |
| Fit a new model | [Retuning guide](retuning.md) |
| Reproduce the completed comparison | [Current experiment commands](../../experiments/annotator/tree_simplification/README.md) |
| Check what has already been tried | [Development history](../../experiments/annotator/development.md) |
| Find source files or assess a code change | [Code map](code_map.md) and [maintainer guide](maintaining.md) |

The standalone commentary-pairing tool requires an explicit `--rally-spans` CSV.
The dataset builder computes replay exclusion in memory. The standalone
replay-mask writer and track-only batch segmentation command have been retired.
