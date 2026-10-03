# Run the auto-annotator

The auto-annotator turns saved shuttle, pose and court data into rally spans,
contact frames, player-side assignments and outcome estimates. The standalone
command and dataset builder use the same annotation code. This guide covers
running it; [How the annotator works](how_it_works.md) explains the pipeline.

There are two common cases:

- **Metadata, shuttle, pose and court outputs already exist** — `python -m annotator` runs annotation for one video.
- **Only source footage exists** — the dataset builder runs the vision stages first and then calls the same annotation code.

The selected model is **base, recorded on 3 October 2026**, fitted with
**scikit-learn 1.9.1**.
Its checked `models.joblib` and `metadata.json` files are included in
[`data/annotator/sset_and_sset22_trained_20261003T041112Z`](../../data/annotator/sset_and_sset22_trained_20261003T041112Z/).
The commands below expect both files in `models/annotator`; `--models` also
accepts the repository directory directly. The [evaluation report](../../experiments/annotator/reports/model_selection.md)
explains why base was selected and which errors remain.

## 1. Environment

From the repository root, the locked project environment is installed with:

```bash
uv sync
```

The project needs Python 3.12 or newer and uses the versions pinned in `uv.lock`. The annotator dependencies are part of the base project environment. No experiment extra is needed for a normal annotation run.

The commands on this page use `uv run`. Inside an already-activated project environment, plain `python` works the same way.

Annotation also needs a fitted model directory containing:

```text
models.joblib
metadata.json
```

The two files form one model bundle. The loader checks their schema, contact
feature order and exact scikit-learn version before annotation starts.

The model needs training inputs consistent with the run's shuttle, pose and
court data. A substantial detector change can require a new fit
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
            ├── court_keep_vote.npy.xz
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
  --models models/annotator \
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

- Rally spans use half-open bounds: `[start, end)`.
- `filtered_by_rally` lists the final contact frames belonging to each rally.
- `contact_events` contains the full final contact stream, including contacts that fall outside all rally spans.
- `Top` and `Bot` refer to physical halves of the visible court: `Top` is the far half and `Bot` is the near half. They are not player names; the labels stay with the court halves when players change ends.
- `rally_confidence` is a score for ordering rallies during review. It does not change the annotation or approve a rally automatically.
- Server, winner, landing and hit-height fields are estimates made after the final contacts are known. They can be empty when the evidence is not good enough.

`filtered_by_rally` is the field to use for contact membership. Filtering `contact_events` against the rally spans is not a substitute.

## 5. Dataset-builder run

When the dataset builder runs the whole pipeline, its config names the annotator model directory:

```toml
[models]
annotator = "models/annotator"
```

The key sits in the `[models]` table beside the vision models. A relative path resolves from the repository root, and the default is `models/annotator`.

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

The CLI selects the saved extraction, video, model and output directory.
Settings that affect predictions are stored with the model: changing them
usually requires a new fit. The table below identifies those settings for
readers preparing a different model; the [maintainer guide](maintaining.md)
explains which changes require refitting.

| Setting | Current default | Where it lives | What it changes |
| --- | ---: | --- | --- |
| Contact score cutoff | `0.9` | `ContactModelConfig` in `contacts/model.py` | Cutoff for the initial contact stream; lower-scoring candidates can still be reconsidered by sequence repair |
| Rule for guarded candidates | off | `ContactModelConfig.reject_masked_without_player` | When on, drops candidate frames that have a rejected shuttle guard grade and no picked player nearby in time; see [Fixed heuristics](heuristics.md#optional-rule-for-guarded-candidates) |
| Contact-tree training | 180 max iterations, 31 leaves, LR 0.06 | `ContactFitConfig` | Shape of the fitted contact classifier |
| Side geometry | `video` | model directory / `--side-geometry` during fitting | Whether player halves use one net band per video or per-scene geometry |
| Rally/contact rules | `BaseAnnotatorConfig` | saved in the model directory | Rough rallies, masks and contact search regions seen by the fitted stages |
| Shuttle/pose/court extraction | dataset-builder config | `configs/dataset_builder/*.toml` | Upstream evidence supplied to annotation |

Separate model and output directories make side-by-side comparisons easy to trace.

## 7. Common failures

**“Annotator model bundle is incomplete”**  
The `--models` directory is missing `models.joblib`, `metadata.json`, or both.

**“models were fitted with scikit-learn X; this runtime has Y”**  
The loader requires an exact scikit-learn match. The model directory needs to be fitted with the current runtime.

**“persisted shuttle guard codes differ from the final track”**  
The shuttle files do not belong to one consistent extraction state. The shuttle stage needs to be regenerated as a set.

**Court evidence has no usable operational inputs**  
The court stage contains an incompatible or incomplete court result. Annotation needs the current `court-evidence/0.2` court inputs.

**Output directory must be new or empty**  
The standalone runner found existing files at the chosen output path.

## 8. Further detail

`python -m annotator.rally_segmentation` is the older, rule-only segmentation
command. `python -m annotator` runs the full fitted annotation chain.

| Task | Document |
|---|---|
| Understand the full pipeline | [How the annotator works](how_it_works.md) |
| Inspect the fixed rules | [Fixed heuristics](heuristics.md) |
| Understand the fitted classifiers | [Tree model stack](tree_stack.md) |
| Check saved inputs and output fields | [Inputs and outputs](inputs_outputs.md) |
| Fit a new model | [Retuning guide](retuning.md) |
| Reproduce the completed comparison | [Annotator reproduction guide](../../experiments/annotator/reports/model_refit.md) |
| Check what has already been tried | [Development history](../../experiments/annotator/development.md) |
| Find source files or assess a code change | [Code map](code_map.md) and [maintainer guide](maintaining.md) |
