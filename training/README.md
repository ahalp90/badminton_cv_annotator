# `training/`

Source training data, per-architecture caches, and experiment records.
Required to train a model. Pipeline inference uses selected weights from the
runtime or experiment trees.

```
training/
├── data/                       source training data (shared)
└── <arch>/
    ├── cache/                  re-derivable preprocessing per architecture
    └── experiments/<run_id>/   training run record + weights
```

## `data/`

Source training data. The ShuttleSet release lives under
`data/shuttleset/`:

```
training/data/shuttleset/
├── annotations/    upstream CSVs (per-match, video metadata, flaw records)
├── raw_video/      source mp4s
├── rally_clips/    per-rally mp4s
└── clips/          per-stroke mp4s
```

Annotations are checked in at the directory level (`.gitkeep`); video
files are gitignored and rsync'd onto the training host. See
`src/bric/preprocessing/slice_rallies.py` for how `rally_clips/` is produced.

## Per-architecture

Each architecture has its own training pipeline. Current state:

- **BRIC** uses this tree — see [BRIC layout](#bric-layout) below.
- **BST-X** organises its training data and experiments under
  `src/bst_x/`; see that subproject's own documentation.

### BRIC layout

`training/bric/cache/` — per-architecture preprocessing caches.
Re-derivable from `data/` and BRIC's preprocessing scripts; not
checked in.

```
training/bric/cache/
├── players/      per-source-video striker bbox tracks (npz, wide format)
├── shuttle/      per-source-video TrackNetV3 trajectory (npz)
└── rgb/          per-stroke 32-frame striker crop tensor (npy)
```

Caches are content-addressable by the source video they derive from;
re-running the producing script (`src/bric/preprocessing/preprocess_videos.py`,
`src/bric/preprocessing/extract_shuttle.py`) on the same input is idempotent.

`training/bric/experiments/<run_id>/` — one directory per training
run, written by `python -m bric.train`. Contains everything needed to
identify the run, reproduce its evaluation, and deploy its model:

```
training/bric/experiments/<run_id>/
├── manifest.yaml    architecture, taxonomy, variant, classes,
                     hparams, seed, git SHA
├── metrics.csv      per-epoch train/val loss, val macro F1, val acc, lr
└── best.pt          best-on-val-macro-F1 model weights
```

## Recording a classifier model variant

A completed run can be retained for classifier comparison or selected by a
future in-process annotator classification stage.

For architectures that use this tree's `experiments/<run_id>/`
convention (currently BRIC):

1. Train a new run → produces `training/<arch>/experiments/<run_id>/`
   with `manifest.yaml`, `best.pt`, `metrics.csv`.
2. Symlink or rsync the run into `runtime/deployed/<arch>/` for shared use —
   see [`runtime/README.md`](../runtime/README.md).

These steps retain the model and its provenance. They do not activate a web
service. An annotator classifier stage should load the selected model directly
inside the pipeline process.

## Adding a new architecture

A new architecture is a project, not a drop-in. It requires:

- A new `src/<arch>/` package (dataset, network, train, infer, eval)
  designed for its own input shape and training loop
- An in-process pipeline adapter if the annotator should call it
- Its own preprocessing scripts and any perception infra it needs
- Optionally adopting the `training/<arch>/` and
  `runtime/deployed/<arch>/` conventions for shared model storage

The conventions in this tree (experiment manifest schema, cache
layout per arch, deployment symlink/rsync workflow) are **opt-in**:
an architecture that adopts them inherits the hot-deployable variant
workflow. An architecture that doesn't manages its own conventions.

See `src/bric/` for one implementation that uses this tree;
`src/bst_x/` for one that maintains its own.

## Shared conventions

For architectures that opt into this tree:

- **Experiment manifest schema** — each
  `<arch>/experiments/<run_id>/manifest.yaml` should declare
  `architecture`, `taxonomy`, `variant`, `classes`, hyperparameters,
  seed, and git SHA. Lets `runtime/deployed/<arch>/` slots point at
  any run uniformly and supports direct pipeline loading.
- **Cache idempotency** — caches should be content-addressable on
  their source so re-running producing scripts is safe.
- **`.gitignore` pattern** — `<dir>/*` plus `!<dir>/.gitkeep` keeps
  the structure in git without committing data.
