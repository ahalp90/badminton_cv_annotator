# Refit guide

`python -m annotator.training` fits the current XGBoost contact, rally and review
models from a saved extraction run. The manifest controls which videos may be
used for fitting. Evaluation is a separate command that reads a saved bundle.

Keep each experiment's configuration, model directory and evaluation report
together. The [tree guide](tree_stack.md) explains what the models learn;
the [maintainer guide](maintaining.md) explains which code changes require a refit.

## Saved inputs and labels

Training reads the same metadata, shuttle, pose and court artefacts as annotation:

```text
<run-dir>/stages/
  metadata/<video-id>/...
  shuttle/<video-id>/...
  pose/<video-id>/...
  court/<video-id>/...
```

The extraction must include shuttle guard codes and usable court geometry.
Training and inference should use comparable extraction settings. A material
change to the court, shuttle or pose detector needs a fresh comparison.

Each video's contact CSV has exactly these columns:

```csv
rally_id,frame,side
rally-1,100,Top
rally-1,130,Bot
rally-1,160,Top
rally-2,300,
rally-2,330,Top
```

Use zero-based source-video frames. `Top` and `Bot` identify the far and near
court halves; a blank side means unknown. A rally's rows must be contiguous and
its frames strictly increasing. Included videos need complete contact labels:
an unlabelled real hit can otherwise become a negative training example.

## Manifest and groups

The manifest names an extraction run and disjoint train, validation and test
splits. Paths resolve relative to the manifest file. Each video ID must match
its directory in the extraction run.

```json
{
  "run_dir": "../data/extracted-run",
  "videos": [
    {"id": "train-a", "group": "match-a", "split": "train", "labels": "labels/train-a.csv"},
    {"id": "train-b", "group": "match-b", "split": "train", "labels": "labels/train-b.csv"},
    {"id": "train-c", "group": "match-c", "split": "train", "labels": "labels/train-c.csv"},
    {"id": "validation-a", "group": "match-d", "split": "validation", "labels": "labels/validation-a.csv"},
    {"id": "test-a", "group": "match-e", "split": "test", "labels": "labels/test-a.csv"}
  ]
}
```

Keep related footage in the same group. A group may contain several videos but
must belong to only one split. Downstream training holds groups out when making
predictions for training examples. Each fitted subset must contain both target
classes. More selector layers require more training groups.

Use validation to choose settings, then test the chosen model separately.
Repeated decisions based on test errors turn that test set into development data.
The existing ShuttleSet22 benchmark has informed development and is now familiar
footage.

## Fit commands

The default is one rally selector with the selected XGBoost settings:

```bash
PYTHONPATH=src uv run python -m annotator.training fit \
  --manifest retune/manifest.json \
  --output models/experiment-a \
  --side-geometry video
```

To fit on a CUDA GPU, add `--device cuda`. Saved models use CPU inference so they
can be loaded on machines without a GPU. `--threads` controls fitting thread
limits. `--rally-layers 2` or `3` requests additional selector layers.

`--side-geometry video` uses one net band per video for player-half guesses.
`--side-geometry scene` uses each scene's geometry. This setting is saved in the
bundle because it changes model evidence.

The command reads only training videos and their labels. It fits the final
contact model, held-group contact models, the rally selectors and the review
model. The saved bundle contains the models used for inference; temporary
held-group fits supply downstream training examples.

## Python fit settings

The Python API exposes the settings used by the command. Use dataclass copies to
change a particular experiment while retaining the other defaults:

```python
from dataclasses import replace
from pathlib import Path

from annotator.training.workflow import (
    TrainingSettings,
    fit_from_manifest,
    load_manifest,
)

settings = TrainingSettings()
rally = replace(
    settings.rally_layers,
    tree=replace(settings.rally_layers.tree, max_leaf_nodes=15, max_iter=200),
)
settings = replace(settings, rally_layers=rally)

fit_from_manifest(
    load_manifest(Path("retune/manifest.json")),
    Path("models/smaller-selector"),
    settings=settings,
)
```

The settings separate contact fitting, contact selection, rally fitting, review
fitting and heuristic preprocessing. Tree controls include learning rate, rounds,
maximum leaves, minimum child weight, regularisation, seed and CPU/CUDA device.
Rally controls also include depth and the proposal budgets.
`settings.contact_prediction` holds the contact probability cutoff and the optional
`reject_masked_without_player` rule. Change it with `replace()` in the same way;
the resulting setting is saved with the fitted bundle.

For a five-seed comparison, keep the data, settings and order fixed and vary the
seeds deliberately. The current benchmark uses offsets 0, 100, 200, 300 and 400
on the contact, selector and review base seeds. These are five seeded runs,
not five-fold cross-validation.

## Configuration and saved results

`models.joblib` and `metadata.json` form one bundle. They retain preprocessing,
contact selection, side geometry, proposal settings and fitted estimators.
The metadata records runtime versions and feature order. Keep both files
when moving a model.

Record the manifest and full fit settings beside the model. Record input versions,
seeds, sampling order and fit-row order when comparing runs. A seed alone does
not reproduce a fit when rows or features change order.

Heuristic experiments remain available through `run_video(...,
heuristic_only=True, base=config)` and the existing
[`heuristic_only_tune`](../../experiments/annotator/heuristic_only_tune/) tools.
Use separate result directories for each configuration. The same preprocessing
can be saved with a newly fitted model when moving from a rule experiment to a
learned comparison.

## Evaluate a saved model

```bash
PYTHONPATH=src uv run python -m annotator.training evaluate \
  --manifest retune/manifest.json \
  --models models/experiment-a \
  --split validation \
  --output retune/experiment-a-validation.json.gz
```

Evaluation reads only the requested validation or test split. It fits nothing
and uses the bundle's preprocessing. The report contains per-video counts and
totals; its path must end in `.json.gz`.

Contact precision and recall use one-to-one matching within ten frames at 30 FPS,
scaled to the video's frame rate. A predicted rally is correct when it contains
one complete labelled rally, has no extra contacts and agrees with known sides.
Missing contacts, merged rallies and known side contradictions count as wrong.
Otherwise, missing side labels can make a rally unjudgeable.

The report counts predicted rally sections. An entirely missed rally creates no
predicted section, so inspect contact recall and rally recovery alongside the
fraction of judged sections that are correct. Contact labels do not validate
winner, landing or hit-height estimates.

## Reproduce the current benchmark

The current comparison retains fixed splits, explicit contact sampling order,
saved preparation, proposal targets, fit settings and per-video streams under
[`experiments/annotator/tree_simplification`](../../experiments/annotator/tree_simplification/README.md).
Its [report](../../experiments/annotator/reports/tree_simplification.md) identifies
the selected raw-preprocessing baseline. Follow that experiment's commands for
an exact reproduction; a new generic manifest run is a separate fit.

Older eight-model results remain historical records. Their model bundles and
training commands are no longer supported by the current annotator. Use the
corresponding repository revision to reproduce that earlier architecture.
