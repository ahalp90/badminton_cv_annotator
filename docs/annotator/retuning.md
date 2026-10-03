# Refit guide

A refit creates a new `models.joblib` and `metadata.json` from labelled videos, then measures that model directory on held-out validation or test videos. `models.joblib` is the Joblib-serialised fitted model bundle.

The fitted pieces are trained together because the later sequence models depend on the contact stream produced earlier in the chain.

![Refit flow](figures/refit_flow.svg)

The commands on this page are the stable, general way to fit and evaluate. The refit planned for the new court inputs uses its own temporary runner; [Planned new-court refit](#planned-new-court-refit) at the end of this page covers it.

## When a new fit is needed

A new fit is normally needed when a change alters the frames, features or sequence alternatives seen by a fitted tree.

Common examples include:

- a new or materially changed court detector;
- a shuttle or pose change that materially changes the values seen by the models;
- changes to contact feature values, units, windows or order;
- changes to rough-rally or contact-search rules;
- a different contact score cutoff, or turning the optional rule for guarded candidates on or off;
- changes to serve or later-contact alternatives;
- changes to sequence features;
- changing between `video` and `scene` side geometry;
- a scikit-learn version change.

A change that only affects winner, landing or hit-height logic after final contacts have been selected does not by itself require a new contact/sequence fit. The [maintainer guide](maintaining.md) has a fuller table.

## 1. Saved extraction run

Training reads the same saved stages as annotation:

```text
<run-dir>/stages/
  metadata/<video-id>/...
  shuttle/<video-id>/...
  pose/<video-id>/...
  court/<video-id>/...
```

Training and later annotation need the same kind of shuttle, pose and court data. A model fitted on old court geometry is not directly comparable with one run against materially different court outputs.

The court stage needs the current court inputs used by annotation, and the shuttle stage needs guard codes. Raw detector JSON files have to be converted into the dataset builder's saved court files first. A missing file or unusable court geometry stops the command.

## 2. Contact labels

Each labelled video has a CSV with exactly these columns:

```csv
rally_id,frame,side
rally-1,100,Top
rally-1,130,Bot
rally-1,160,Top
rally-2,300,
rally-2,330,Top
```

Rules for the file:

- one row per human-labelled contact;
- frame numbers use the zero-based source-video timeline;
- `side` is `Top`, `Bot` or blank;
- a blank side means unknown, not a guess;
- rows for one rally are contiguous;
- frames within a rally are strictly increasing;
- every rally/contact in the included video is labelled.

The labels provide contact times and known court halves. Rally start and end frames are not labelled separately; the evaluation code matches the labelled contacts to predicted sections.

Partial labelling can turn an unlabelled real contact into a negative training example, so the included videos need complete contact labels.

## 3. Manifest

The manifest points at the saved extraction run and assigns each labelled video to `train`, `validation` or `test`.

Example `retune/manifest.json`:

```json
{
  "run_dir": "../data/court-extracted-run",
  "videos": [
    {"id": "train-a", "group": "match-a", "split": "train", "labels": "labels/train-a.csv"},
    {"id": "train-b", "group": "match-b", "split": "train", "labels": "labels/train-b.csv"},
    {"id": "train-c", "group": "match-c", "split": "train", "labels": "labels/train-c.csv"},
    {"id": "validation-a", "group": "match-d", "split": "validation", "labels": "labels/validation-a.csv"},
    {"id": "test-a", "group": "match-e", "split": "test", "labels": "labels/test-a.csv"}
  ]
}
```

Paths are resolved relative to the manifest file. Each `id` matches a video directory name in the extraction run and can appear only once.

### Groups

`group` keeps related footage together. Videos from the same match, or footage with closely shared scene/player conditions, belong to the same group.

A group can contain several video IDs but cannot appear in more than one split.

Sequence fitting needs at least three training groups. That is only the mechanical minimum: every fitted classifier also needs examples of both target classes in the folds that train it. A fold with only useful or only useless edits, or review labels with only correct or only wrong rallies, stops the fit with a clear error.

## 4. Fit commands

Video-level side geometry:

```bash
PYTHONPATH=src uv run python -m annotator.training fit \
  --manifest retune/manifest.json \
  --output models/annotator-video \
  --side-geometry video
```

Scene-level side geometry:

```bash
PYTHONPATH=src uv run python -m annotator.training fit \
  --manifest retune/manifest.json \
  --output models/annotator-scene \
  --side-geometry scene
```

`--side-geometry` defaults to `video`. The fitting command reads only `train` videos and labels.

Using a separate output directory for each model variant keeps later validation results tied to the exact model that produced them.

## 5. What happens during fitting

### Contact tree

For each training video, the training code runs the rule-based preprocessing and builds contact feature rows.

Training labels are assigned around human contact frames, with distances scaled for the source FPS:

- within 1 frame at 30 FPS — positive;
- 2–4 frames away — ignored as ambiguous;
- up to 15 frames away — nearby negative;
- more distant candidate rows — sampled to fill the remaining negative budget.

All nearby negatives are kept. Distant negative sampling uses a fixed seed and follows the order of the training videos, so manifest order is part of a reproducible comparison.

The order of the selected rows matters too. The same examples and seed fitted in a different row order give a different tree. The `fit` command samples and fits in manifest order. From Python, `fit_contact_model(..., fit_video_order=[...])` reorders the selected rows by video for the fit while sampling stays in the supplied order. Leaving it out keeps the default behaviour.

The contact fit keeps scikit-learn's automatic early stopping: a large fit sets aside part of its own training rows to decide when to stop. `ContactFitConfig.early_stopping` makes that choice explicit.

The final contact tree is then fitted on all training videos.

### Contact scores for sequence training

Sequence training uses contact probabilities from a tree that did not train on the same match group.

For each training group, an additional contact tree is fitted on the other groups. Its probabilities are used for videos in the held-out group. This produces out-of-group contact scores for sequence training.

The contact score cutoff and the optional rule for guarded candidates are applied at this point, exactly as they are during annotation. Neither changes the rows the contact tree itself is trained on.

### Sequence trees

The same option builder used at runtime creates keep, serve-repair, deletion and later-contact-insertion alternatives.

Human labels mark which alternatives are useful or correct. Grouped fitting then trains the sequence models from those rows.

The final sequence models are fitted after the out-of-group training examples have been constructed.

The contact folds are not nested a second time inside the sequence folds. This matches the original training procedure. The separate validation and test reports are the quality measurement; validation and test groups stay outside every fit.

### Rally review tree

The review tree is trained from completed out-of-group predictions. Each predicted section receives one of three labels:

- `1` — correct;
- `0` — known wrong;
- `-1` — not judgeable because required human side labels are missing, or because no labelled rally overlaps the section.

Rows labelled `-1` are left out of the confidence fit.

### Saved model directory

The resulting directory contains models and settings for the whole learned chain:

- contact tree;
- sequence model stack;
- rally review tree;
- contact-selection settings;
- rough-rally, mask and other preprocessing settings;
- side-geometry mode.

Keeping these together ensures that the sequence models receive the same kind of contact stream and rule-based inputs used to create their training examples.

## 6. Validation

Example for the video-geometry model:

```bash
PYTHONPATH=src uv run python -m annotator.training evaluate \
  --manifest retune/manifest.json \
  --models models/annotator-video \
  --split validation \
  --output retune/validation-video.json.gz
```

Example for the scene-geometry model:

```bash
PYTHONPATH=src uv run python -m annotator.training evaluate \
  --manifest retune/manifest.json \
  --models models/annotator-scene \
  --split validation \
  --output retune/validation-scene.json.gz
```

Evaluation loads the saved model directory and does not fit anything. It reads only the requested `validation` or `test` split and uses the settings stored in the model directory. The output path must end in `.json.gz`.

## 7. Reading the evaluation report

The report contains per-video counts and totals.

### Contact metrics

Predicted and labelled contacts are matched one-to-one within ten frames at 30 FPS, scaled for each video's FPS.

The report includes:

- labelled contacts;
- predicted contacts;
- matched contacts;
- contact precision;
- contact recall.

These numbers measure the contact stream directly.

### Rally-section metrics

A predicted rally section counts as correct when it matches one labelled rally and contains every labelled contact from that rally. It must contain no extra contacts and must agree with all known human side labels.

A missing contact, an extra contact, a merged rally, a partial rally or a contradicted known side makes a section wrong. Otherwise, missing human side labels make it unjudgeable. A predicted section that overlaps no labelled rally is also unjudgeable.

The report counts:

- judged predicted sections;
- correct predicted sections;
- unjudgeable predicted sections.

A completely missed labelled rally does not create a predicted-section row. These counts therefore measure correctness of predicted sections, not rally recall. Contact recall still captures missed contacts.

Winner, landing and hit-height accuracy are outside this report and need separate evaluation when those rules change.

## 8. Validation and test roles

Model variants are compared on validation with the same data splits and labels. After a model choice is settled, test provides the final held-out measurement.

Repeated model selection against test turns it into another validation set. The test report therefore works best as the final measurement of the chosen setup.

## 9. Python fit settings

The CLI exposes the manifest, output path and side-geometry choice. More detailed settings are in `TrainingSettings`, passed to `fit_from_manifest()` from Python. The command stays small on purpose; it is not a menu for research variants.

Current defaults are below.

### Contact tree

| Setting | Default |
| --- | ---: |
| learning rate | `0.06` |
| max iterations | `180` |
| max leaf nodes | `31` |
| min samples per leaf | `40` |
| L2 regularisation | `1.0` |
| seed | `20260824` |
| early stopping | `"auto"` |

### Sequence trees

General sequence-tree defaults:

| Setting | Default |
| --- | ---: |
| learning rate | `0.05` |
| max iterations | `200` |
| max leaf nodes | `15` |
| min samples per leaf | `20` |
| L2 regularisation | `1.0` |
| seed | `20260905` |

The serve tree uses 100 iterations, 7 leaves and seed `20260824`.

### Rally review tree

| Setting | Default |
| --- | ---: |
| learning rate | `0.05` |
| max iterations | `100` |
| max leaf nodes | `7` |
| min samples per leaf | `20` |
| L2 regularisation | `1.0` |
| seed | `20260905` |

A comparison is easier to interpret when one variable changes at a time. For example, a contact-tree hyperparameter comparison can keep preprocessing and the data split fixed.

## 10. Contact selection settings

`ContactModelConfig` holds two settings. Both are saved in the model directory and used again at annotation time.

| Setting | Default | Effect |
| --- | ---: | --- |
| `score_cutoff` | `0.9` | Lowest contact score kept in the initial contact stream |
| `reject_masked_without_player` | `False` | When true, drops candidate frames that have a rejected shuttle guard grade and no picked player at the five feature offsets |

[Fixed heuristics](heuristics.md#optional-rule-for-guarded-candidates) gives the exact condition for the second setting.

Both settings affect which contacts reach sequence refinement. A different value therefore also changes the sequence-model inputs and the examples used to fit the review tree. The contact tree's own training rows do not change.

A comparison sets the value through `TrainingSettings.contact_prediction` and fits the full model directory again. The later models then train on the same contact stream used during annotation. The `fit` command has no flag for either setting:

```python
from pathlib import Path

from annotator.contacts.model import ContactModelConfig
from annotator.training.workflow import TrainingSettings, fit_from_manifest, load_manifest

settings = TrainingSettings(
    contact_prediction=ContactModelConfig(reject_masked_without_player=True),
)
fit_from_manifest(
    load_manifest(Path("retune/manifest.json")),
    Path("models/annotator-rule"),
    settings=settings,
)
```

## 11. Checks before selecting a model directory

A complete model comparison normally includes:

- disjoint train, validation and test groups;
- full labels for every included video;
- model selection based on validation rather than test;
- both contact precision and contact recall;
- rally-section metrics interpreted as section correctness, not rally recall;
- inspection of representative changed predictions as well as aggregate counts;
- the same scikit-learn version for fitting and annotation;
- both `models.joblib` and `metadata.json` kept together;
- the selected model path recorded in the dataset-builder config (`models.annotator`).

## Planned new-court refit

**Status: fit and V validation queue launched on 3 October 2026; results pending.** The run uses the released 86-video patched court dataset. Machine-specific input paths stay in the untracked run configuration. Select on V before running the familiar ShuttleSet22 comparison.

This paired comparison has its own runner. It is separate from the `python -m annotator.training` commands above, although it calls the same fitting functions. The [runner README](../../experiments/annotator/good_court_refit/README.md) has the commands, the input config and the list of settings both builds hold fixed. The [evaluation template](../../experiments/annotator/good_court_refit/evaluation.md) lists what to measure afterwards.

Two builds are prepared, in this order:

1. **Baseline** — the current default settings on the new court inputs.
2. **Baseline plus the rule** — the same settings with `ContactModelConfig(reject_masked_without_player=True)`.

The rule does not change contact-tree training. The two builds therefore differ in their sequence and review trees and in the candidates used at annotation time.

Both builds use `video` side geometry. A `video` versus `scene` comparison is not part of this refit.

ShuttleSet22 video 15 is left out of this refit's inputs and of every split because its labels are misaligned. The old-court regression runner retains its historical 47-video test list, including video 15. Original-ShuttleSet `sset_15` is a different video and stays in.

The rule was first tried on the old court inputs. The [old-court findings](../../experiments/annotator/old_court_regression/findings.md) report that trial and explain why the earlier refit changed the benchmark. Those results describe the old court inputs. Court geometry affects which players get picked, so they do not show how the rule behaves on the new ones.

Once the builds have been run and evaluated, this section is where the chosen model directory, the evaluation it was chosen on and the remaining known failures get recorded.
