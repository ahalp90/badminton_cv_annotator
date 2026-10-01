# Retuning on new vision evidence

Fit a fresh complete bundle from labelled videos, compare it on held-out groups,
and use the chosen bundle for annotation. The contact tree, sequence trees and
review-ranking tree are fitted together by the supported command. This guide
does not require reconstructing the research experiments.

The improved court extraction needs a new retune and follow-up analysis. Its
performance has not yet been established. Use a completed extraction run.
The separate dataset rebuild can use existing human contact and rally labels;
it does not depend on this annotator retune.

## Prepare a manifest and labels

Use an existing dataset-builder run containing metadata, pose, court and
shuttle artefacts. For each video, the workflow reads its matching directory
under `run_dir/stages/`. Current operational court inputs and shuttle guard
codes must be available. Raw detector JSON files must first be converted into
the builder's operational court artefacts. Missing artefacts or unusable court geometry stop the
operation.

Create a JSON manifest such as `retune/manifest.json`:

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

Paths resolve relative to the manifest. `id` must match a video directory name
in the extraction run. IDs cannot repeat. Use `group` to keep related footage,
such as clips from the same match, together. A group may contain several videos
but may belong to only one split. At least three distinct training groups are
required for the sequence fits.

Each label file is a plain CSV with exactly these columns:

```csv
rally_id,frame,side
rally-1,100,Top
rally-1,130,Bot
rally-1,160,Top
rally-2,300,
rally-2,330,Top
```

Use one row per human-labelled contact. Frames are zero-based source-video
frames, within the video timeline. Each rally occupies one contiguous block;
frames within it must be strictly increasing. `side` is `Top`, `Bot` or empty.
An empty side is unknown, not a guessed player assignment. The labels supply
contact times and sides, not separately annotated rally start/end bounds.
Label every rally and contact throughout each included video. Partial
labelling makes training targets and evaluation results misleading.

The three-group minimum is not enough by itself. Each required tree fit needs
both target classes. Folds with only useful or only useless edits, or confidence
labels with only correct or only wrong rallies, fail clearly. Choose enough
labelled examples to represent those cases in each training fold.

## Fit and evaluate

Run these commands from the repository root in the supported Python environment:

```bash
PYTHONPATH=src python -m annotator.training fit \
  --manifest retune/manifest.json \
  --output models/annotator-video \
  --side-geometry video

PYTHONPATH=src python -m annotator.training evaluate \
  --manifest retune/manifest.json \
  --models models/annotator-video \
  --split validation \
  --output retune/validation-video.json.gz
```

Fitting reads only `train` videos and their label files. Evaluation loads a saved
bundle and reads only the requested `validation` or `test` split. It performs no
fitting. The bundle stores its preprocessing and geometry choice; evaluation
uses those stored settings.

The contact fit keeps positives within one frame at 30 FPS of a human contact.
It ignores ambiguous nearby rows through four frames, keeps nearby negatives
through fifteen, and samples more distant negatives with a fixed seeded
budget. These distances scale to the video's FPS. Manifest video order affects
seeded sampling, so keep it stable when comparing fits. The contact fit retains
automatic early stopping: large fits reserve an internal validation subset of
their training rows. `ContactFitConfig.early_stopping` makes that choice explicit.

Each training video is first scored by a contact tree fitted on the other
training groups. Sequence training also withholds its entire group from the
downstream models that produce its training scores. Confidence fitting uses the resulting
held-group predictions and their matching held-group insertion model. The
final bundle's trees are then fitted using all training groups.

The final contact tree uses all training videos for future inference. The
contact folds are not nested again inside the downstream folds; this matches
the original training procedure. Treat the separate validation/test reports as
the quality assessment. Validation and test groups remain outside every fit.

For deliberate fitting changes, call `fit_from_manifest` with
`TrainingSettings` from Python. The command keeps the settings small; it does
not expose a research or ablation menu. Changing fixed preprocessing or feature
policy requires a full refit, not just another score cut-off.

## Read the evaluation

The compressed JSON report contains per-video and total counts:

- Labelled, predicted and matched contacts, with contact precision and recall.
  Matching allows ten frames at 30 FPS, scaled to each video's FPS.
- Judged, correct and unjudgeable predicted rally sections. A correct section
  contains all of one labelled rally's contacts, no extras, and the correct
  known sides after the final side assignment.

A missing contact, extra contact, merged rally, partial rally or known side
contradiction makes a section wrong. Otherwise, missing human side labels make
it unjudgeable. A predicted section with no overlapping labelled rally is also
unjudgeable. Unknown sections are excluded from confidence fitting.

Whole-rally counts measure correctness of **predicted sections**. They do not
measure missed-rally recall: a labelled rally with no predicted section does not
create a wrong-section row. Contact recall still counts its missed contacts.
Read both sets of measures and inspect representative errors. The report does
not validate winners, landings or hit heights.

## The next court retune

Keep the labelled splits fixed. Fit one bundle with `--side-geometry video` and
another with `--side-geometry scene`, using separate output directories.
Evaluate both on validation and inspect differences in contact repairs and side
assignments. This is the remaining practical geometry choice, not a request to
rebuild the algorithm.

Choose using validation, then evaluate the chosen bundle on test. Avoid using
test results to keep adjusting the choice. Complete any small cleanup prompted
by real failures and point `models.annotator` at the chosen directory for future automatic annotation runs.

The fit/runtime scikit-learn versions must match exactly. Keep the complete
bundle with the resolved runtime used to fit it; after upgrading scikit-learn,
fit a new bundle. Historical binaries and earlier court-failure analysis remain
useful context, but they do not replace this fresh held-out evaluation.
