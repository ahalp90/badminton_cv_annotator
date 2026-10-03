# Reproduce the old-court refit comparison

The old-court comparison checked whether the annotator refactor preserved the
historical behaviour and what changed when its models were retrained. Reusing
the historical contact scores and sequence models reproduced all 47 saved
output streams. Refitting reduced fully correct rallies from 1,763 to 1,734;
the [findings report](refit_regression.md) explains that loss and the attempted fixes.

This runner reproduces the comparison using the original court, shuttle and
pose data. It fits the models, annotates the held-out videos, and scores the
historical PR149 output and new output against the same labels. To reproduce
the later comparison that selected the new-court base model, use the
[new-court reproduction guide](model_refit.md).

## Inputs

The runner needs saved shuttle, pose and court data for 40 development videos
and 47 ShuttleSet22 test videos. The test set includes video 15, matching the
historical comparison. Supply the two input roots as follows:

- `--dev-stages DIR`: `DIR/stages/{shuttle,pose,court}/sset_NN/` with
  `shuttle_track.npy.xz`, `shuttle_guard_codes.npy.xz`, five `pose_*.npy.xz`,
  `court_evidence.json.gz`, `court_keep_vote.npy.xz` and `court_present.npy.xz`.
  `scratch/contact_det/raw/region_v2_inputs` has this layout for three videos.
- `--test-inputs DIR`: the ShuttleSet22 inpaint root. Each `DIR/"NN <name>"/`
  holds `shuttle_track_inpainted.npy.xz`, `shuttle_guard_codes_inpainted.npy.xz`,
  the pose files and the three court files.

The other defaults use the repository's saved `shots_master.csv`, group and
split records, and PR149 outputs in `scratch/contact_det_closing_pass/`.
`--clean-labels` defaults to the untracked file
`scratch/contact_det_full_ds_fit/raw/shuttleset22-test-result/clean_labels.json.gz`.

The old court files use `court-evidence/0.1`. Because current loaders require
0.2, this runner uses `old_inputs.py` to read the fields annotation needs:
`raw_cuts`, `inputs`, keep vote and court presence. It passes those fields to
`run_video()` using the same arguments as the normal annotation path. The old
per-scene provenance records are left out.

### Optional checks against saved research inputs

These options help trace a difference back to features or training examples:

| Option | Additional check |
|---|---|
| `--frozen-dev-features .../raw/full_raw` | Fit a contact tree on the saved research-time features for all 40 development videos, and compare features per video. |
| `--frozen-test-features .../raw/shuttleset22-test-predictions` | Compare regenerated test features with the saved features. |
| `--contact-records scratch/contact_det_full_ds_fit/raw` | Compare example counts and the 80 saved check-row scores. |
| `--shuttleset22-annotations DIR` | Also evaluate against all source annotations, containing 3,965 rallies. |

## Run

```bash
PYTHONPATH=.:src python -m experiments.annotator.old_court_regression.runner \
  --dev-stages DEV_RUN --test-inputs SS22_INPAINT_ROOT --output OUT \
  --frozen-dev-features scratch/contact_det_full_ds_fit/raw/full_raw \
  --frozen-test-features scratch/contact_det_full_ds_fit/raw/shuttleset22-test-predictions \
  --contact-records scratch/contact_det_full_ds_fit/raw \
  --jobs 4 --threads 4
```

Parallel stages run `--jobs` worker processes, each with a numerical-thread
allowance of `--threads`. A stage containing one item receives the combined
`jobs × threads` allowance. Workers exit after completing their item.

The sequence fit is one long item, so its completion line appears only when
the fit finishes; the log emits a heartbeat each minute in the meantime. A
three-video sequence fit took about 20 minutes.

Rerun the same command to resume. An item's summary `.json.gz` marks it as
complete. If an item fails, its traceback is saved under `failures/<stage>/`;
the other items in that stage finish, then the run stops with a `FAILED` marker.
To repeat completed work, remove that stage's outputs and every later stage's
outputs. Use a new `--output` directory when changing inputs.

## What it runs

The development videos are divided into groups A–D and validation group V.
The contact tree used for the final test fit sees all 40 development videos.
The validation tree sees only the 32 videos in A–D, keeping V out of training.

The sequence models need examples of the contact tree's mistakes. Each A–D
video is therefore scored by a contact tree trained on the other three groups.
Those held-out predictions supply the sequence training examples. The sequence
and confidence models use A–D only.

| Stage | Work | Current functions used |
|---|---|---|
| `labels` | Label CSVs for 40 development and 47 test videos | `load_labels` contract |
| `old-streams` | PR149 test stream, test confidence, development held-group stream | — |
| `features` | Heuristic pass and contact features per development video | `run_video(heuristic_only=True)`, `features_from_evidence` |
| `contact-fits` | 40-video final tree, four 24-video fold trees, the 32-video tree for V | `fit_contact_model` |
| `pools` | A-D option pools from each video's held-group tree | `sequence_training_video` |
| `sequence-fit` | Nested sequence fits, confidence on held-group output, two bundles | `fit_sequence_models`, `fit_confidence_model`, `save_models` |
| `evaluate` | Full annotation of 8 V and 47 test videos | `load_models`, `run_video` with models |
| `compare` | Old and new streams, same labels and scorer, per video | `evaluation_counts`, `whole_rally_correctness` |
| `report` | `comparison.json.gz` and `comparison.md` | — |

For reproduction, the final 40-video contact tree and the 32-video validation
tree fit their selected examples in video-ID order. The smaller trees that
hold out one of A–D fit in group order. The historical
`final_contact_scores/group_V` records establish the validation tree's order.

## Outputs

The output directory contains two complete model bundles:

- `bundle/`: the 40-video contact tree and fitted sequence/review models, used
  for the 47 test videos;
- `bundle_v32/`: the 32-video contact tree and fitted sequence/review models,
  used for the eight V videos.

`eval/<split>/<id>/` holds each video's `annotator_result.json.gz`, final contact
stream, confidence scores and summary. `comparison.md` reports contact precision
and recall, fully correct labelled rallies, rallies gained and lost, and the
quality of the confidence ranking. `comparison.json.gz` adds per-video detail.

The logs let a resumed or inspected run be traced back to its inputs:
`progress.log` records each item's UTC time, stage, done/total count, identity,
status and duration; `timings.json` records stage wall times; and `run.json`
records invocation inputs, library versions and the Git commit.

## How to interpret the comparison

The scorer was checked by replaying PR149's saved test output. It reproduced
the published trusted-label results:

| Measurement | Reproduced result |
|---|---:|
| Labelled contacts found | 33,716 of 38,218 |
| Predicted contacts | 41,605 |
| Fully correct rallies | 1,763 of 3,422 |
| Clips above the historical 0.757 confidence cutoff | 784 |
| Fully correct clips among the 740 that trusted labels could judge | 616 |

The remaining 44 selected clips could not be judged with the trusted labels.
For a new confidence model, compare the quality of an equally sized review
queue as well as applying 0.757: the same numerical cutoff can select a
different number and quality of clips after refitting.

The end-to-end old/new comparison combines regenerated features, intended
edge-case changes in the code, the move from scikit-learn 1.6.1 to 1.9.1, and
fresh fitting. For example, regenerated features match the saved rows exactly
for `sset_15`, but differ in 312 of 54,656 rows for `sset_01`. The optional
frozen-feature fit helps measure how much those input differences affect the
contact tree. The [component experiments](refit_regression.md#why-the-fitted-results-differ)
separate the fitted-model changes more closely.

A few details determine which results can be compared directly:

- **Loader coverage:** the runner tests the current `run_video()` path through
  an adapter for old court files. The current saved-input loaders and
  `run_full_annotation_stage()` need a separate check with 0.2 court evidence.
- **Validation:** only the new system has integrated V output, so V has no
  corresponding PR149 result in this run.
- **Development:** both sides use predictions made while each video's group
  was held out. The historical stream combines PR149's local-insertion contacts
  with its corrected rally boundaries.
- **Exact reproduction:** the fold trees retain group order. Their fitted
  probabilities can change with the scikit-learn version; aggregate results
  alone do not establish identical fits.
