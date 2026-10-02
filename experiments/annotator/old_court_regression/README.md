# Old-court retrain and regression check

This runner refits every selected annotator model with the current `src/`
code and scikit-learn, using the original old-court vision files. It then runs
full annotation on held-out videos and scores the old PR149 output and the new
output with one scorer. The run checks the current annotator end to end before
the new-court retune. It does not replace that retune.

The completed [findings and small follow-up experiments](findings.md) give the
regression conclusion. The [new-court preparation](../good_court_refit/README.md)
is a separate comparison.

## What it runs

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

Populations follow the original design. The final contact tree fits all 40
development videos in ID order and serves the 47 test videos. Each A-D video
is scored by a tree fitted on the other three groups, in group order. V is
scored by a tree fitted on A-D in ID order, as `final_contact_scores/group_V`
records. Downstream and confidence trees fit A-D only. Test includes video 15.

## Inputs

Old vision files are court-evidence/0.1. The current loaders reject 0.1, so
`old_inputs.py` reads its operational fields (`raw_cuts`, `inputs`, keep vote,
court presence) with the existing parsers. It passes them to `run_video` with
the arguments `load_contact_evidence` and `run_full_annotation_stage` use.
The old per-scene provenance records are ignored.

- `--dev-stages DIR`: `DIR/stages/{shuttle,pose,court}/sset_NN/` with
  `shuttle_track.npy.xz`, `shuttle_guard_codes.npy.xz`, five `pose_*.npy.xz`,
  `court_evidence.json.gz`, `court_keep_vote.npy.xz` and `court_present.npy.xz`.
  `scratch/contact_det/raw/region_v2_inputs` has this layout for three videos.
- `--test-inputs DIR`: the ShuttleSet22 inpaint root. Each `DIR/"NN <name>"/`
  holds `shuttle_track_inpainted.npy.xz`, `shuttle_guard_codes_inpainted.npy.xz`,
  the pose files and the three court files.
- Defaults point at tracked repository records: `shots_master.csv`, the group
  and split records, and PR149 outputs in `scratch/contact_det_closing_pass/`.
  `--clean-labels` defaults to the untracked
  `scratch/contact_det_full_ds_fit/raw/shuttleset22-test-result/clean_labels.json.gz`.
- Optional: `--frozen-dev-features .../raw/full_raw` adds a 40-video fit on the
  research-time features and per-video feature comparisons.
  `--frozen-test-features .../raw/shuttleset22-test-predictions` compares test
  features. `--contact-records scratch/contact_det_full_ds_fit/raw` compares
  example counts and the 80 saved check-row scores.
  `--shuttleset22-annotations DIR` adds the all-source read (3,965 rallies).

## Run

```bash
PYTHONPATH=.:src python -m experiments.annotator.old_court_regression.runner \
  --dev-stages DEV_RUN --test-inputs SS22_INPAINT_ROOT --output OUT \
  --frozen-dev-features scratch/contact_det_full_ds_fit/raw/full_raw \
  --frozen-test-features scratch/contact_det_full_ds_fit/raw/shuttleset22-test-predictions \
  --contact-records scratch/contact_det_full_ds_fit/raw \
  --jobs 4 --threads 4
```

Parallel stages run `--jobs` worker processes with `--threads` numerical
threads each. Single-item stages get `jobs × threads` threads. Each worker
handles one item and exits. `sequence-fit` is one long item and writes no
progress lines until it finishes; on three videos it took about 20 minutes.

Rerun the same command to resume. An item is complete when its summary
`.json.gz` exists. Failed items leave a traceback under `failures/<stage>/`
and the run stops after that stage with a `FAILED` marker. The other items in
the stage still finish. To redo finished work, delete that stage's outputs and
every later stage's outputs. A different set of inputs needs a new `--output`.

## Outputs

`progress.log` has one line per item: UTC time, stage, done/total, item,
status and seconds. Long-running items also emit a heartbeat each minute.
`timings.json` holds per-stage wall time. `run.json`
records each invocation's inputs, library versions and Git commit.

`bundle/` (40-video contact tree) and `bundle_v32/` are complete supported
bundles. `eval/<split>/<id>/` holds the full `annotator_result.json.gz`, the
final stream with confidence scores and a summary. `comparison.md` gives
contact precision/recall, distinct fully correct rallies over labelled rallies, rallies
gained and lost, and confidence ranking metrics. `comparison.json.gz` adds
per-video detail.

## What a finished run shows, and what it does not

The scorer is the supported evaluator. On PR149's saved test stream it gives
PR149's published trusted-label result exactly: 33,716 of 38,218 contacts
matched from 41,605 predicted, 1,763 of 3,422 rallies fully correct, and 784
sections kept at the 0.757 confidence threshold with 616 of 740 correct.

Old/new differences mix four causes: regenerated features, intended edge
changes in the current code, scikit-learn 1.6.1 to 1.9.1, and refitting.
Regenerated features match the frozen rows exactly for sset_15 and differ in
312 of 54,656 rows for sset_01. The optional frozen-feature fit separates the
contact-tree part of that drift.

Limits:

- The supported loaders and `run_full_annotation_stage` are not exercised,
  because they reject 0.1 court evidence. Their `run_video` call is.
- V has no integrated PR149 output; V reports the new system only.
- The development comparison is held-group output on both sides. The old
  stream combines PR149's local-insertion stream with its corrected bounds.
- PR149's 0.757 threshold belongs to the old confidence model. Compare the
  new model at the same kept count as well.
- Fold trees use group order. Exact equality with the original fits is not
  expected after the library change.
