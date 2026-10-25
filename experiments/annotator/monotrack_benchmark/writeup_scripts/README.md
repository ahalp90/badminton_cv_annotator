# Write-up scripts

These scripts retrain and evaluate the annotator, select confidence thresholds
on validation, and produce the benchmark's numbers, tables and figures.
Earlier reports use the scripts at their corresponding git revisions.
Scoring reads saved predictions. Retraining and inference need prepared vision
inputs, and the galleries need frames from the source videos.

Run every command from the repository root.

## Scripts

| Script | What it produces |
| --- | --- |
| [retrain_boundaries.py](retrain_boundaries.py) | Refits the full annotator stack for 2/3, 3/3 and 5/5-second boundary policies, sharing prepared inputs and newly fitted contact trees. |
| [metrics.py](metrics.py) | Contact, rally and confidence scores for every condition, rescored from the committed prediction streams and checked against each run's saved scores. Ignores the 524 rallies with flawed labels. |
| [court_metrics.py](court_metrics.py) | Each rally's court error and IoU against ShuttleSet22's reference corners, for our per-rally, MonoTrack's and our per-scene courts. |
| [fit_contact_40.py](fit_contact_40.py) | Refits the contact model on all 40 original ShuttleSet videos for each test bundle. First checks that refitting on 32 videos reproduces the bundle's contact model exactly. |
| [annotator_runs.py](annotator_runs.py) | Runs our annotator's bundles on test (whole videos, and given rally bounds on both court sets) and in discovery mode on validation. Keeps each found rally's confidence score. |
| [annotator_batch_runs.py](annotator_batch_runs.py) | Runs all three boundary policies together, sharing preparation and compatible rally options. Resumes completed video predictions in place. |
| [boundary_comparison.py](boundary_comparison.py) | Runs the contact-size comparison, batched inference, validation policy selection and all nine training/evaluation combinations. |
| [compare_training_targets.py](compare_training_targets.py) | Compares the actual consumed selector targets after checking that proposal rows and features align. |
| [confidence_cutoffs.py](confidence_cutoffs.py) | Searches every distinct validation confidence score for the three operating objectives, then reports test results at those cut-offs. |
| [kept_rally_errors.py](kept_rally_errors.py) | Counts the errors in kept rallies that need a fix, at the exact validation-selected cut-offs. |
| [left_out_rallies.py](left_out_rallies.py) | Counts the rallies with flawed labels for Appendix G. |
| [validation_label_sensitivity.py](validation_label_sensitivity.py) | Checks the raw validation flaw flags against scored labels and measures coverage on the unflagged subset. |
| [flaw_scoring.py](flaw_scoring.py) | Scores discovery mode with the flawed rallies standard, ignored and counted, for Appendix G. |
| [make_figures.py](make_figures.py) | Figures 1–6 and 10. |
| [extract_frames.py](extract_frames.py) | Decodes named frames from the source videos for the galleries. |
| [render_gallery.py](render_gallery.py) | Gallery drawing helpers. Run directly, it draws the earlier reports' galleries, including the video 17 contact sheet and the full-size panels in `writeup_figures/court_gallery/` that the reports link to. |
| [render_gallery_v3.py](render_gallery_v3.py) | Figures 7–9. |

## Inputs

- Predictions and court inputs: [`../benchmark_results/`](../benchmark_results/).
- Our annotator with trees on: fresh runs made by `annotator_runs.py` or
  `annotator_batch_runs.py`. Test runs sit in `test/<condition>/<models>/seed-<N>/`
  and validation runs in `validation/32_videos/seed-<N>/`. Compact v7 results
  are in [`../benchmark_results/boundary_retrain_v7/`](../benchmark_results/boundary_retrain_v7/).
- The selected test bundles:
  `../../tree_simplification/results/boundary_retrain_v7/2_3/models/raw_baseline-seed-<N>/`
  and `body_window_36-seed-<N>/`. The contact-size comparison bundles are under
  `2_3/models_40/seed-<N>/` in the same refit directory.
- Our annotator with trees off: `../../heuristic_only_tune/results/`.
- ShuttleSet22 labels and reference corners: `data/shuttleset22/set/`.
- Original ShuttleSet labels: `training/data/shuttleset/annotations/shots_master.csv`.

## Run

```bash
PY=python
W=experiments/annotator/monotrack_benchmark/writeup_scripts
SHOTS=training/data/shuttleset/annotations/shots_master.csv
OUT=report_data
RUNS=NEW_RUN/annotator_runs

$PY $W/metrics.py --annotator-runs $RUNS --out $OUT \
  --boundary-before-seconds 2 --boundary-after-seconds 3
$PY $W/court_metrics.py --out $OUT
$PY $W/left_out_rallies.py --data $OUT --annotator-runs $RUNS --shots-master $SHOTS
$PY $W/confidence_cutoffs.py --validation $RUNS/validation/32_videos \
  --metrics $OUT --shots-master $SHOTS --out $OUT \
  --boundary-before-seconds 2 --boundary-after-seconds 3
$PY $W/kept_rally_errors.py --data $OUT --out $OUT
$PY $W/validation_label_sensitivity.py --validation $RUNS/validation/32_videos \
  --shots-master $SHOTS --out $OUT/validation_label_sensitivity.json.gz
$PY $W/flaw_scoring.py --annotator-runs $RUNS --out $OUT
$PY $W/make_figures.py --data $OUT --out figures
```

By default, `metrics.py` reads the committed five-seed MonoTrack results in
`benchmark_results/monotrack_five_seed/`. To score a new `five_seed.py` run,
pass `--monotrack-runs /path/to/five-seed-results`. Both MonoTrack court
conditions use all five seeds of the validation-selected configuration.
The remaining scripts read those updated summaries.

`metrics.py` takes `--workers` (default 6) and runs for several minutes.

### Annotator runs

Retrain all three boundary policies with the fixed selector settings and five
seeds used in the benchmark. Each fitted tree uses two CPU cores; the driver
limits total parallel work to 28 cores. Hyperparameter retuning after the
boundary changes remains a TODO.

```bash
$PY $W/retrain_boundaries.py --paths paths.toml --output NEW_RUN/training \
  --workers 28 --device cuda
```

For each policy, run the contact-size comparison and inference with its own
selector and confidence models:

```bash
TREES=NEW_RUN/training/2_3
$PY $W/fit_contact_40.py --prepared NEW_RUN/training/prepared/raw_baseline \
  --tree-results $TREES --out $RUNS/models_40_videos --device cuda
$PY $W/annotator_runs.py --paths paths.toml --tree-results $TREES \
  --models-40 $RUNS/models_40_videos --out $RUNS --workers 14 \
  --boundary-before-seconds 2 --boundary-after-seconds 3
```

- `training/prepared/raw_baseline` is the baseline preparation of the 40 original videos, made by
  [`tree_simplification.prepare`](../../tree_simplification/README.md#reproduce-the-current-fit).
  The refit runs on a CUDA GPU, as the test bundles' contact fits did, so the
  32-video check can match them exactly.
- The paths file's `[paths]` table needs `test_courts`, `inpaint_root`,
  `pose_root`, `stages` and `development_courts`.
- `annotator_runs.py` runs one two-core job per video, with `--workers`
  jobs at once (default and maximum 14). `--only` limits it to runs
  whose folder starts with a given prefix, such as `test/discovery`.

### Batched inference and resume

For the three-policy comparison, run compatible bundles together:

```bash
$PY $W/annotator_batch_runs.py --paths paths.toml \
  --training-root NEW_RUN/training --models-40 NEW_RUN/models_40 \
  --out NEW_RUN/annotator_runs --workers 14
```

The training, 40-video model and output roots each contain `2_3`, `3_3` and
`5_5` subdirectories. Each 40-video bundle retains its policy's selector and
confidence models while using the shared, freshly refitted contact model.

The runner loads a video's inputs and prepares contact features once per court
condition and preprocessing setting. It builds rally options once per contact
model, then applies each policy's own selector and confidence model. Workers
stay alive between videos. Each tree still uses two CPU cores.

Repeat the same command to resume. Completed prediction files are retained;
missing predictions are written atomically. When a shared batch contains an
existing prediction, the runner checks that the recomputed stream and confidence
scores agree before adding the other policies. Use the same inputs and model locations when resuming; the runner checks
the saved model-location and boundary-policy records.

For the policy comparison, score every training policy under each of the three
evaluation policies. Keep a separate output directory for each pair. Choose
the training policy using unfiltered, fully correct validation-rally coverage
under the common 2/3-second evaluation rule; then choose its confidence
cut-offs on validation. Test results do not choose either.

The full post-training comparison can be run with:

```bash
$PY $W/boundary_comparison.py --training-root NEW_RUN/training \
  --paths paths.toml --output NEW_RUN/reporting \
  --monotrack-runs experiments/annotator/monotrack_benchmark/benchmark_results/monotrack_five_seed
$PY $W/compare_training_targets.py --training NEW_RUN/training \
  --out NEW_RUN/reporting/target_comparison
```

The driver records completed stages and resumes them only when their commands
and required outputs agree. Use `--plan-only` to inspect its commands. Generated
predictions are under `NEW_RUN/reporting/annotator_runs/<policy>/`; metric and
confidence results are under `NEW_RUN/reporting/metrics/train-<policy>__eval-<policy>/`.
Use a fresh output directory after changing the scoring rules.

### Galleries

The galleries draw on frames decoded from the source videos, which are not in
the repository. `extract_frames.py` takes a JSON list of
`{"video": path, "frame": number, "name": file name}` requests. With the
middle frames in one folder and the main-camera frames in another:

```bash
$PY $W/render_gallery_v3.py --courts $OUT/rally_courts.csv.gz \
  --frames MIDDLE_FRAMES --main-frames MAIN_FRAMES --out figures
```
