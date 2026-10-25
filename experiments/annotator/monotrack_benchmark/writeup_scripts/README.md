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
| [retrain_boundaries.py](retrain_boundaries.py) | Refits the full annotator stack for 2/3, 3/3 and 5/5-second boundary policies, with optional reuse of saved preparation, contact fits and proposal features. |
| [metrics.py](metrics.py) | Contact, rally and confidence scores for every condition, rescored from the committed prediction streams and checked against each run's saved scores. Ignores the 524 rallies with flawed labels. |
| [court_metrics.py](court_metrics.py) | Each rally's court error and IoU against ShuttleSet22's reference corners, for our per-rally, MonoTrack's and our per-scene courts. |
| [fit_contact_40.py](fit_contact_40.py) | Refits the contact model on all 40 original ShuttleSet videos for each test bundle. First checks that refitting on 32 videos reproduces the bundle's contact model exactly. |
| [annotator_runs.py](annotator_runs.py) | Runs our annotator's bundles on test (whole videos, and given rally bounds on both court sets) and in discovery mode on validation. Keeps each found rally's confidence score. |
| [annotator_batch_runs.py](annotator_batch_runs.py) | Runs all three boundary policies together, sharing preparation and compatible rally options. Resumes completed video predictions in place. |
| [boundary_comparison.py](boundary_comparison.py) | Runs the contact-size comparison, batched inference, validation policy selection and all nine training/evaluation combinations. |
| [compare_discovery_context.py](compare_discovery_context.py) | Selects between two discovery configurations on all 668 validation rallies under a common no-maximum-context rule, with a separate 354-rally unflagged view. |
| [discovery_boundary_surveys.py](discovery_boundary_surveys.py) | Judges saved discovery outputs under both 1/1 and 2/3 context limits and searches every unique validation confidence threshold. |
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
  and validation runs in `validation/32_videos/seed-<N>/`. Current compact results
  are in [`../benchmark_results/discovery_context/`](../benchmark_results/discovery_context/).
- The selected test bundles:
  `../../tree_simplification/results/discovery_context/5_5/models/raw_baseline-seed-<N>/`
  and `body_window_36-seed-<N>/`. The contact-size comparison bundles are under
  `5_5/models_40/seed-<N>/` in the same refit directory.
- Our annotator with trees off: `../../heuristic_only_tune/results/`.
- ShuttleSet22 labels and reference corners: `data/shuttleset22/set/`.
- Original ShuttleSet labels: `training/data/shuttleset/annotations/shots_master.csv`.

## Score the selected model

Headline rally correctness uses no maximum context limit. Confidence surveys
separately judge the discovery window under 1/1 and 2/3 limits; they do not judge
the dataset export clip. All contact metrics retain outside-rally contacts.

```bash
PY=python
W=experiments/annotator/monotrack_benchmark/writeup_scripts
SHOTS=training/data/shuttleset/annotations/shots_master.csv
OUT=report_data
RUNS=SAVED_RUN/reporting/annotator_runs/5_5

$PY $W/metrics.py --annotator-runs $RUNS --out $OUT --no-max-buffer
$PY $W/discovery_boundary_surveys.py --run 5_5 $RUNS --out SURVEYS --workers 8
$PY $W/court_metrics.py --out $OUT
$PY $W/make_figures.py --data $OUT --confidence-data SURVEYS/5_5/2_3 --out figures
```

Court measurements and MonoTrack predictions are unchanged and can be reused.
`metrics.py` reads the committed five-seed MonoTrack results by default. Pass
`--monotrack-runs` to score a new run. Both court conditions use all five seeds
of the validation-selected configuration. `--workers` controls scoring workers.

### Compare capped and uncapped discovery

The comparison selects on five-seed mean fully correct coverage at ±10 frames,
using all 668 validation rallies and the same no-max rule. Ties use correct-player
F1, then contact F1. The unflagged view removes 314 marked rallies after judgement;
it does not remove contacts from F1 or labels from matching and neighbour checks.

```bash
$PY $W/compare_discovery_context.py \
  --run 5_5 SAVED_RUN/reporting/annotator_runs/5_5/validation/32_videos \
  --run no_max NEW_RUN/reporting/annotator_runs/no_max/validation/32_videos \
  --output COMPARISON --workers 8
$PY $W/discovery_boundary_surveys.py \
  --run 5_5 SAVED_RUN/reporting/annotator_runs/5_5 \
  --run no_max NEW_RUN/reporting/annotator_runs/no_max \
  --out SURVEYS --workers 8
```

The uncapped arm was trained and run at revision `65311a3e`. That checkout removes
output trimming as well as the training limits. To reproduce it, run its
`retrain_boundaries.py --policies no_max --reuse-training-root SAVED_RUN/training
--reuse-policy 5_5` with the normal paths/output arguments. Preparation, contact
fits and proposal features are reused; labels, selector and review fits change.
The selected current checkout restores the winning 5/5 model's 2/3 finishing.
These are two complete configurations, so their difference does not isolate
training from finishing. Keep saved selector predictions immutable; export
checks their parity before adding review confidence.

### Earlier three-policy training comparison

Retrain all three boundary policies with the fixed selector settings and five
seeds used in the benchmark. Each fitted tree uses two CPU cores; the driver
limits total parallel work to 28 cores. Hyperparameter retuning after the
boundary changes remains a low-priority TODO.

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
  --out NEW_RUN/reporting/target_comparison --workers 10
```

The target comparison runs model/seed groups in parallel and saves each finished
video under `target_comparison/per_video/`. Repeat the same command to resume
without reloading finished videos. Use a fresh output directory if the training
inputs or comparison rules change.

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
