# Write-up scripts

These scripts retrain and evaluate the annotator, select confidence thresholds
on validation, and produce the benchmark's numbers, tables and figures.
Scoring reads saved predictions. Retraining and inference need prepared vision
inputs, and the galleries need frames from the source videos.

Run every command from the repository root.

## Scripts

| Script | What it produces |
| --- | --- |
| [retrain_annotator.py](retrain_annotator.py) | Fits the annotator across preprocessing configurations and seeds, sharing preparation and optionally reusing contact fits and option pools. |
| [metrics.py](metrics.py) | Contact, rally and confidence scores for every condition, rescored from the committed prediction streams and checked against each run's saved scores. Ignores the 524 rallies with flawed labels. |
| [court_metrics.py](court_metrics.py) | Each rally's court error and IoU against ShuttleSet22's reference corners, for our per-rally, MonoTrack's and our per-scene courts. |
| [fit_contact_40.py](fit_contact_40.py) | Refits the contact model on all 40 original ShuttleSet videos for each test bundle. First checks that refitting on 32 videos reproduces the bundle's contact model exactly. |
| [annotator_runs.py](annotator_runs.py) | Runs our annotator's bundles on test (whole videos, and given rally bounds on both court sets) and in discovery mode on validation. Keeps each found rally's confidence score. |
| [annotator_batch_runs.py](annotator_batch_runs.py) | Runs the `no_max` bundles by default, sharing preparation and compatible rally options across seeds and optional configurations. Resumes completed video predictions in place. |
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
  `../../tree_simplification/results/discovery_context/no_max/models/raw_baseline-seed-<N>/`
  and `body_window_36-seed-<N>/`. The contact-size comparison bundles are under
  `no_max/models_40/seed-<N>/` in the same refit directory.
- Our annotator with trees off: `../../heuristic_only_tune/results/`.
- ShuttleSet22 labels and reference corners: `data/shuttleset22/set/`.
- Original ShuttleSet labels: `training/data/shuttleset/annotations/shots_master.csv`.

## Fit and run the baseline

Training and normal discovery use no maximum context limit. The driver fits
both preprocessing configurations across five seeds and shares preparation
between seeds. Parallel work is limited to 28 CPU cores.

```bash
PY=python
W=experiments/annotator/monotrack_benchmark/writeup_scripts
TREES=NEW_RUN/training/no_max
RUNS=NEW_RUN/annotator_runs/no_max
OUT=report_data

$PY $W/retrain_annotator.py --paths paths.toml --output $TREES \
  --workers 28 --device cuda
$PY $W/fit_contact_40.py --prepared $TREES/prepared/raw_baseline \
  --tree-results $TREES --out NEW_RUN/models_40/no_max --device cuda
$PY $W/annotator_batch_runs.py --paths paths.toml \
  --training-root NEW_RUN/training --models-40 NEW_RUN/models_40 \
  --out NEW_RUN/annotator_runs --workers 14
```

The training output contains `models/<candidate>-seed-<N>/` and
`test-run/candidates/<candidate>/seed-<N>/`. Add
`--reuse-training-root EXISTING_TRAINING_RUN` to reuse preparation and contact
fits. Add `--reuse-options` as well to reuse matching current-schema proposals,
features and targets. Each stage checks compatibility; incompatible pools need
a fresh output directory. `--plan-only` shows the planned stage counts without
running fits.

The paths file's `[paths]` table needs `test_courts`, `inpaint_root`,
`pose_root`, `stages` and `development_courts`. The 40-video contact comparison
uses the baseline preparation of the original 40 videos. Run it on the same
device used for the bundles' contact fits so its 32-video parity check can match.

### Batched inference and resume

The batch runner defaults to `--policies no_max`. Each selected label names a
subdirectory in the training, 40-video model and output roots. Labels identify
saved configurations; all use the same current discovery rule. To run one
configuration directly, pass its directories to `annotator_runs.py`:

```bash
$PY $W/annotator_runs.py --paths paths.toml --tree-results $TREES \
  --models-40 NEW_RUN/models_40/no_max --out $RUNS --workers 14
```

Batch workers share prepared inputs and contact features, then share rally
options where contact models agree. Each configuration keeps its own selector
and confidence model. Workers stay alive between videos; each tree uses two
CPU cores. `--only test/discovery` limits either runner to discovery test runs.

Repeat the same command to resume. Completed predictions are retained and
missing predictions are written atomically. When a batch extends existing
results, it checks that the recomputed streams and confidence scores agree.
Keep input and model locations unchanged when resuming.

## Score the baseline

Normal discovery judges the first and last accepted contacts with no maximum
outward context. It allows inward shortening by the contact tolerance: ±10
base-30 frames for the headline result. Full rally correctness also requires
all contacts and players to match. Contact F1 includes outside-rally contacts.

Boundary-only confidence filtering allows two seconds before the first labelled
contact and three after the last, with the same inward tolerance. Interior
contact and player errors do not affect this measure. The survey also reports
1/1 limits. These tests judge contact endpoints; dataset export separately adds
2/3-second viewing context around the accepted contacts.

```bash
$PY $W/metrics.py --annotator-runs $RUNS --out $OUT
$PY $W/discovery_boundary_surveys.py --run no_max $RUNS --out SURVEYS --workers 8
$PY $W/court_metrics.py --out $OUT
$PY $W/make_figures.py --data $OUT --confidence-data SURVEYS/no_max/2_3 --out figures
```

`metrics.py` reads the committed five-seed MonoTrack results by default; pass
`--monotrack-runs` to score a new run. Court measurements and saved predictions
can be reused. Use a fresh scoring output directory when its rules change.

For an optional comparison, give each saved configuration a label and its
validation directory:

```bash
$PY $W/compare_discovery_context.py \
  --run baseline BASELINE_RUN/validation/32_videos \
  --run candidate CANDIDATE_RUN/validation/32_videos \
  --output COMPARISON --workers 8
```

The comparison uses five-seed mean fully correct coverage on all 668 validation
rallies and separately reports the 354 unflagged rallies. All labels remain in
contact matching and neighbouring-rally checks. Reuse saved full-video contact
counts with `--reuse-contact-counts PREVIOUS_COMPARISON` when predictions and
timing tolerances are unchanged. Confidence thresholds are selected on validation
and applied unchanged to test.

### Galleries

The galleries draw on frames decoded from the source videos, which are not in
the repository. `extract_frames.py` takes a JSON list of
`{"video": path, "frame": number, "name": file name}` requests. With the
middle frames in one folder and the main-camera frames in another:

```bash
$PY $W/render_gallery_v3.py --courts $OUT/rally_courts.csv.gz \
  --frames MIDDLE_FRAMES --main-frames MAIN_FRAMES --out figures
```
