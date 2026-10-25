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
| [confidence_cutoffs.py](confidence_cutoffs.py) | Searches every distinct validation confidence score for the three operating objectives, then reports test results at those cut-offs. |
| [kept_rally_errors.py](kept_rally_errors.py) | Counts the errors in kept rallies that need a fix, at the two cut-offs in the project README. |
| [left_out_rallies.py](left_out_rallies.py) | Counts the rallies with flawed labels for Appendix G. |
| [flaw_scoring.py](flaw_scoring.py) | Scores discovery mode with the flawed rallies standard, ignored and counted, for Appendix G. |
| [make_figures.py](make_figures.py) | Figures 1–6 and 10. |
| [extract_frames.py](extract_frames.py) | Decodes named frames from the source videos for the galleries. |
| [render_gallery.py](render_gallery.py) | Gallery drawing helpers. Run directly, it draws the earlier reports' galleries, including the video 17 contact sheet and the full-size panels in `writeup_figures/court_gallery/` that the reports link to. |
| [render_gallery_v3.py](render_gallery_v3.py) | Figures 7–9. |

## Inputs

- Predictions and court inputs: [`../benchmark_results/`](../benchmark_results/).
- Our annotator with trees on: [`../benchmark_results/annotator_runs/`](../benchmark_results/annotator_runs/),
  made by `annotator_runs.py`. Test runs sit in `test/<condition>/<models>/seed-<N>/`
  and validation runs in `validation/32_videos/seed-<N>/`. The 40-video bundles
  are in `models_40_videos/`.
- The test bundles: `../../tree_simplification/results/models/raw_baseline-seed-<N>/`
  and `body_window_36-seed-<N>/`. Their tracked discovery predictions and
  validation results are elsewhere in `../../tree_simplification/results/`.
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

For the policy comparison, score every training policy under each of the three
evaluation policies. Keep a separate output directory for each pair. Choose
the training policy using unfiltered, fully correct validation-rally coverage
under the common 2/3-second evaluation rule; then choose its confidence
cut-offs on validation. Test results do not choose either.

### Galleries

The galleries draw on frames decoded from the source videos, which are not in
the repository. `extract_frames.py` takes a JSON list of
`{"video": path, "frame": number, "name": file name}` requests. With the
middle frames in one folder and the main-camera frames in another:

```bash
$PY $W/render_gallery_v3.py --courts $OUT/rally_courts.csv.gz \
  --frames MIDDLE_FRAMES --main-frames MAIN_FRAMES --out figures
```
