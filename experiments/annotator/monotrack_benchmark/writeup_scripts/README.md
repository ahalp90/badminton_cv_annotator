# Write-up scripts

These scripts produce the [v6 report](../annotator_monotrack_own_weights_compared_writeup_v6.md)'s
numbers, tables and figures.
The [v5 report](../annotator_monotrack_own_weights_compared_writeup_v5.md) and
earlier reports used earlier versions of these scripts, which are in the git history.
Most steps read committed results only. Three steps need data that is not in the
repository: the annotator runs need the saved inference inputs, the 40-video
contact refit needs prepared training inputs, and the galleries need frames from
the source videos.

Run every command from the repository root.

## Scripts

| Script | What it produces |
| --- | --- |
| [metrics.py](metrics.py) | Contact, rally and confidence scores for every condition, rescored from the committed prediction streams and checked against each run's saved scores. Ignores the 524 rallies with flawed labels. |
| [court_metrics.py](court_metrics.py) | Each rally's court error and IoU against ShuttleSet22's reference corners, for our per-rally, MonoTrack's and our per-scene courts. |
| [fit_contact_40.py](fit_contact_40.py) | Refits the contact model on all 40 original ShuttleSet videos for each test bundle. First checks that refitting on 32 videos reproduces the bundle's contact model exactly. |
| [annotator_runs.py](annotator_runs.py) | Runs our annotator's bundles on test (whole videos, and given rally bounds on both court sets) and in discovery mode on validation. Keeps each found rally's confidence score. The 32-video discovery runs must match the tracked predictions exactly. |
| [confidence_cutoffs.py](confidence_cutoffs.py) | Chooses the confidence thresholds on validation and reads them on test, for the threshold table. |
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
- The review threshold that keeps half of the validation rallies:
  `../../tree_simplification/results/input-review-thresholds.json.gz`.
- Our annotator with trees off: `../../heuristic_only_tune/results/`.
- ShuttleSet22 labels and reference corners: `data/shuttleset22/set/`.
- Original ShuttleSet labels: `training/data/shuttleset/annotations/shots_master.csv`.

## Run

```bash
PY=python
W=experiments/annotator/monotrack_benchmark/writeup_scripts
SHOTS=training/data/shuttleset/annotations/shots_master.csv
OUT=report_data

$PY $W/metrics.py --out $OUT
$PY $W/court_metrics.py --out $OUT
$PY $W/left_out_rallies.py --data $OUT --shots-master $SHOTS
$PY $W/confidence_cutoffs.py --data $OUT --shots-master $SHOTS --out $OUT
$PY $W/kept_rally_errors.py --data $OUT --out $OUT
$PY $W/flaw_scoring.py --out $OUT
$PY $W/make_figures.py --data $OUT --out figures
```

By default, `metrics.py` reads the committed five-seed MonoTrack results in
`benchmark_results/monotrack_five_seed/`. To score a new `five_seed.py` run,
pass `--monotrack-runs /path/to/five-seed-results`. Both MonoTrack court
conditions use all five seeds of the validation-selected configuration.
The remaining scripts read those updated summaries.

`metrics.py` takes `--workers` (default 6) and runs for several minutes.

### Annotator runs

The committed runs are enough for every step above. To repeat them, run the
contact refit first, because the runner uses its bundles:

```bash
RUNS=experiments/annotator/monotrack_benchmark/benchmark_results/annotator_runs
$PY $W/fit_contact_40.py --prepared PREPARED --out $RUNS/models_40_videos
$PY $W/annotator_runs.py --paths paths.toml --out $RUNS
```

- `PREPARED` is the baseline preparation of the 40 original videos, made by
  [`tree_simplification.prepare`](../../tree_simplification/README.md#reproduce-the-current-fit).
  The refit runs on a CUDA GPU, as the test bundles' contact fits did, so the
  32-video check can match them exactly.
- The paths file's `[paths]` table needs `test_courts`, `inpaint_root`,
  `pose_root`, `stages` and `development_courts`.
- `annotator_runs.py` runs one single-threaded job per video, with `--workers`
  jobs at once (default 28, about 1.3 GiB each). `--only` limits it to runs
  whose folder starts with a given prefix, such as `test/discovery`.

### Galleries

The galleries draw on frames decoded from the source videos, which are not in
the repository. `extract_frames.py` takes a JSON list of
`{"video": path, "frame": number, "name": file name}` requests. With the
middle frames in one folder and the main-camera frames in another:

```bash
$PY $W/render_gallery_v3.py --courts $OUT/rally_courts.csv.gz \
  --frames MIDDLE_FRAMES --main-frames MAIN_FRAMES --out figures
```
