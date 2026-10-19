# Write-up scripts

These scripts produce the numbers, tables and figures in the
[v3](../annotator_monotrack_own_weights_compared_writeup_v3.md) and
[v4](../annotator_monotrack_own_weights_compared_writeup_v4.md) reports. Most
steps read committed results only. Two steps need data that is not in the
repository: the confidence reruns need the saved inference inputs, and the
galleries need frames from the source videos.

Run every command from the repository root.

## Scripts

| Script | What it produces |
| --- | --- |
| [metrics.py](metrics.py) | Contact, rally and confidence scores for every condition, rescored from the committed prediction streams and checked against each run's saved scores. Ignores the 524 rallies with flawed labels. |
| [court_metrics.py](court_metrics.py) | Each rally's court error and IoU against ShuttleSet22's reference corners, for our per-rally, MonoTrack's and our per-scene courts. |
| [confidence_rerun.py](confidence_rerun.py) | Reruns the 32-video models in discovery mode on test and keeps each found rally's confidence score. Checks that hits and rally spans match the committed streams. |
| [confidence_rerun_validation.py](confidence_rerun_validation.py) | The same rerun on the 8 validation videos. |
| [confidence_cutoffs.py](confidence_cutoffs.py) | Chooses the confidence cut-offs on validation and reads them on test, for the cut-off table. |
| [left_out_rallies.py](left_out_rallies.py) | Counts the rallies with flawed labels for Appendix G. |
| [flaw_scoring.py](flaw_scoring.py) | Scores discovery mode with the flawed rallies standard, ignored and counted, for Appendix G. |
| [make_figures.py](make_figures.py) | Figures 1–6 and 10. |
| [extract_frames.py](extract_frames.py) | Decodes named frames from the source videos for the galleries. |
| [render_gallery.py](render_gallery.py) | Gallery drawing helpers. Run directly, it draws the earlier reports' galleries, including the video 17 contact sheet and the full-size panels in `writeup_figures/court_gallery/` that v3 and v4 link to. |
| [render_gallery_v3.py](render_gallery_v3.py) | Figures 7–9. |

## Inputs

- Predictions and court inputs: [`../benchmark_results/`](../benchmark_results/).
- Our annotator with trees on: `../../tree_hp_tune/results/test_rerun/`. The
  report uses the `grid-017` runs, set by `CONFIGURATION` in `metrics.py`.
- Our annotator with trees off: `../../heuristic_only_tune/results/`.
- Confidence reruns: [`../benchmark_results/confidence_rerun/`](../benchmark_results/confidence_rerun/),
  with `test/` and `validation/` folders of five seeds each.
- ShuttleSet22 labels and reference corners: `data/shuttleset22/set/`.
- Original ShuttleSet labels: `training/data/shuttleset/annotations/shots_master.csv`.

## Run

```bash
PY=python
W=experiments/annotator/monotrack_benchmark/writeup_scripts
RERUN=experiments/annotator/monotrack_benchmark/benchmark_results/confidence_rerun
SHOTS=training/data/shuttleset/annotations/shots_master.csv
OUT=report_data

$PY $W/metrics.py --confidence $RERUN/test --out $OUT
$PY $W/court_metrics.py --out $OUT
$PY $W/left_out_rallies.py --data $OUT --shots-master $SHOTS
$PY $W/confidence_cutoffs.py --validation $RERUN/validation --data $OUT --shots-master $SHOTS --out $OUT
$PY $W/flaw_scoring.py --out $OUT
$PY $W/make_figures.py --data $OUT --out figures
```

`metrics.py` takes `--workers` (default 6) and runs for several minutes.

### Confidence reruns

The committed rerun output is enough for every step above. To repeat the reruns,
pass the repository root and a TOML file whose `[paths]` table points at the
saved inference inputs:

```bash
$PY $W/confidence_rerun.py --repository . --paths test_paths.toml --output $RERUN/test
$PY $W/confidence_rerun_validation.py --repository . --paths validation_paths.toml --output $RERUN/validation
```

- Test keys: `test_inputs`, `test_courts`, `inpaint_root` and `pose_root`.
- Validation keys: `shots_master`, `stages` and `development_courts`.

### Galleries

The galleries draw on frames decoded from the source videos, which are not in
the repository. `extract_frames.py` takes a JSON list of
`{"video": path, "frame": number, "name": file name}` requests. With the
middle frames in one folder and the main-camera frames in another:

```bash
$PY $W/render_gallery_v3.py --courts $OUT/rally_courts.csv.gz \
  --frames MIDDLE_FRAMES --main-frames MAIN_FRAMES --out figures
```
