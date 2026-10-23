# MonoTrack benchmark

The benchmark compares MonoTrack and our annotator on 46 ShuttleSet22 videos:
3,327 rallies and 37,184 labelled hits. Video 15 is excluded from this fixed set.

The main comparison has five conditions:

| Method | Courts | Rally boundaries |
|---|---|---|
| MonoTrack | MonoTrack, per rally | Provided |
| MonoTrack | Ours, per rally, after the person vote | Provided |
| Our annotator | Ours, per rally, after the person vote | Provided |
| Our annotator | MonoTrack, per rally | Provided |
| Our annotator | Stored fast-robust courts from PySceneDetect scenes | Discovered |

MonoTrack uses its own player selection and the retrained HitNet weights.
The adapter gives HitNet the court-corner order used in training; the vendored
MonoTrack code stays unchanged. When our annotator receives MonoTrack's courts,
it accepts every successful fit without applying our detector's person vote.

Both MonoTrack rows use one configuration chosen on the eight original
ShuttleSet validation videos. The comparison covers kept/zeroed shuttle input,
L2 weights 0, 1e-6 and 1e-4, and inference interpolation on/off. Every setting
uses seeds 0, 100, 200, 300 and 400, with dropout off.
The highest mean fully correct rally percentage at ±10 frames wins;
Correct-player F1 breaks a tie. Selection calls the report's scoring function,
including its overlapping-rally judgement. All 668 validation rallies count.
The same five checkpoints are then tested with both court sets.

For provided boundaries, contacts are scored inside the common padded rally
windows. Discovery is scored across the whole video, including false positives
outside those windows. Fully correct rallies use the same fixed windows in all
five conditions. The main results use ±10 frames. Appendix E gives the ±5-frame
timing-sensitivity comparison.

Each tolerance has three measures:

- Contact precision, recall and F1 count one-to-one matches within the time allowance.
- Attributed-contact precision, recall and F1 require those matches to have the correct
  hitter too. A wrong hitter counts as both a false positive and a missed labelled contact.
- Fully correct rallies (%) require every labelled contact, no extra contacts, and the
  correct hitter for every contact within the common rally window.

Court and inference failures remain in the denominator. An inference exception
fails the affected rally or video and the remaining work continues. Missing
result files mean a run is unfinished and prevent scoring.

## Run

For the five-seed MonoTrack comparison, set `stages`, `spans` and
`development_courts` to the original ShuttleSet inputs used for training.
Keep all thirty full training outputs under `TRIALS/<arm>_l2<value>/seed-<N>/`.
The six setting names are `kept_l20`, `kept_l21e-06`, `kept_l20.0001`, and their
`zeroed` equivalents. Reuse all six completed dropout-off seed-0 runs,
including their summaries and histories.
Run the selector with the annotator environment; it starts inference with the
configured TensorFlow interpreter:

```bash
python five_seed.py \
  --config config.local.toml \
  --trials /path/to/trials \
  --output /path/to/five-seed-results \
  --parallel 16
```

The output includes every validation prediction, `selected.json.gz`, ten test
runs, report-compatible scores, and all thirty checkpoints with training records.
The output directory must be new. Missing or failed jobs stop the workflow.
An inference failure within a rally remains an empty prediction for that rally.
Test results are produced only after validation selection is written.

The commands below describe the individual runners and the annotator baseline.

Use the annotator's Python environment for the preparation, scoring and launch
commands. Set the separate TensorFlow 2.15 interpreter in `config.local.toml`.

First copy the existing court inputs and common windows into a portable input
folder. This also builds the court inputs for our annotator on MonoTrack courts:

```bash
python prepare_inputs.py \
  --monotrack-manifest /path/to/monotrack/controlled_manifest.json.gz \
  --ours-manifest /path/to/rally_courts/controlled_manifest.json.gz \
  --discovery-manifest /path/to/scene_courts/controlled_manifest.json.gz \
  --output ../benchmark_results/inputs
cp config.toml config.local.toml
```

Set the data paths and Python interpreters in `config.local.toml`. The checked-in
input folder can be reused when the evaluation population and court inputs are
unchanged. Run from this directory with the annotator environment active. Set
`monotrack_python` to the TensorFlow 2.15 interpreter and choose a fresh output
directory:

```bash
benchmark_output=/path/to/benchmark-output
monotrack_python=/path/to/tensorflow-2.15-environment/bin/python

python run_benchmarks.py \
  --config config.local.toml \
  --phase ours \
  --output "$benchmark_output" &&
"$monotrack_python" monotrack_runner.py \
  --config config.local.toml \
  --arm zeroed --courts monotrack --interpolation off \
  --output "$benchmark_output/mono_zeroed_own_off" &&
python score.py \
  --inputs ../benchmark_results/inputs \
  --run "$benchmark_output/mono_zeroed_own_off" &&
"$monotrack_python" monotrack_runner.py \
  --config config.local.toml \
  --arm zeroed --courts ours --interpolation off \
  --output "$benchmark_output/mono_zeroed_our_courts_off" &&
python score.py \
  --inputs ../benchmark_results/inputs \
  --run "$benchmark_output/mono_zeroed_our_courts_off"
```

Use the same input folder for `score.py` as `paths.inputs` in the configuration.
Each condition needs a fresh output directory. The annotator phase can run while
HitNet trains; the MonoTrack commands require the exported weights.

To evaluate the frozen tuned heuristic with supplied rally windows, pass its
selected settings to `annotator_runner.py`:

```bash
python annotator_runner.py \
  --config config.local.toml --condition ours_our_courts \
  --heuristic-settings ../../heuristic_only_tune/results/selected.json.gz \
  --output "$benchmark_output/heuristic_our_courts" &&
python score.py \
  --inputs ../benchmark_results/inputs \
  --run "$benchmark_output/heuristic_our_courts"
```

Repeat with `--condition ours_mono_courts` and a fresh output directory for
MonoTrack's courts. Both conditions supply the exact stored windows, from one
second before the first labelled contact to one second after the last. The
heuristic settings stay fixed across court conditions. Omitting
`--heuristic-settings` retains the full annotator.

The reports use these five condition directories under
`../benchmark_results/comparison/`:

| Condition | Directory |
|---|---|
| Our annotator, our per-rally courts | `ours_our_courts/` |
| Our annotator, MonoTrack courts | `ours_mono_courts/` |
| Our annotator, full discovery | `ours_discovery/` |
| MonoTrack, its own courts | `mono_zeroed_own_off/` |
| MonoTrack, our per-rally courts | `mono_zeroed_our_courts_off/` |

To reproduce the remaining Appendix A configurations, run `monotrack_runner.py`
with `--courts monotrack` and each of the following settings, then score each
output with `score.py`:

| Shuttle input | Inference interpolation | Output directory |
|---|---|---|
| `--arm kept` | `--interpolation on` | `mono_kept_own_on/` |
| `--arm kept` | `--interpolation off` | `mono_kept_own_off/` |
| `--arm zeroed` | `--interpolation on` | `mono_zeroed_own_on/` |

The outputs contain per-video predictions, failure records and scores.
Each condition's `scores.json.gz` supplies the report's results. These files
separate failures by stage and count discovery predictions outside the labelled
rally windows. Local configuration and execution logs are ignored by Git.

## Review failures

To inspect MonoTrack inference failures, generate clips with source-frame numbers
and the ground-truth hit labels overlaid:

```bash
python review_failures.py \
  --config config.local.toml \
  --results "$benchmark_output" \
  --output /path/to/new-review-clips
```

Each affected rally gets one clip, even if it failed in several conditions.
`index.json.gz` lists the failures and GT labels. The reviewed clip and its note
are retained under `../benchmark_results/review_clips/`.
FFmpeg with subtitle and text-rendering support is required.

## Checks

```bash
python -m pytest tests -q
```

The tests cover failure denominators, discovery false positives, corner ordering,
shuttle interpolation, court acceptance without the person vote, and review-clip
length. Each inference runner also accepts `--only <video_id>` for a smoke test.
