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

Both shuttle variants are evaluated on MonoTrack's courts with the released
inference interpolation enabled. The variant with higher test contact F1 at
±10 frames supplies the two MonoTrack rows above. Both variant results are
retained for the report's footnote. Each is also evaluated with interpolation
disabled as a separate comparison.

For provided boundaries, contacts are scored inside the common padded rally
windows. Discovery is scored across the whole video, including false positives
outside those windows. Fully correct rallies use the same fixed windows in all
five conditions. Results include ±10- and ±5-frame contact scores.

Court and inference failures remain in the denominator. An inference exception
fails the affected rally or video and the remaining work continues. Missing
result files mean a run is unfinished and prevent scoring.

## Run

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

Set the data paths and Python interpreters in `config.local.toml`, then run:

```bash
python run_benchmarks.py \
  --config config.local.toml \
  --output ../benchmark_results/comparison
```

Each condition needs a fresh output directory. To run our annotator while
HitNet trains, use `--phase ours`. Later run `--phase monotrack`, followed by
`--phase summarise`, with the same output root.

The outputs contain per-video predictions, failure records and scores.
`comparison.json.gz` holds the five main results and interpolation comparisons;
`shuttle_comparison.json.gz` holds both shuttle results and the selected variant.
Local configuration and execution logs are ignored by Git.

## Review failures

To inspect MonoTrack inference failures, generate clips with source-frame numbers
and the ground-truth hit labels overlaid:

```bash
python review_failures.py \
  --config config.local.toml \
  --results ../benchmark_results/comparison \
  --output ../benchmark_results/review_clips
```

Each affected rally gets one clip, even if it failed in several conditions.
`index.json.gz` lists the failures and GT labels. The clips are ignored by Git.
FFmpeg with subtitle and text-rendering support is required.

## Checks

```bash
python -m pytest tests -q
```

The tests cover failure denominators, discovery false positives, corner ordering,
shuttle interpolation, court acceptance without the person vote, and review-clip
length. Each inference runner also accepts `--only <video_id>` for a smoke test.
