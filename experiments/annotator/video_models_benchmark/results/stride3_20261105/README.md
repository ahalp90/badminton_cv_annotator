# Stride-three benchmark results — November 2026

This collection preserves the completed X3D and VideoMAE comparison started on
5 November, including the three additional X3D seeds started afterward. It
contains summaries for all **23 training runs / 240 validation checks**, and
detailed validation and test predictions for the **five X3D and two VideoMAE
models selected for testing**.

## Files

| File | What you can inspect |
|---|---|
| `runs.json.gz` | All three queue manifests; each run's settings, continuation schedule, checkpoint summaries, selected step and final step. |
| `selection.json.gz` | The saved X3D configuration ranking and VideoMAE learning-rate ranking. |
| `data.json.gz` | Original cache indexes: video identities, split membership, frame rates, rally bounds and labelled contacts. |
| `scores.csv.gz` | One row per winning seed, split, hitter rule and timing tolerance: 56 rows altogether. |
| `predictions/{x3d,videomae}/seedN/{validation,test}.json.gz` | Every retained hit, with its original and forced-alternation hitter; frame metrics and aggregate contact/rally scores. |

Download the seven model checkpoints from the [weights release](../../weights/README.md).
Frame-probability arrays and RGB frame caches are stored separately.

## Reading the predictions

Each prediction file groups hits by video and rally. Each `events` row is:

```text
[frame, raw_confidence, raw_hitter, forced_alternation_hitter]
```

`frame` is the absolute video-frame index. `raw_confidence` is the model's
probability for its original predicted hitter at that frame; it is not the
probability of the replacement hitter. `Top` and `Bot` identify court halves.
The same hit times and confidence values apply to both hitter rules.

Use `item_id` to join a video to `data.json.gz`, and `span_id` to join its rally.
Rally intervals include `start_frame` and exclude `end_frame`. The original
`rally_id` is also retained. Unknown labelled hitters remain `null`.
Source filenames identify the original videos and label files.

`metrics.raw` keeps every hit's original hitter. `metrics.forced_alternation`
uses the alternating sequence with most agreement; equal votes are resolved by
the summed class probabilities, then agreement with the first original hitter.

## Comparison and selection

- Data: 32 training videos, eight validation videos and 46 test videos.
  Validation has 668 rallies / 5,696 contacts; test has 3,327 rallies / 37,184
  contacts. Validation retains flaw-marked rallies; test excludes them.
- Inference: every third video frame, offset zero, with supplied rally bounds.
  Each input clip contains 16 consecutive frames. Nearby predicted contacts are
  suppressed using the shared six-frame-at-30-fps rule.
- Timing tolerances: ±5 and ±10 frames at 30 fps, scaled to each video's frame rate.
- Checkpoints: earliest minimum validation cross-entropy within each run.
- Configurations: highest mean fully correct validation rallies across seeds 0/1
  at ±10 frames, breaking ties by correct-player F1. Test results do not select
  settings. Additional X3D seeds use the frozen winner.
- X3D winner: ±1-frame positive centres, 50% contact sampling, LR 0.02, seeds 0–4.
- VideoMAE winner: the same positive window/contact share, LR 1e-4, seeds 0/1.
  Centres two frames from a contact are ignored unless another contact makes
  them positive. Checkpoint selection and continuation settings are saved in
  `runs.json.gz`.

Contact F1 measures hit detection; correct-player F1 additionally requires the
right hitter. A fully correct rally requires every labelled hit, no extra hits,
and every hitter correct. Event scores are percentages. Frame `macro_f1` is
between zero and one; frame confusion matrices use no-contact / Bot / Top order.
Ignored frame targets contribute to neither the frame loss nor its confusion
matrix. A history row's `seconds` is elapsed time since that training invocation
started, not the duration of that validation check.

Queue names and training manifests are retained in `runs.json.gz`.

## Rebuild and verification

From the repository root, using the downloaded queues and cache indexes:

```bash
PYTHONPATH=src:. python \
  experiments/annotator/video_models_benchmark/writeup_scripts/collect_stride3_results.py \
  --source "$BENCHMARK_RUNS"
```

The collector reads the saved hit times and forced hitters, then recovers the
original hitter from the corresponding frame probabilities. It checks frame
alignment and confidence values, rescores both hitter rules with the shared
matcher, and requires every forced-alternation count to equal the original
saved result at both tolerances. Only the collected hit lists and summaries
are written here; rebuilding requires the larger source arrays.
