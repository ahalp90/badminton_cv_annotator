# Terminal progress

`python -m dataset_builder run`, `python -m dataset_builder replay`, and
`python -m annotator` show tqdm progress automatically when stderr is a terminal.
Use `--progress off` for the existing plain output, or `--progress on` to force
the display. No additional dependency is required.

The current stage has an indented progress line underneath it. Once the stage
finishes, that line disappears and the stage becomes an aligned completion row
with its actual output count and elapsed time. Reused, unavailable and failed
stages are labelled distinctly. Configuration-wide bypasses appear once at the
top; a failure in a normally bypassed stage remains visible.

Counters come from completed work, rather than a historical countdown:

- TrackNet and InpaintNet report completed batches. Streaming loaders without
  a known batch count show the count and elapsed time without a percentage.
- Pose shards report frames; their disjoint ranges combine into one video-wide
  counter. Stitching and saving are separate activities after extraction.
- Court detection reports scenes, followed by court pooling when applicable.
- Annotation shows rally finding, contact features, contact scoring, sequence
  refinement, confidence, outcomes, hit heights and saving. Feature intervals
  and rally outcomes have counters; bulk model calls show their activity.
- Projection counts rallies, assembly counts videos, and downloads count
  collected video outcomes. Remaining operations show activity and elapsed time.

ETA appears only when a total and a finite positive processing rate are
available. It can appear after processing has started, and disappears when
that operation finishes or a new activity lacks a total. It estimates the
current operation, not all remaining stages of the run. A 100% counter means
that operation's units have finished; saving and validation can still follow.

The coordinator keeps its existing phase order and scheduling. Parallel pose
workers share one progress channel; this change does not start multiple e2e
runs concurrently.

With the terminal display enabled, Python stdout/stderr from active stages are
saved under `terminal-logs/` in the run directory. Standalone annotation stores
these logs beside its output directory. Failure rows link to the stage log.
Child-process capture and failure details are preserved. The display uses a
short-lived local event file, inherited by child processes and removed after
the stage. Display events do not become annotation outputs or manifest inputs.

Validation uses CPU fixtures, including spawned pose workers. A real CUDA
TrackNet/RTMLib/DeepLSD run has not been performed for this change.

Progress is best-effort: terminal, counter-channel, renderer, and progress-log
failures disable the display or fall back to ordinary output. They do not retry
or skip pipeline work, change processing errors, or suppress cancellation.
TrackNet's Python status messages are captured in the same per-stage log when
progress is active; with progress off it keeps its normal terminal output.
