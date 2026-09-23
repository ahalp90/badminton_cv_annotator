# Measure SVD pruning before changing search

Compare the frozen pair matcher on nine baseline B cases, using all 16
direction families and the 12 retained by the reviewed SVD ranking. This
measures the camera-direction gate, pair matching and per-pair shortlist
selection. It excludes full evidence scoring and scene processing.

## Protocol

- Preserve direction coordinates, original family/pair/candidate IDs,
  normalisation, camera thresholds and per-pair caps.
- Recompute SVD ranking from the frozen line arrays and masks; require the
  result to match the reviewed ranking. Record this overhead separately.
- Run both arms sequentially for each case. Alternate the first arm across
  cases, using up to six case workers and one numerical thread each.
- Compare every shared pair's non-timing output between arms. Check against
  the historical pair records when the cache directory is supplied.
- Record elapsed time and process CPU time separately. Compare summed arm
  times and per-case ratios; the mixed batch wall time is not a standalone
  12-family or 16-family runtime.

The [cached retention check](../evidence/webui_followup3_20260922/review_20260923/automatic_retention/README.md)
already preserves all eight historically approved automatic fits. Some score
winners change, as recorded there. These timings do not settle ranking quality.

## Execution

`run_benchmark.py --help` documents the runner. `--max-pairs` explicitly marks
an output as a smoke run; it must not be reported as a complete benchmark.
Each output directory must be new or empty.

The active protocol and checks are recorded in [WORKLOG.md](WORKLOG.md).
The full 16-family path and G0 fallback remain available. Matcher optimisations
from the independent compute audit will be assessed separately from this
initial comparison.
