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

## Active detector integration

Fresh W5 generation now uses `w5_holistic/automatic_generation.py`. The
`wider_evaluation/run_cases.py` CLI defaults to `--direction-budget 12`;
`--direction-budget 16` keeps the full comparison. Use a fresh `--output`
directory for each budget. Resume checks reject a different budget or method.
Historical records without screen metadata count as full-16 only.

The screen ranks the original support groups using SVD residuals. It transforms
line coefficients into the estimator's saved normalised coordinates and gives
each line a unit two-dimensional normal. Lower residual wins; ties prefer more
support lines, then the original group ID. Matching uses the original direction
points. The SVD-fitted points are used only to measure each group's residual.
Original group, ordered-pair and candidate IDs remain intact.

Both the unfiltered G0 and paint-filtered G1 populations remain available.
Their shared direction estimator remains at 16 families. The matcher still
keeps up to 512 axes per direction and 256 courts per pair and overall. Fitting,
scoring, line-template proposals and acceptance rules are unchanged. The older
`wider_evaluation/run_remote.sh` launcher is pinned to 16 families because it
writes into the historical 22 September evaluation directory.

This is the active experimental detector path. The complete scene-level
runtime is still a separate integration task. Any increase in matching depth
should be measured separately: more axes increase the number of combinations
within each pair, so a 45% reduction in pair count does not translate directly
into a 45% increase in useful search depth.
