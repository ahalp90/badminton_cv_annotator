# SVD: current state

**The 12-family SVD screen is integrated and pushed as `1353541`.** It reduces
matching work in fresh experimental W5 generation. The full 16-family option
remains available. Deeper search is now authorised as a separate experiment;
its effect on court quality has not yet been measured.

## What is implemented

The screen ranks the original direction-support groups by SVD residual and
keeps 12 of the 16 families. Matching still uses their original directions.
Original family, pair and candidate identities are preserved at unchanged
search depth. Fresh G0 and G1 generation use
[`automatic_generation.py`](../w5_holistic/automatic_generation.py).

Fitting, scoring, G0 fallback and line-template proposals are unchanged. The
complete scene-level detector is still a separate integration task. Its planned
sampling across each scene already addresses additional-frame evidence.

Use a fresh output directory for each configuration. `run_cases.py` defaults
to `--direction-budget 12`; use `--direction-budget 16` for the comparator.
The historical `wider_evaluation/run_remote.sh` stays pinned to 16 because it
writes into the earlier evaluation directory.

## What the evidence supports

- On nine saved development cases, SVD12 retains all nine best reference-fit
  candidates and all eight historically approved automatic fits. Three existing
  score-winner roles are lost; substitutes have mixed quality. This supports an
  efficiency tradeoff, not an accuracy improvement or a general safety guarantee.
  See the [retention results](../evidence/webui_followup3_20260922/review_20260923/automatic_retention/README.md).
- The integration passed a bounded real matcher/scoring comparison, focused
  tests and type checks. Original geometry and IDs were preserved. The largest
  shared numerical difference in that smoke was 8.47e-11. The [worklog](WORKLOG.md)
  records checks and the existing whole-project lint/PATH exceptions.
- The nine-case Carmack timing run is complete. SVD12 including ranking used
  52.9% less summed matcher wall time than full16. All 2,160 historical and
  1,188 shared-arm comparisons passed. Direct timings contain between-pass
  variation; the [timing results](RESULTS.md) also report retained-work and
  shared-pair measures. The run excludes full image scoring, refitting and
  scene processing.

## Spending the saved computation

The new experiment holds SVD12, automatic G0 inputs, fitting and selection fixed:

| Arm | Axis assignments kept | Courts kept per pair / overall |
|---|---:|---:|
| Baseline | 512 | 256 / 256 |
| Deeper matching | 640 | 256 / 256 |
| Larger refit shortlist | 512 | 512 / 512 |

The six cases are SS03-19, SS03-34, GX0, GX5, Am1-54 and Am2-28019. Increasing
both axis caps to 640 permits about 56% more combinations. The gallery will
separate the detector's selection from the best reference-agreement candidate.
That tests whether extra search generates better courts and whether selection
actually chooses them.

**Current status:** all 18 case/configuration combinations are running on
Carmack with six workers at `14310f7`. Opus 5-5's implementation findings were
checked and resolved before launch. Its
[worklog](../svd_search/WORKLOG.md) and [run instructions](../svd_search/RUN_READY.md)
hold execution details. Detector defaults remain unchanged.

## Remaining work

1. Collect the existing search-depth run and build the gallery. Keep the
   combined worker count at six or fewer; do not launch a duplicate run.
2. Judge search depth from runtime and visible court quality.

The separate [compute-efficiency audit](COMPUTE_AUDIT.md) identified possible
matcher optimisations. Those changes have not been applied to either benchmark.
