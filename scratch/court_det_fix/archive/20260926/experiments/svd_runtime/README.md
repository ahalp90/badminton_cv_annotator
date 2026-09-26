# SVD: current state

**The 12-family SVD screen is integrated and pushed as `1353541`.** It reduces
matching work in fresh experimental W5 generation. The full 16-family option
remains available. The separate six-case search-depth experiment is complete. Deeper matching
substantially improves Am2, confirmed by the user's visual review.

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

**Complete:** all 18 outputs returned with coordinator exit 0 at `14310f7`.
Across six cases, deeper axes cost 26.6% more summed full-trial runtime than
SVD baseline; wider shortlists cost 10.0% more. Deeper reduces Am2's selected
reference disagreement from 323.6 to 18.4 native pixels. Its best refitted
reference disagreement falls from 288.8 to 7.9 px. This is the strongest search-
depth result. The user judges only saved W5 and deeper suitable on Am2; deeper
is acceptable before refitting and perfect afterwards.

The user judges the three arms' reference-best refits perfect on both easy
broadcast cases, and all SVD refits near-perfect on GX0. GX5 and Am1 remain
hallucinations in the SVD gallery. Saved W5 is great on GX5 and includes G1 and
templates, which the G0-only experiment excludes. Keep those candidate sources.
The [search worklog](../svd_search/WORKLOG.md) contains the complete table,
visual assessments and timing limits. Detector defaults remain unchanged.

## Remaining work

1. Retain deeper matching as a demonstrated useful way to spend saved compute;
   choosing a new default or adaptive search policy is a separate decision.
2. Review the completed colour diagnostic on preserved candidates. No new
   direction search or refitting was needed for that measurement pass.

The separate [compute-efficiency audit](COMPUTE_AUDIT.md) identified possible
matcher optimisations. Those changes have not been applied to either benchmark.

The [colour-consistency diagnostic](../colour_consistency/PLAN.md) is complete.
It reused preserved fits and assignments without rerunning SVD search. Its
71-view gallery includes selections and retrospective refits from all six
SVD cases.
