# SVD search-depth worklog

## Resume

**Complete: all 18 outputs returned, coordinator exit 0.** Six saved cases
were run with three SVD12 search configurations at commit `14310f7`. Each arm
generated automatic G0 proposals and reused the W5 fitter and ranker. Six remote
workers each used one numerical thread. The earlier nine-case timing benchmark
is separate and also completed with exit 0.

- Local results: `run_20260923/cases/`; generation records: `run_20260923/generation/`
- Receipts: `receipts/full.exit`, `revision.txt` and `full.log`
- Remote run: `/scratch/ahalperi/court_det_fix/svd_search_run_20260923`
- The monitor has finished and released its connection. Do not relaunch
- The six-case gallery and preview are complete at `http://127.0.0.1:8879/`

## Completed search comparison

**Deeper axes help on Am2; wider shortlists have not shown a comparable win.**
Baseline keeps 512 axes and 256 courts per pair/global; deeper keeps 640 axes
with the same court caps; shortlist keeps 512 axes and 512 courts per pair/global.
All use SVD12. These six development cases are not held-out accuracy evidence.

Each cell below gives **detector selection / best selectable refitted candidate**
as maximum corner disagreement with the saved reference, in native pixels.
The second value is a retrospective oracle: it uses reference annotations to
choose a candidate and is not the detector's output. Pixel disagreement is not
a substitute for the user's visual assessment.

| Case | Baseline | Deeper axes | Wider shortlist |
|---|---:|---:|---:|
| SS03-19 | 8.4 / 3.0 | 7.0 / 3.0 | 8.4 / 3.0 |
| SS03-34 | 5.8 / 2.8 | 6.9 / 2.8 | 5.8 / 2.8 |
| GX0 | 12.9 / 10.7 | 12.9 / 10.7 | 12.9 / 10.7 |
| GX5 | 919.7 / 727.3 | 918.1 / 909.7 | 698.9 / 698.9 |
| Am1-54 | 2288.0 / 2288.0 | 2288.0 / 2288.0 | 2288.0 / 2027.6 |
| Am2-28019 | 323.6 / 288.8 | 18.4 / 7.9 | 323.6 / 272.1 |

Summed full trial wall times were **6,255.4 s baseline, 7,919.1 s deeper and
6,879.3 s shortlist**. Deeper costs 26.6% more; shortlist costs 10.0% more.
These sums include view preparation, fresh directions, generation and W5
measurement/refitting/ranking. They exclude imports and result writing. This
was one six-worker run; per-case reversals in runtime should not be treated as
repeatable speedups. The earlier 52.9% saving measured matcher work against
full16, so its denominator differs from these full-trial sums.

Am2 is the material deeper-search result: its selected disagreement falls from
323.6 to 18.4 px and its best refit from 288.8 to 7.9 px. The user confirms only
saved W5 and deeper axes are suitable: deeper is acceptable before refitting
and perfect after refitting. Its detector selection exactly matches saved W5
(maximum corner difference 0.0 px). GX0 and both broadcast cases have identical best-refitted
reference errors across all three arms. GX5 and Am1 remain visually unacceptable
in the user's SVD review despite some numerical changes.

Keep SVD12 and retain G1/templates. Deeper G0 search has a confirmed visual benefit on Am2; the experiment does not yet justify changing defaults or
replacing the wider candidate population. Colour diagnostics reuse these saved
candidates to examine evidence for selection and placement.

The generator and runner are frozen for this experiment. Gallery controls and
stripe geometry shapes are fixed in the local display code. All further preview
updates belong to GPT-6 Sol medium, escalating to high only for a specific
unresolved problem. The Luna monitor owns monitoring and transport only.

The [colour diagnostic](../colour_consistency/PLAN.md) is complete, including
the bounded audit corrections and native-resolution batch. Every delegate
must reuse existing work and avoid extra audits, manifests or bookkeeping.
The user performs visual checks.

## User review: two broadcast controls

**Detector selections:** the user prefers deeper axes among the three SVD arms.
It offers no material improvement over the saved W5 G1 selections. Selected
corners from deeper and saved G1 are numerically identical in both cases. The
wider shortlist selects exactly the baseline corners and offers no benefit here.
Baseline and deeper are both usable, with small overshoot/undershoot under
different conditions; deeper and saved G1 look more reliable on this sample.

**Best refitted by reference:** the user clarified that all three SVD arms look
like perfect, equivalent fits on both samples. All slightly outshine the marginal
flaw seen in saved W5 G1. Baseline therefore wins on wall time for that available
candidate quality. These are reference-selected examples, not the detector's
choices. The practical lead is better ranking of already available fits; this
pair does not show a need for more search to obtain a good fit.

This assessment covers only SS03-19 and SS03-34, two easy broadcast cases.
The user wants challenging amateur cases before drawing a broader conclusion.
GX0, GX5, Am1-54 and Am2-28019 were already in the same run. They are now
complete and reviewed below; no additional dispatch was needed.

## User review: GX0

All three SVD arms are judged near enough to perfect after refitting and better
than saved W5, whose back-left corner overshoots. Saved W5 remains usable in a
pinch. The user reports back-left overshoot in the initial detections across
arms, corrected by refitting. Extra search depth supplies no visible benefit
here. Preserve the distinction between initial and refitted geometry.

## User review: GX5 and Am1-54

The user reviewed `gxBQ_window_00_frame_5` and `am1_window_00_frame_54` in
the SVD gallery. All SVD examples are judged deranged hallucinations. The GX
hallucinations nevertheless use a real court line for their long-edge support.
Saved W5 is judged great on GX5. On Am1, saved W5 looks potentially good apart
from the known net-line assignment as its far baseline. The user did not
separately specify each gallery dropdown for this review; preserve that scope.

These search arms generate automatic G0 only. Saved W5 also includes G1 and
line templates. The finding supports retaining those populations; it does not
isolate SVD screening as the cause. More search within G0 has not supplied a
visually acceptable outcome in the reviewed examples. Colour could expose
contradictory paint evidence, but a wrong geometry supported by genuine same-
colour court lines may remain colour-consistent.

## Audit decision

Opus 5-5 inspected the generator, runner and gallery. Its cap-bound finding
described an earlier code version: the final implementation reports a cap
*reached* from retained counts. Its oracle-caption defect was confirmed and
fixed. Captions, errors and accessibility labels now follow the displayed view;
oracle outlines use a distinct dash pattern. Saved selections show their source
population, because most came from G1 or line templates outside these G0 arms.

Reference-based oracles now consider only candidates in the ranker's selectable
list. Reference data still enters after ranking. Measured total time stops
before reference evaluation and excludes runtime imports and report writing.
These changes were checked against source after the audit. No defect in the
search changes was established. Broader accuracy remains an experimental question.

Scoped generation/search tests passed (9 tests, exit 0); the final search fixes
passed both focused tests (exit 0). Scoped lint and JavaScript syntax passed.
Whole-project Pyrefly passed with 0 errors and 39 existing suppressions. The
real one-pair smoke generated and refitted 256 candidates, exit 0. The rendering
fixture duplicates that one result across arms and is not experimental evidence.
The original corner magnification and stripe-overlay controls are restored.
Shared grid rows align images despite wrapped headings. An initial browser
check passed control-state assertions but missed a redraw error. The user
reported broken controls in the actual preview. Projected stripe points were
flat arrays where the renderer expected endpoint pairs; the builder now emits
centres `(12, 2, 2)` and edges `(24, 2, 2)`. All 20 geometries in the rebuilt
two-case preview pass shape/finite checks, and scoped lint/syntax pass (exit 0).
The user will do the visual check; no further browser checks are requested.
These display changes do not alter the frozen compute revision.

The preview is served at `http://127.0.0.1:8879/`. The native
`/root/svd_run_monitor` delegate completed all transfers and released its Carmack
connection. GPT-6 Sol medium completed the final gallery build and preview refresh
using the existing builder (exit 0). The frozen compute run is complete.

## Implementation record

Scope: six frozen views, three independent SVD12 automatic G0 arms. Only axis
enumeration and per-pair/global shortlist depths change. W5 fitting and ranker
are reused directly. The gallery reports the detector's choice separately from
reference-agreement oracles. No remote execution in this worker session.

Implementation: optional generator caps default to 512/256/256. Each case-arm
gets a fresh generation record and compressed candidate summary. The runner
records wall and CPU time separately for preparation, generation and W5 measurement/refit/ranking.
The per-pair raw and retained counts, axis exclusions and cap-reached indicators
show where limits matter. W5 uses the prepared-measurement path from wider evaluation.
The generator builds the old pre-gate array sidecar only when `pool_path` is supplied;
this run does not supply it. Candidate selection and saved generation fields are unchanged.

Axis exclusions come from the matcher's own diagnostics. Retain deduplication also
removes courts, so the court-cap indicators report when a cap is reached rather
than attributing all raw-to-retained loss to that cap. Fresh direction estimation is
timed as preparation and included in total time.

Final bounded fixes, 2026-09-23: The generated and refitted reference oracles
consider only candidates in the ranker's `provisional_rank`. The runner keeps
every parent and child geometry for diagnosis. Total timing stops immediately
after ranking, before reference loading and evaluation. It includes view
preparation, fresh direction estimation, generation, measurement, refit and
ranking. Module setup and reporting/serialisation are outside that interval.
The two input JSON records are now `.json.gz`; parsed contents were checked
equivalent before removing the superseded plain files. Gallery captions,
titles and aria labels follow the selected view, including with outlines off.
The saved W5 panel shows its selected key and population. Oracle outlines use
a distinct dash pattern.

Checks: scoped Ruff exit 0; focused pytest exit 0 (2 passed, including an
ineligible closer oracle candidate); gallery fixture build with `--allow-smoke`
exit 0 (one case); generated gallery JavaScript `node --check` exit 0;
whole-project Pyrefly exit 0 (0 errors, 39 suppressed). Chromium was present,
but headless startup failed in this sandbox while creating its user-data
container or crash reporter socket. This was the initial implementation check;
subsequent checks, the real smoke and the user's preview review are recorded
above.
