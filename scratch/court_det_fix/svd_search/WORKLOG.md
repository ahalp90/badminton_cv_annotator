# SVD search-depth worklog

## Resume

The full six-case, three-arm experiment is running on Carmack at commit
`14310f7`. Six workers each use one numerical thread. The earlier nine-case
timing benchmark completed with exit 0 before this launch.

- Checkout: `/scratch/ahalperi/court_det_fix/svd_search_checkout_20260923`
- Run root: `/scratch/ahalperi/court_det_fix/svd_search_run_20260923`
- Coordinator PID: `2671966`; completion receipt: `full.exit`; log: `full.log`
- Results: `results/`; per-case progress: `results/generation_logs/`
- Next: collect completed results, build the gallery and judge runtime plus
  court quality. Do not relaunch without checking the existing job

The generator and runner are frozen for this experiment. Restoring the old
gallery's corner crops and stripe-overlay controls is separate display work.

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
Browser interaction remains to be checked outside the worker's restricted
Chromium environment.

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
container or crash reporter socket. Browser interaction remains unverified.
The full experiment and 256-parent refit were not run in this pass.
