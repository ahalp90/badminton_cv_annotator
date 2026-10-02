# SVD matcher runtime comparison

## Resume

Read [the consolidated SVD state](README.md) first for the current decision,
evidence and separate search-depth experiment. This file holds execution detail.

The 12-family SVD screen is integrated into fresh experimental W5 generation.
Default budget is 12; explicit 16 preserves the full-family comparator. Frozen
helpers and seed snapshots are unchanged. Original directions and pair/candidate
IDs remain intact. G0, G1, line templates, fitting and matching depth remain.
The code review and bounded real-matcher check pass. Detailed checks are below.

**The original nine-case Carmack benchmark and local aggregation are finished.**
The detached coordinator, PID 2653088, completed at revision `7935ce1`.
Its run root was `/scratch/ahalperi/court_det_fix/svd_runtime_20260923`;
`full.exit` records exit 0. Six workers used one numerical thread each.
All nine raw cases and the summary are under local `full/`. There is no
remaining SSH-owned delegate or benchmark job to monitor.

The final [timing report](RESULTS.md) links the compact
[`results.json.gz`](results.json.gz) and
[`history_check.json.gz`](history_check.json.gz) receipts. The existing
`check_history.py` and `summarise.py` each exited 0. Historical comparisons
passed for 2,160 pairs and shared-arm comparisons passed for 1,188 pairs;
there were zero mismatches. Raw outputs remain local artefacts. The core
integration remains `1353541`. Matcher timing excludes full scoring and
scene processing.

Real SS03-19 integration smoke: first two original ordered pairs, real matcher
and full scoring. Live16 equals the saved generator exactly except timing and
new metadata. SVD12 skips pair 0 and retains pair 1 and approved winner `1:60`
for both rankings; both arms shortlist 256 courts. The retained pair has 47
small numeric differences, largest 8.47e-11 (shortlist score), below the existing
1e-8 tolerance. Receipt: `integration_smoke.json.gz`; rerun: `smoke_integration.py`.
The initial strict-equality probe failed on that numerical variation; its
repeat preserves raw outputs before comparison.

## Scope and concerns

- Measure the unchanged matcher and camera-direction gate, with original
  family and pair IDs. Keep direction normalisation, scoring and caps fixed.
- Measure SVD ranking overhead separately. This is matcher-stage timing;
  image loading, full evidence scoring and scene processing are excluded.
- Compare shared-pair outputs across arms. Keep the 16-family comparator and
  G0 fallback. The initial benchmark keeps the matcher fixed; the subsequent authorised integration is recorded below.
- Cached winner substitutions are mixed: GX5 line-reference error improves
  530 to 352 working pixels; GX5 paint error worsens 531 to 1,652; Am2-28019
  paint error improves 892 to 17. These are not fresh visual judgements.
- Run case pairs sequentially within each worker, alternating arm order.
  Up to six workers, one native numerical thread each. Record CPU time as
  well as elapsed time because shared-host scheduling may affect timing.

## Modules and execution

- `run_benchmark.py`: new diagnostic runner only; delegated to Sol medium,
  default service tier. Reuses frozen matcher functions.
- Existing fitting and detector modules remain unchanged.
- Carmack connection succeeded with network access enabled after sandbox DNS
  resolution failed. No alternate host was used.
- Existing branch authority allows a checkpoint commit and push for remote
  execution. Planned header: `Measure SVD pruning before changing search`.
- Sol's syntax/import, nine-case SVD rank and Ruff checks pass. Whole-project
  Pyrefly passes (0 errors, 39 suppressed). The SS03-16 smoke
  completes matching and fresh-arm comparison, then fails the historical
  cache check at pair 1, horizontal axis entry 156 (axis ID 2283 versus 2293).
  This ordering difference needs diagnosis; the comparison tolerance is unchanged.
- Opus is auditing runner correctness and, at the user's additional request,
  simple compute-efficiency opportunities in the called matcher. Both are
  read-only and use `claude-opus-5-5` at high effort.
- Carmack's two-worker smoke on SS03-16 and SS21-20 passes, exit 0, 37.1 s
  batch wall time. All four full-arm pairs also match historical cache records
  in a separate comparison, exit 0. The local SS03-16 difference does not
  reproduce in the Carmack environment. Native thread counts are one.
- Opus runner audit found no verified defect. It independently replayed three
  SS03-19 pairs against the local cache. Its useful limitations: host contention
  can affect elapsed-time ratios, historical mismatch before writing loses a
  result, and matcher-only timing excludes later scoring. The full Carmack run
  will preserve its results first and compare the historical cache afterwards.
  Record shared-pair timing ratios alongside overall savings.
- Runtime runner committed and pushed as `394b9ed`. Remote checkout is
  `/scratch/ahalperi/court_det_fix/svd_runtime_checkout_20260923`; run outputs
  are under `/scratch/ahalperi/court_det_fix/svd_runtime_20260923`.
- Launch script and smoke receipt notes committed/pushed as `7935ce1`.
  Carmack has 32 logical CPUs and 375 GiB RAM; six-worker memory is adequate.
- Opus compute audit completed on the requested model. See
  [the assessed recommendations](COMPUTE_AUDIT.md). Geometry-first player
  filtering and a two-endpoint maximum are the smallest high-value leads.
  The player-projection arithmetic rewrite needs remote boundary checks and
  canonical constants. No compute optimisation has been applied.
- User queried the earlier repeated-greyscale-conversion issue. The later
  wider-evaluation and polarity wrappers still use `prepared_measurements`,
  which converts once per immutable image. This benchmark calls neither that
  scoring stage nor any image decoder/converter: preparation reads embedded
  segments and player feet. Greyscale conversion cannot explain its timings.

## Authorised integration — 23 September

The user has decided to integrate the 12-family SVD screen and asked the current
nine-case run to finish. Matching depth remains a separate adjustment. The four
completed broadcast cases pass all 960 historical pair comparisons at the
existing 1e-8 absolute tolerance; the partial receipt records exact versus
floating-point-only matches. Am2-150 has also completed (1,847.342 s full16,
894.338 s svd12, 0.00555 s ranking), pending retrieval/history comparison.

Plan: add an owned `w5_holistic/automatic_generation.py` and route fresh G0/G1
generation through it. Current W5 imports its automatic generator from the
saved WebUI seed source; there is no integrated scene detector in `src/` yet.
The new generator reuses the existing matcher and scoring helpers. SVD ranks
support families, while original directions and pair/candidate IDs remain
intact. Default budget is 12; explicit 16 preserves the full-family comparator.
Population and final-result caches must reject a mismatched budget.

OUT-list: frozen helpers, saved seed sources, archived evidence and the running
remote revision stay unchanged. No fitting, acceptance, scoring, matching-depth
or compute-audit changes in this batch. G0 and line-template proposals remain.
The user accepted the known score-winner substitutions; broader accuracy
improvement is not claimed. Silent changes on surviving pair inputs are not
acceptable.

Sol medium/default owns the new generator, wider-evaluation adapters and focused
checks. Parent owns records/results. Check nine saved rankings, original IDs,
16-family bypass, small populations, cache identity and adapter wiring; run
relevant lint/types and an actual matcher smoke, then bounded Opus code review.
Planned authorised branch checkpoint: `Reduce court matching with an SVD family screen`.
Final benchmark report follows all nine completed cases and history comparisons.

Integration review: Sol's initial eight focused tests passed. Parent checks
found two real boundaries those tests missed: importing `generation` before
`load_runtime` broke a fresh `run_case` process; an estimator with zero lines
serialises its arrays as `[]` and failed the new shape check. A bounded Sol
follow-up fixes both with regression coverage. The SVD coefficient transform
is into normalised coordinates; the initial worker's prose called it working
coordinates, so its variable name and report wording need correction.

Timing caveat: identical retained pairs show wall-time ratios of 0.600, 1.038,
0.841 and 1.068 (SVD arm/full arm) across the four broadcast cases. Full-arm
work fractions retained are 0.376, 0.439, 0.553 and 0.379 respectively. Report
both these within-full-run fractions and independent-arm timings. The direct
77% reduction on SS03-16 contains substantial between-pass timing variation.

The ordinary Pyrefly profile excludes scratch. An explicit changed-file check
initially lacked the runtime's added import paths; adding `w5_holistic` and
`wider_evaluation` to that check's search paths passes (exit 0). No project
configuration was changed. Whole-project types will also be run at close-out.

Independent integration review: `claude-opus-5-5` confirmed by modelUsage.
Opus found no verified screen/pair-loop defect, reproduced all nine ranks and
traced original IDs and camera-bound skips with a real estimator. Its first
trace stubbed the expensive matcher/scorer; the parent real matcher check is
separate. The historical `wider_evaluation/run_remote.sh` wrote new defaults
into a fixed old run directory. It now explicitly forces the original 16-family
budget; fresh SVD runs use `run_cases.py` with a new output root. Shell syntax
passes. Budget metadata in population and result files is sufficient for current
resume checks; case records link those population files rather than duplicate
that metadata. The remaining repeated method string is a small maintenance
lead, not a runtime defect.

Post-fix W5/adapter checks: 60 passed, exit 0. Whole-project Pyrefly: 0 errors,
39 suppressed, exit 0. Whole-project pytest: 2,309 passed, 29 skipped and one
failure (exit 1): `resolve_interpreter("python")` could not find `python` on the
unactivated shell PATH. Re-running that one test with the project venv on PATH
passes, exit 0. Whole-project Ruff: 1,021 diagnostics outside the changed files,
exit 1; full output retained in `/tmp/svd-project-lint.log`. Scoped lint passes.

Six completed remote cases now pass all 1,440 historical pair comparisons and
all 792 retained-pair comparisons at the existing 1e-8 absolute tolerance.
No mismatch. Am3-0: 2,427.837 s full16 versus 1,013.485 s svd12, plus 0.00664 s
ranking. Three cases remain. Updated partial receipt stays explicitly partial.

Integration checkpoint `1353541` is committed and pushed to `fix/court-det`.
The remote benchmark remains pinned to `7935ce1`. An additional bounded check
freshly estimated directions from the three frozen source packs: all 47 unique
views produced 16 families and passed both budgets' ID/count/unchanged-estimator
checks. No matcher or image scoring ran. Receipt: `saved_input_check.json.gz`.
This is input compatibility, not accuracy or broader candidate-retention evidence;
it supplies no real non-empty case below 16 families.

Token-cost steering: the user asked to avoid unnecessary Astra machinery.
Monitoring and aggregation were delegated to GPT-6 Sol medium/default under
`local_scratch/external_delegate/20260923-svd-finish/`. That attempt stopped
because the worker's sandbox could not resolve the SSH host. It did not alter
the remote job or aggregate results. The parent now handles remote transport;
there is no Sol-owned SSH connection. The original GX0 SVD arm has since
completed; all nine cases were aggregated locally. Integration is already
pushed as `1353541`.

The user has separately authorised a six-case search-depth experiment and
an Opus 5-5 implementation audit. Its code and current checks are in
`../svd_search/`; see the consolidated README for the three configurations.
Do not mix those results with this unchanged-matcher timing run.
