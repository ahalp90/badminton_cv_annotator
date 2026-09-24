# Red-team review: court-detector speed patches

Scope: active W5 import paths and repo callers, plus small synthetic comparisons. This was a single-reviewer source audit; it does not independently repeat the supplied full-pool experiment. Reproduce the checks with `~/.venvs/badminton-cicd/bin/python redteam/reproduce_exactness.py`.

Relevant output:

```text
C1 valid [True, False] old [1.0, 0.0] [1.0, 0.0] new [1.0, nan] [1.0, nan]
C1 valid-row bits True
C1 empty shapes [(0,), (0,)]
C1 16-row subset bits True
C2 24 interval comparisons: shapes, dtype and bytes identical
C2 NaN/inf and negative-zero input bits True
C2 measure bytes True
C4 repeated map bytes True
```

## C1 — REFUTED

The accepted-candidate mask remains the same in the tested case, but the claim covers *every consumer* of the fraction arrays. The patch fills geometry-invalid rows with NaN (`scratch/court_det_fix/next_steps_20260916/webui_seed/source/run_given.py:151-159`); the old call measured all transforms. The reproduction gives `valid=[True, False]`, old fractions `([1,0], [1,0])`, new fractions `([1,nan], [1,nan])`. Its one valid row and 16-row subset are byte-identical; both empty forms have shape `(0,)`. `zone_net.player_fractions` applies a batched inverse and per-row einsum (`scratch/court_det_fix/frozen_helpers_20260914/legacy/zone_net.py:38-46`), so these checks also exercise the suspected batch-size risk without finding one.

This change reaches a real diagnostic consumer. `automatic_generation.generate` passes both arrays to `write_pool` (`scratch/court_det_fix/w5_holistic/automatic_generation.py:119-126,145-148`), and `write_pool` saves them in the NPZ (`scratch/court_det_fix/next_steps_20260916/webui_seed/source/run_automatic.py:116-129`). The pregate-loss reader reports the nearest **combined** court's fractions even when that court fails geometry (`scratch/court_det_fix/line_identity/prior_checks/pregate_loss/analyse.py:139-154,168-179,209-220`). Concrete failure: an invalid nearest combined court with previously measured zero fractions is now reported as `nan`; the diagnostic loses the distinction between measured absence and unmeasured geometry failure. The equivalent reader under `scratch/court_det_fix/evidence/direction_search/diagnostics/pregate_loss/` has the same use. `run_given.run_case` consumes candidates and role records, rather than the fraction arrays (`run_given.py:197-209`), so this finding does not establish a changed accepted court.

## C2 — CONFIRMED

`interval_evidence` keeps the original `(3, samples, fragments)` forward shape, `(fragments, fragment_samples, 3)` reverse shape and Boolean resolvable shape (`experiments/annotator/independent_court/stripe_observations.py:76-91`). Its skipped columns receive the same positive `0.0` and `inf` that the former `np.where` masks supplied. `distances_to_segments` performs operations independently for each point/segment pair, reducing only the two coordinate components (`experiments/annotator/independent_court/assignment.py:115-121`). The script compares the former expression against the patch byte-for-byte in 24 interval cases, including zero fragments, no compatible fragment, custom centres and nonzero boundary tolerance; it also checks NaN, inf and negative-zero inputs and a full `measure` result. All match, including dtype and shape. `measure` passes both optional arguments through (`stripe_observations.py:94-126`); inspected non-default callers include W5 `verifier.py:605-608` and `run_paint_refit.py:98-104`. No patch-specific defect was established.

## C3 — CONFIRMED

Pool `stripe` and `profile` are created in `run_automatic.py:78-93`; their generator winners are written at `automatic_generation.py:145-159`. W5 validates winner IDs only for membership (`scratch/court_det_fix/w5_holistic/run_w5.py:281-288`). Duplicate compatibility compares corners, metadata and gate fields, not these scores (`run_w5.py:454-467`). `compact_legacy` copies the scores into report metadata (`run_w5.py:418-426`), and comparison arm A reads them (`scratch/court_det_fix/w5_holistic/verifier.py:651-687`). Parent selection evidence is freshly computed from its homography (`run_w5.py:644-655`; `verifier.py:598-619`); ranking C reads that evidence and historical gates (`verifier.py:762-805`). The bounded selector then reads C's order, full-court gate and evidence score (`scratch/court_det_fix/net_recovery/bounded_trial.py:138-153,170-188,95-112`). Thus changing the pool's legacy score fields alone has no path to the accepted choice.

## C4 — CONFIRMED

`attempt_refit` rebuilds maps at `scratch/court_det_fix/w5_holistic/run_w5.py:742-755` from `context.segments` and `context.size`, not child geometry. The same context is passed throughout the parent/refit loop (`run_w5.py:900-924`), after `prepare_view` sets those inputs once (`scratch/court_det_fix/w5_holistic/verifier.py:209-239`). `_wide_line_families` and `_distance_maps` depend only on these inputs (`experiments/annotator/independent_court/detector.py:173-177,253-261`); the small two-family repeat in the script produced identical bytes. Computing once per unchanged view is therefore exact.

## Worklog

- `git status --short`, `git diff -- <two patched files>` — exit 0.
- Batched `rg -n` searches for callers, score fields, pool readers, and map builders; `nl -ba ... | sed -n ...` source reads — exit 0.
- `rg --files | rg 'courtkeynet/court_corners.py$|camera_diagnostic.py$'` — exit 0.
- `~/.venvs/badminton-cicd/bin/python redteam/reproduce_exactness.py` — first run exit 1 (`courtkeynet` path absent); after adding `src` to `sys.path`, exit 0, with the results quoted above.
- `git status --short` — exit 0; only the two supplied patch files and `redteam/` appear.
- Final `~/.venvs/badminton-cicd/bin/python redteam/reproduce_exactness.py`, `git diff --check`, `git status --short`, and `wc -l redteam/CODEX_REDTEAM.md redteam/reproduce_exactness.py` — exit 0 each.
