# Dead, parallel and duplicate code audit plan

The audit looked for dead code, parallel implementations of the same problem,
and near-duplicates where a small change would let one implementation replace
another. It covered `src/` and `scripts/`. The audit was read-only and made no
source-code changes.

The final plan used separate work-package ownership, a production owner for
CourtKeyNet, distinct dead-code categories, and a root manifest for liveness
claims. An independent plan review on 2026-08-01 led to those changes.

## Scope

Included:

- src/annotator, src/api, src/bric, src/bst_x, src/courtkeynet, src/scraper, src/shared
- scripts/ excluding scripts/archive
- tests/ as evidence (duplicated helpers; test-only production symbols as a
  classification, not a delete list)

Excluded:

- docs/**/*.py, one-shot analysis and evidence scripts that were deliberately parked
- scripts/archive, already archived; WP7 runs one explicit tracked-content
  search for imports, path loads, or subprocess calls into it, nothing more
- Internal style of vendored code (src/bric/perception/_vendor,
  src/courtkeynet/_vendor, src/bst_x/TrackNetV3). Duplication BETWEEN mirrors is
  in scope; line-level critique of upstream vendor code is not.
- data/, experiments/ artefacts

Liveness universe: tracked repository content. Untracked or ignored callers are
out of the audit's world by definition; the report states this limit once.

## Root manifest (liveness roots; verified 2026-08-01)

A zero-caller claim only counts if the symbol is unreachable from every root:

- CI (.github/workflows): pytest suite; scripts/pr_main_files.py;
  scripts/pr_advisory.py
- Docker: uvicorn src.api.main:app. Every FastAPI route or handler in src/api is live
- Documented `python -m` modules (git grep over tracked *.md/*.py/*.sh/*.yml/*.toml),
  including: pipeline.data_access, pipeline.build_dataset, pipeline.shuttle_extractor,
  pipeline.download_videos, pipeline.clip_generator, pipeline.verify, hparam_sweep,
  bst_x_train, bst_x_infer, collation_runner, preparing_data.prepare_train_on_shuttleset,
  preparing_data.apply_heuristic, preparing_data.raw_extract, bric.train,
  bric.smoke_test, bric.preprocessing.{preprocess_videos,slice_rallies,extract_shuttle},
  bric.diagnostics.{validate_rgb,evaluate_shuttle}, validation_scripts.* (several),
  scraper.stage{1,2,3,11}_*, annotator.calibration.gt_scoring,
  annotator.validation_overlay.overlays.shuttle_track, scripts.build_shots_master
- Modules with a `__main__` block but no documented invocation anywhere tracked:
  report as "runnable, undocumented" evidence, not automatically dead
- conftest.py fixtures and tests-by-name selection; pre-commit hooks; getattr /
  importlib / registry / `__all__` string dispatch

## Finding categories

- **D-unreach**, unreachable from every root above, static and dynamic. The only
  category that supports "delete" outright.
- **D-prod**, no production reference; kept alive only by tests or by nothing.
- **T**, test-only surface (helpers, fixtures, production symbols only tests
  touch). A classification for a maintainer ruling, not a delete list.
- **U**, unused surface inside live code: parameters never passed, write-only
  state, dead branches, stale config knobs.
- **P**, parallel implementations of one problem class, with matched inputs,
  outputs, invariants, and side effects shown. Unproven equivalence stays C.
- **S**, supersedable near-duplicate: matched contract plus the named small tweak
  that lets one side absorb the other, and which side is better with the losing
  side's concrete deficiency stated.
- **C**, comparison note: similar code where equivalence is unproven or the
  divergence looks intentional (e.g. a mirror preventing coupling). States what
  coupling the duplication currently prevents.
- **O**, overengineering per .github/AGENTS.md, admitted only with a concrete
  maintenance, comprehension, or correctness cost stated. Module size and
  caller counts are prompts for inspection, not violations.

Evidence bar: every finding names file:symbol, the roots and call sites checked,
and a proposed disposition (delete / absorb into X / park as archive / leave,
one-line reason). Findings without evidence are marked UNVERIFIED and stay out
of the report.

## Work packages

Every directory and cross-package comparison had one owner. Cross-boundary
notes were routed to that owner's ledger entries during the merge, which
prevented duplicate findings. Each work package was an automated read-only
sweep.

| WP | Owns | Focus |
|----|------|-------|
| 1 | src/bric/perception/_vendor, src/bst_x/TrackNetV3, src/courtkeynet/_vendor | File-by-file mirror diff (bric vendor vs bst_x TrackNetV3); which side is newer; exact surface each consumer calls; minimal live surface per mirror |
| 2 | All cross-package comparisons (no directory) | Cross-package P/S/C: bst_x/pipeline vs scraper stages; court_utils vs shared/court vs courtkeynet/court_corners; player_mapping pair; shuttle_extractor vs bric extract_shuttle; bst_x's non-use of src/shared incl. legacy taxonomy names; duplicated data contracts, schemas, path/config resolution, CLI orchestration, validation logic and repeated constants |
| 3 | src/annotator + src/courtkeynet (production code) | Internal D/U/T/P/S/O; calibration/ and validation_overlay/ included; courtkeynet wrapper/constants/court_corners liveness |
| 4 | src/scraper + src/shared | Stage modules, config contracts, dead branches; shared/ internal quality and adoption map |
| 5 | src/bst_x excluding TrackNetV3 and validation_scripts | pipeline/, preparing_data/ (including their mutual overlap), model/, loss/, train/infer/reporting/run_tracker/result_utils/run_overview |
| 6 | src/bric excluding _vendor and diagnostics, + src/api | dataset/network/train/eval/perception/preprocessing; the api inference triplet (inference.py, bric_inference.py, bst_x_inference.py); bric/dataset vs shared/dataset |
| 7 | scripts/ (non-archive), src/bst_x/validation_scripts, src/courtkeynet/validation_scripts, src/bric/diagnostics | Liveness census: live tool / one-shot gate from a finished pass (archive candidate) / D-unreach; plus the scripts/archive inbound-reference check |
| 8 | tests/ | Duplicated helpers/fixtures; T-list of production symbols only tests reference |

## Verification

1. Every D-unreach, D-prod, and S finding recommended for action received a
   first-hand check. Semantic references and `git grep` over tracked content
   were checked against the root manifest.
2. Contested or refuted findings drop to a report appendix with the refutation
   noted.

## Deliverables (docs/archive/dead_code_clean/)

- audit_plan.md, this scope and method record
- decisions.md, the final refactor rulings
- worklog.md, the internal audit execution record
- findings.md, the merged ledger
- findings/wp1.md through findings/wp8.md, the raw sweep returns
- report.md, the reader-facing summary

## Order of execution

1. Review the plan. Ten points from the 2026-08-01 review were folded into this revision.
2. Run WP1 to WP8 as separate read-only sweeps.
3. Merge returns into findings.md, route outward notes to owners, and verify
   actionable findings first-hand.
4. Write report.md.
5. Run an independent final review and fold in the findings. No refactor was executed.
