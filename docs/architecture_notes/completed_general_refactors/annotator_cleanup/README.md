# Annotator cleanup

This directory records the annotator cleanup chain that ran on
`feature/commentary-scraper` in late July 2026. Nothing here changes current
behaviour. It captures what shipped, what retired, what moved, and the one
paired behavioural change (W2.9) that moved the live calibration reference.

## Contents

- `summary.md` — what the cleanup achieved, what ships now, what retired,
  and where the historical harness boundary sits.
- `worklog.md` — one row per landed commit, drawn from Git.
- `w2_9_delta.diff` — exact `diff -u` output showing the W2.9 pre/post capture
  and `REFERENCE_SCORES` change.

## Provenance

- Local authoritative tip when this record landed: `9f8b59f`.
- Pushed cleanup tip: `93477bd`.
- Underlying source records live under
  `local_scratch/autograder_architecture/now_tracked/annotator_cleanup/source_session/cleanup_session_brief_20260728T1024/`
  (gitignored); this directory carries the public-facing distillation.

## Boundaries

- The retired S28 harness and its four frozen CSVs are historical evidence
  under `scripts/archive/autoseg_trials/`. Do not run or re-pin them.
- The clean GT-injected regression harness is a separate TODO
  (`docs/scraper_pipeline/annotator_regression_harness.md`).
- The live three-fixture calibration capture is
  `tests/data/annotator_calibration/reference/`, written by
  `annotator.calibration.gt_scoring --capture`.
