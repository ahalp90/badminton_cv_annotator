# Annotator calibration reference

A three-fixture (`sset_01`, `sset_15`, `sset_21`) capture of the current annotator
calibration chain, kept for comparison during ongoing annotation changes. `manifest.json`
records the exact provenance: capture date, source commit, the command that produced
these files, every file's SHA-256/size/line count, and the fixture input digests
declared in `src/annotator/calibration/fixtures.py`.

## What can be compared

- **Per-rally rows** (`per_rally/*.csv`): one row per ground-truth rally — boundary
  classification, ball-round count, stroke-timing matches, player, server, hit
  height, landing side, and point winner (the `RallyRow` fields written by
  `annotator.calibration.gt_scoring.write_rallies_csv`). This is the primary layer
  to diff against a new capture: it is where a behavioural change first shows up as
  a changed field on a specific rally.
- **Aggregate metrics** (embedded in `aggregate_stdout.txt`): coverage fraction,
  contact F1, raw and filtered contact counts, and the other summary numbers
  currently pinned in `REFERENCE_SCORES`
  (`src/annotator/calibration/gt_scoring.py`). `flatten_metrics` produces these
  values from the full scoring result; the per-rally CSV alone does not reproduce
  every aggregate.
- **Diagnostics** (`diagnostics/geometric_verdicts/*.csv`,
  `diagnostics/rejections/*.csv`): supplementary rows that explain *why* a per-rally
  field changed — a geometric point-winner verdict, or the rule and frame that
  rejected a candidate contact. They are not semantic comparison targets on their
  own; use them to investigate a per-rally row that moved.

## What each layer proves

- The per-rally rows prove what the scoring harness concluded about each rally, in a
  form stable enough to diff row-by-row against a new capture.
- The diagnostics prove the internal reasoning behind specific per-rally outcomes:
  which rule rejected which contact, and how the geometric point-winner check voted.
- `aggregate_stdout.txt` is a convenience record of one full `--capture` run's
  stdout — the same comparison table and `REFERENCE_SCORES` literal a human running
  the command would see. Read it for a quick human check. Do not machine-diff it: it
  interleaves a comparison table and a Python-literal dump in one stream. Use the
  per-rally and diagnostic CSVs for structured rally-level comparisons, and the
  aggregate record for full-run metrics that those CSVs do not contain.

## What this reference is not

SHA-256 equality is **not** a pass/fail gate. The hashes in `manifest.json` are for
spot-checking that a specific file has not silently changed; a real calibration or
scoring change is expected to change these bytes. The source commit
(`93477bd`) and the fixture input digests are provenance too, not an equality
requirement. They name which code and which external fixture arrays produced this
capture; a future run is not required to reproduce it byte-for-byte.

For a design that isolates *why* a comparison moved — rally segmentation, contact
detection, or downstream attribution/landing/hit-height/server/winner logic — see
`docs/scraper_pipeline/annotator_regression_harness.md`. That three-mode,
GT-injected harness is not built yet; this reference set is today's evidence from
the existing production-chain command, not an implementation of that design.

## Reproducing a capture

```
ANNOTATOR_FIXTURES_ROOT=<external-fixture-root> PYTHONPATH=src \
  python -m annotator.calibration.gt_scoring --capture --out <output-dir>
```

See `manifest.json` for the exact command, the source commit this capture was taken
at, and per-file provenance.
