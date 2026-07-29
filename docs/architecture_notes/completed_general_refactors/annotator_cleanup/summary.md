# Annotator cleanup: summary

The cleanup chain trimmed the annotator to one canonical path per problem
class and pinned the current live calibration. It landed on
`feature/commentary-scraper` as thirteen commits between `6ebf9cc`
(2026-07-28) and `93477bd`, plus one post-cleanup archive commit `9f8b59f`
that only moves retired trial scripts and does not change runtime behaviour.

## What ships today

- **Batch through `run_video`.** The batch command is a thin wrapper around
  `annotator.run_video.run_video`. Pose-gated flags, `--thresholds`, and the
  rally-span sticky fallback are gone; the surviving batch is ungated and
  court-optional (`src/annotator/run_video.py`, W2.6).
- **W2.9 defaults live in `BaseAnnotatorConfig`.** Missing shuttle-track
  periods now enter a gap state with a two-sided re-entry guard, and
  invisible-frame coordinates no longer contribute to smoothing. Shipped
  values: `SmoothingMode.IGNORE_INVISIBLE`, base-30 demotion bound `75`,
  `ReentryGuardVariant.TWO_SIDED`, re-entry buffer `0.05`
  (`src/annotator/config.py`).
- **Three-pair `CourtGeo`.** The four-pair `CourtBox` is gone; every fixture
  carries `court_geo = (x_range, y_range, net_band)` (W2.7,
  `src/annotator/calibration/fixtures.py`,
  `src/annotator/rally_segmentation.py`).
- **Canonical fixture names.** Local identities are `sset_01`, `sset_15`,
  `sset_21`. `Fixture.name` derives every operational and pin path (W3.1,
  `src/annotator/calibration/fixtures.py`).
- **One tracked calibration reference.**
  `tests/data/annotator_calibration/reference/` holds the live
  three-fixture capture produced by
  `annotator.calibration.gt_scoring --capture`. It includes the aggregate
  output, per-rally scores, and diagnostic CSVs under canonical fixture
  names.
- **Inpaint sidecar contract.** `docs/tracknet/inpaint_sidecar.md` is the
  producer contract for the fill-mask sidecar shipped in commit `9475036`;
  see `docs/tracknet/inpaint_sidecar_consumption.md` for consumer state.

## What retired

- ~4,268 lines of the frozen Stage 8 sweep, scorer, and wrist analysis
  scripts, with their tests (W2.1).
- `stage8_rally_segmentation.py` and the four historical trial scripts
  moved to `scripts/archive/` (W2.2).
- Five Stage 2 compatibility shims and their re-export block (W2.3).
- The old serve-start / wideshot API surface; retained
  `_serve_distance_ratio_passes`, `court_scale_boxes`, and
  `court_scale_slots` (W2.4).
- The direction-change threshold and `pilot_geometry.py`; the impulse rule
  is the only remaining contact detector (W2.5).
- Four output-path constants and eight point-winner dependencies relocated
  into annotator-owned modules; `scraper.config` re-exports the same objects
  (W2.8).

## W2.9 behavioural delta

The paired W2.9 flip is the one commit in this cleanup that moves numbers.
`w2_9_delta.diff` is the exact `diff -u` between the pre-flip capture
(`s5_post_stage5_scores.txt` as of 2026-07-27 14:42) and the
post-flip capture (`/tmp/capture_w2_9.txt` as of 2026-07-28 21:24).

- **Contact F1 improved on all three fixtures** (sset_01 0.6483 → 0.6572;
  sset_15 0.5605 → 0.5707; sset_21 0.4723 → 0.4780). Contact recall and
  precision each move in the same direction, and the raw contact count
  falls on every fixture.
- **Downstream movements were mixed.** Landing lifted on sset_01 (0.274 →
  0.301) and sset_15 (0.230 → 0.250); sset_21 landing was unchanged
  (primary 0.243, covered 0.340). Player attribution lifted on sset_01
  (0.522 → 0.549) and sset_15 (0.356 → 0.375) but **fell slightly on
  sset_21** (0.387 → 0.373). Getpoint lifted on sset_15 (+0.035) and
  sset_21 (+0.027) but **fell on sset_01** (server 0.566 → 0.504,
  getpoint 0.509 → 0.420).
  Attribute the movement to the two paired defaults (gap-state re-entry
  guard, invisible-frame smoothing); sset_01's server/getpoint drop is the
  known trade-off, and sset_21's small player-attribution dip is a second,
  smaller one.

Do not read the delta as causal beyond the paired default change: the
cleanup ran no ablation of the two defaults separately, and W2.10's re-pin
was based on the joint capture only. Commit `85b8751` refreshed the former
scratch reference to the post-flip numbers immediately after `eee3e29`
shipped the defaults. The tracked reference now lives under
`tests/data/annotator_calibration/reference/`.

## Historical harness boundary

The old end-to-end yardstick, GT-anchored point-winner pin, and S28
sticky-anchor pin harness are historical evidence. Their scripts and four
frozen CSVs sit under `scripts/archive/autoseg_trials/` with unchanged
bytes and recorded checksums. The S28 harness
will not import at the current tip; W2.1–W2.3 removed the scorer, shim, and
`CourtBox` surfaces it depends on. The owner ruled its four outputs frozen
historical evidence with no gate value, so W2.9 measured through the live
calibration capture instead.

A clean three-mode GT-injected regression harness is a separate TODO. Its
design is at `docs/scraper_pipeline/annotator_regression_harness.md`. It
does not replace, rebuild, or repair the S28 harness.

## Final gate

The cleanup tip `93477bd` passed Ruff 0, Pyrefly 0 (7 pre-existing
suppressions), pytest 1,267 passed / 20 skipped / 33 warnings, calibration
merge smoke, and a live-capture byte-equality check against the
then-current scratch reference. All 27 canonical external data files
retained their pre-move MD5s across the W3.1 rename. The follow-up archive
commit `9f8b59f` (frozen autoseg trials) also passed the same gate.

## Reproducing the current live capture

    ANNOTATOR_FIXTURES_ROOT=local_scratch/autograder_architecture \
        PYTHONPATH=src python -m annotator.calibration.gt_scoring --capture

Compare the resulting aggregate and per-rally scores with
`tests/data/annotator_calibration/reference/`. Treat metric changes as
behavioural changes to explain; byte identity is only a provenance check
when confirming an unchanged capture.
