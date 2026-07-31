# Historical serve-miss evidence (2026-07-23)

Three CSVs measuring the GT serves the annotator's rally-detection chain
missed on each calibration fixture, produced 2026-07-23 by a scratch
script that no longer runs against the current tip. **Read as historical
evidence, not current behaviour.**

## Boundary

- **Pre-W2.9.** The measurement predates the gap-classification and
  invisible-frame smoothing defaults shipped in `eee3e29` on 2026-07-28.
  W2.9 changed contact behaviour on all three fixtures; see
  `docs/architecture_notes/completed_general_refactors/annotator_cleanup/w2_9_delta.diff`
  for the exact per-metric movement.
- **Pre-W3.1 rename.** The CSVs use the pre-W3.1 fixture stems
  (`pilot`, `vid15`, `sset21`). Canonical stems at the current tip are
  `sset_01`, `sset_15`, `sset_21` respectively; nothing else about fixture
  identity changed. See `src/annotator/calibration/fixtures.py:295-355`.
- **Producer not ported.** The scratch script that produced these files
  (`local_scratch/autograder_architecture/now_tracked/serve_prepend/serve_miss_scope.py`,
  gitignored) reads pre-W3.1 identifiers and pre-W2.9 chain outputs.
  Do not run it against the current tip; the current-chain rerun uses the
  substitute described in `local_scratch/autograder_architecture/TODO.md`
  (built on `annotator.calibration.gt_scoring.build_run_video_inputs`).

## Files

| CSV (historical name) | Canonical stem | Data rows | MD5 |
|---|---|---:|---|
| `pilot_missed_serves.csv` | `sset_01` | 64 | `8cbda6c100dd55843598ade36a4df826` |
| `vid15_missed_serves.csv` | `sset_15` | 38 | `353b4fe1f183c6d16befe1e69c99157c` |
| `sset21_missed_serves.csv` | `sset_21` | 34 | `690dab8efa1f04ab2cdb5c978f20367e` |

Total: **136 missed GT serves** across the three fixtures. All three files
use CRLF line endings on every line (Python `csv` writer dialect); do not
normalise.

Row schema: `rally_first`, `rally_last`, `serve_frame`, `n_gt_strokes`,
`n_matched_strokes`, `nearest_raw_candidate`, `nearest_accepted_candidate`,
and window-stats columns. One row per missed GT serve.

## Headline counts referenced elsewhere

- **136 misses total** (64 / 38 / 34).
- **113 of 136 have a clean visible shuttle-track run** within one second
  of the GT serve frame (53 / 35 / 25).
- **23 of 136 have no clean track run nearby** (11 / 3 / 9).
- **17 of 34 sset_21 misses sit inside the believed replay mask** and are
  unrecoverable under the shipped no-contacts-on-believed-frames rule
  without an exemption.

The `113 / 23` split and the 17-of-34 in-mask figure come from the
companion console record (`serve_miss_scope_console.txt`, gitignored).

## Rerun after W2.9

`docs/archive/serve_prepend_lookback.md` § 5 points to
the `TODO.md` rerun spec. The rerun uses the current tip, canonical stems,
and the same schema so the pre/post numbers can be compared directly. Do
not treat the counts above as current chain behaviour.

## Related tracked docs

- Archived design record: `docs/archive/serve_prepend_lookback.md`
- Measurement history: `docs/scraper_pipeline/annotator_measurement_history.md`
- Inpaint sidecar (producer contract): `docs/tracknet/inpaint_sidecar.md`
- Inpaint sidecar (consumer state): `docs/tracknet/inpaint_sidecar_consumption.md`
