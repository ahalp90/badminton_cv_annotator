# Archived autoseg trials

Historical S28-S29 measurement records. These scripts are not current tooling
and do not run on the cleaned tree without restoring retired compatibility
paths. Do not repair, rerun, or re-pin them.

## Retained files

- `s28_sticky_measure.py`: four-cell sticky measurement.
- `s29_sweep_measure.py`: S29 threshold sweep.
- `s28_sticky_pin_anchor_picks.py`: frozen sticky-anchor comparison harness.
- `s28_sticky_pin_anchor_picks_outputs/`: four pre-cleanup per-rally CSVs for
  radius 7 and 9 on pilot and video 15.

The CSVs are the recoverable row-level evidence behind the MD5 values in
`s28_sticky_pin_anchor_picks.py`. They preserve the rally classifications,
mapped spans, stroke counts, timing errors, player/server calls, hit-height
counts, point winners, and landings produced by the historical chain. The
hashes prove identity; the CSVs preserve the results that can still be
inspected and compared.

Ongoing regression checks use the live three-fixture calibration capture.
Related older support scripts remain in the parent `scripts/archive/`
directory.
