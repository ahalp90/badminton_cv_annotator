# Rally-start visibility pilot decisions

`pilot_decisions.csv.gz` is the primary human-decision table for the 32-row
issue-32 pilot reviewed on 2026-08-09. It stores only the exact rally key and
human-entered decision fields.

The audit-guide builder joins these decisions to fresh source-derived pilot
rows. It does not trust duplicated GT, timeline, review-window, or stratum
columns from the working spreadsheets. The generated per-video reviewed files
live in `../rally_start_visibility_audit_20260809/`.

The decision contract is:

- `visible`: contact is observable and `visible_serve_frame` is required;
- `broadcast-omitted`: contact is not broadcast and both return markers are
  required;
- `off-frame`: current-rally service action is shown but contact falls outside
  the camera image;
- `uncertain`: the footage does not support a resolved outcome.

`off-frame` and `uncertain` leave all marker fields blank. An off-frame decision
uses certain confidence. An uncertain decision uses uncertain confidence. All
reviewed rows retain a human note.

The three uncompressed working files and raw Google Sheets exports remain under
ignored `local_scratch`. They are not generated evidence and must not replace
this validated table.
