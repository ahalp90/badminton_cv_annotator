# Archived review material

Start with
[`../readability_refactor_scoping_handoff.md`](../readability_refactor_scoping_handoff.md).
This directory preserves raw inputs and review machinery that support an
evidence check but are not needed for normal refactor scoping.

## Path map

- `worklog.md` moved to `archive/worklog.md`
- `packets/*__scans.json` moved to `archive/raw_packets/`
- `packets/*__probe.json` moved to `archive/raw_packets/`
- `build_codex_briefs.py`, `build_workflow_run.py`, `extract_journal.py`,
  `run_codex_batch.sh`, `validate_batch.py` and
  `workflow_template_adapted.js` moved to `archive/review_pipeline/`
- `report.md` became `readability_review_evidence_report.md` in the parent
  directory

The worklog remains byte-for-byte intact as an audit trail. Its references use
the old layout, and its summary contains an original tallying error. The 33
verdict files contain 398 rows: 275 `CONFIRMED`, 95 `KILLED` and 28
`MISREAD_BUT_TELLING`.

The archived pipeline scripts describe how the review was run. Their relative
paths have not been rewritten into a polished rerun harness.

## Keep this folder navigable

- Keep `readability_refactor_scoping_handoff.md` as the sole entry point
- Add new evidence only when a live document points to it
- Keep generated briefs, logs and worker output outside this folder
