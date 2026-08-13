# Worklog

## Resume

- Stage: tidy committed locally; awaiting push and pull request
- Current conclusion: freeze the preferred layered server rule for unseen testing
- Active implementation work: none
- Open technical ruling: none
- Next action: push the branch, then open the follow-up pull request against
  `investigation/serve-start-trajectory`
- Branch: `investigation/serve-id-by-lookback-followup`
- Production code: unchanged

## Checked result

- Population: 239 one-to-one development rallies
- PR #82: 163 server sides, 125 visible starts, 96 joint
- Preferred rule: 170 server sides, 132 visible starts, 117 joint
- Paired server changes: 20 fixes and 13 damages
- Nearby first-contact fallback: 171 server sides, 131 visible starts, 117 joint
- Narrow high-shot correction: two visible-start fixes and one server fix

## Live files

- `README.md`: entry point and decision
- `report.md`: owner-facing narrative
- `HANDOVER.md`: fast re-entry and rerun commands
- `docs/next_steps.md`: frozen next experiment
- `docs/results.md`: detailed evidence
- `docs/reproducibility.md`: source and output trace

The complete session log is preserved in full beneath an archive notice at
`archive/original_investigation/worklog.md`.

## Final checks

- Focused Ruff check: passed
- Clean recomputation check: passed
- Investigation unit tests: 6 passed
- Mermaid render: passed
- Fresh-agent re-entry check: passed
- Whole-project Pyrefly check: passed with 0 errors and 20 configured suppressions
- Whole-project tests: 1,456 passed and 29 skipped
- Whole-project Ruff check: 661 existing findings outside this subtree
