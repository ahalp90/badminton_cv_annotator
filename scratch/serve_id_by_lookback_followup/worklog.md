# Worklog

## Resume

- Stage: Batch 0b planning correction in progress
- Next action: review and commit the simplified sequential-search runbook, then implement Batch 1
- Active workers or audits: none
- Worktree: `/home/ariel/Documents/COSC594/badminton_cv_annotator`, branch `investigation/serve-id-by-lookback-followup`, actual tip `caa8207`
- Last verified gate: Batch 0 staged `git diff --cached --check`; exit 0
- Blockers: none
- Critical outputs: `decisions.md`, `plan.md`, `findings.md`, `evidence.md`, `mechanisms.md`, and `00_SHARED_CONTRACT.md`

## Concerns and observations

- Scoping: the 97 unmatched rows are an analysis slice, not the population on which the rule runs
- Scoping: unavailable pre-contact evidence at the selected contact is `not enough shuttle trajectory to tell`; it cannot support a visible serve or an implied serve
- Tooling: Serena/Pyrefly is visible and active at `http://127.0.0.1:9121/mcp`
- Source: the direct post-contact run can mirror the existing strict pre-contact convention
- Ruling: scan accepted contacts chronologically and skip every contact without credible outgoing motion
- Ruling: stop at the first credible outgoing contact and classify it with the existing PR #82 incoming check
- Ruling: incoming means first visible post-serve contact; measured no incoming means visible serve
- Ruling: missing post-contact evidence and measured no-outgoing both fail the binary predicate; neither gets a separate reporting state
- Ruling: a later contact never overrides an earlier no-outgoing verdict
- Ruling: backwards tracing, contact reconnection, contact chains, cross-gap tests, the 75-frame cap, and the gap distribution are out
- Ruling: the outgoing sensitivity run is removed; only the selected contact's pre-contact check remains three-way
- History: Batch 0 committed a now-superseded reconnection plan before the final simplification

## Module state

- `scratch/serve_id_by_lookback_followup/`: scope, source findings, corrected runbook, and no Python implementation yet
- `scratch/serve_start_trajectory_exploration/`: read-only source of accepted-contact and trajectory conventions; unchanged
- `src/annotator/`: read-only; unchanged

## Readiness

- Goal, population, GT boundary, binary chronological search, three-way incoming classification, review roster, and commit authority are pinned
- `plan.md` carries the OUT-list, batch gates, reference checks, halt conditions, and exact authorised commit messages
- The direct post-contact helper, eligibility checks, and direction rule are mapped
- No connection or gap machinery remains in the planned implementation

## Execution log

### Batch 0: superseded rule sheet

- Files: `decisions.md`, `evidence.md`, `mechanisms.md`, `runs.md`, `worklog.md`, `plan.md`, and `audit_index.md`
- Change: recorded the earlier permissive positive-endpoint connection and inclusive 75-base-30fps gap
- Gate: staged `git diff --cached --check`; exit 0; fresh native review findings resolved before staging
- Commit: `caa8207 Pin the accepted-contact trace experiment`

### Batch 0b: simplified rule sheet

- Files: the eight planning and living-record files listed in `plan.md`
- Change: remove all reconnection and gap machinery; pin the sequential outgoing search and incoming classification
- Gate: fresh native review complete; three concrete findings resolved; staged diff check pending
- Commit: pending
