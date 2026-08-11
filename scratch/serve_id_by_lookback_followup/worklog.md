# Worklog

## Resume

- Stage: investigation complete; final result commit staged
- Next action: commit the audited result batch and hand off the report
- Active workers or audits: none
- Worktree: `/home/ariel/Documents/COSC594/badminton_cv_annotator`, branch `investigation/serve-id-by-lookback-followup`, actual tip `d9b50d5`
- Last verified gate: Batch 2 staged `git diff --cached --check`; exit 0
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
- Source: the 239-rally population crosswalk is GT-derived, but the search receives a projection with no GT frames, labels, boundaries, or truth tables
- Source: the follow-up deliberately adds the rally span bound to PR #82's tracker-scene path bound

## Module state

- `scratch/serve_id_by_lookback_followup/`: implemented helper and driver, checked 239-row evidence, audited report, and living records
- `scratch/serve_start_trajectory_exploration/`: read-only source of accepted-contact and trajectory conventions; unchanged
- `src/annotator/`: read-only; unchanged

## Readiness

- Goal, population, GT boundary, binary chronological search, three-way incoming classification, review roster, and commit authority are pinned
- `plan.md` carries the OUT-list, batch gates, reference checks, halt conditions, and exact authorised commit messages
- The direct post-contact helper, eligibility checks, direction rule, analysis driver, and compressed evidence are complete
- No connection or gap machinery remains in the planned implementation

## Batch 1 checks

- Focused tests: exit 0; 17 passed
- Ruff: exit 0
- Pyrefly 1.1.1 with the scratch search path: exit 0; 0 errors
- Serena diagnostics: no diagnostics in the helper or tests
- Serena reference lookup returned no indexed callers; text search confirms the focused test call flow
- `git diff --check`: exit 0
- Fresh native review found three API and test gaps; all were fixed
- Fresh native re-review: `CLEAN`

## Execution log

### Batch 0: superseded rule sheet

- Files: `decisions.md`, `evidence.md`, `mechanisms.md`, `runs.md`, `worklog.md`, `plan.md`, and `audit_index.md`
- Change: recorded the earlier permissive positive-endpoint connection and inclusive 75-base-30fps gap
- Gate: staged `git diff --cached --check`; exit 0; fresh native review findings resolved before staging
- Commit: `caa8207 Pin the accepted-contact trace experiment`

### Batch 0b: simplified rule sheet

- Files: the eight planning and living-record files listed in `plan.md`
- Change: remove all reconnection and gap machinery; pin the sequential outgoing search and incoming classification
- Gate: fresh native review complete; three concrete findings resolved; staged diff check exit 0
- Commit: `977f456 Simplify the accepted-contact opener rule`

### Batch 1: sequential helper

- Files: `accepted_contact_trace.py` and `test_accepted_contact_trace.py`
- Change: binary outgoing selection, deferred three-way incoming classification, and focused boundary tests
- Gate: focused tests, Ruff, Pyrefly, Serena diagnostics, diff check, and fresh native re-review all pass
- Commit: `a0f3a33 Add the sequential accepted-contact opener search`

### Batch 2: checked analysis

- Files: `analyse_accepted_contact_trace.py` and expanded `test_accepted_contact_trace.py`
- Change: build 239 GT-free projected search rows, join GT only for scoring, and write/check deterministic compressed evidence
- Gates: 20 tests pass; Ruff passes; Pyrefly 1.1.1 reports 0 errors; full check mode rebuilds and matches 239 rows
- Review: one structural GT-boundary finding fixed with `SearchInputs`; re-review `CLEAN`
- Commit: `d9b50d5 Build the accepted-contact trace analysis`

### Batch 3: checked result and report

- Files: compressed 239-row evidence, compressed summary, report, and living investigation records
- Result: at +/-10, 16 fixed, 34 damaged, 62 unchanged, 100 pre-contact unknown, and 27 no credible contact
- Audits: Claude Opus 4.6 Thinking `PASS`; Gemini 3.1 Pro High `PASS`; both read-only tripwires passed
- Gate: 20 focused tests pass; Ruff passes; Pyrefly reports 0 errors; working and cached diff checks pass
- Commit: authorised `Record the accepted-contact trace results` result commit
