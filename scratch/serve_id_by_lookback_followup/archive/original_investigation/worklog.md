> ARCHIVED 2026-08-12: complete original worklog. Live re-entry record: `../../worklog.md`.

# Worklog

## Resume

- Stage: additive correction complete and checked
- Next action: hand over without a commit
- Active workers or audits: none
- Worktree: the repository root, branch `investigation/serve-id-by-lookback-followup`, actual tip `eb83513`
- Last verified gates: deterministic H3/R8 check exit 0; independent review
  `CLEAN`; 58 combined scratch tests and focused Ruff pass; whole Pyrefly 0 errors;
  whole pytest 1,456 passed; whole Ruff has 661 unrelated existing findings
- Blockers: none. Whole-project Ruff is not green because of 661 existing
  findings outside this extension; the extension's focused Ruff check is green.
- Critical restart file: `02_LAUNCH_H3_R8_DUAL_SEARCH.md`
- Worktree note: `overlay_sample/` and `render_overlay_sample.py` are untracked
  work from the completed visual review. Preserve them.

## Concerns and observations

- Extension ruling: use a three-source-frame recurrence halo and an inclusive
  8.0 gross-step ratio for both searches
- Extension ruling: freeze pre- and post-contact evidence for every accepted
  contact before deriving either result
- Extension ruling: Search B starts at the earliest accepted contact with
  positive incoming evidence and does not use outgoing evidence
- Consequence: every earlier contact is not-incoming or unavailable, so the
  backwards search has at most one predecessor hop
- Extension ruling: admit the nearest predecessor within 60 base-30fps frames,
  or through the fixed measured high-shot exception with a 12-base-30fps buffer
  at each state endpoint
- Caution: predecessor admission is a search window, not proof that the earlier
  impulse caused the incoming shot
- Ruling: keep unavailable distinct from not-incoming throughout Search B
- Execution: the user returned after automatic compaction and authorised the
  saved plan to continue
- Tooling: the Serena/Pyrefly server is healthy at
  `http://127.0.0.1:9121/mcp`, but this client's MCP inventory does not expose
  its semantic tools

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
- `accepted_contact_trace_variants.py`: H3 guard, R8 verdicts, and both pure
  searches implemented and focused-tested
- `analyse_accepted_contact_trace_variants.py`: all-contact evidence builder,
  disk freeze, GT-separated scoring, deterministic write/check, and summaries
  implemented and checked on real data
- `test_accepted_contact_trace_variants.py`: 38 focused helper and driver tests
  passing
- `h3_r8_contact_evidence.csv.gz`: 3,200 checked GT-free contact rows
- `h3_r8_search_results.csv.gz`: 239 checked dual-search rows with GT appended
  after the frozen prefix
- `h3_r8_summary.json.gz` and `h3_r8_report.md`: fixed counts and conclusion
- `analyse_additive_correction.py`: GT-free high-shot correction and separate
  visible-start/server scoring
- `additive_correction_results.csv.gz` and
  `additive_correction_summary.json.gz`: frozen 239-row result and summary
- `analyse_serve_setup_continuation.py` and
  `serve_setup_continuation_summary.json.gz`: rejected broad and strict
  serve-setup combinations
- `overlay_sample/`: same frozen 20 clips rerendered at 1920x1080 with a 32 px
  HUD and CRF 18; all 20 decode with expected frames
- `scratch/serve_start_trajectory_exploration/`: read-only source of accepted-contact and trajectory conventions; unchanged
- `src/annotator/`: read-only; unchanged

## Additive correction checks

- Additive correction deterministic check: exit 0
- Serve-setup deterministic check: exit 0
- Focused tests: exit 0; 7 passed
- Focused Ruff: exit 0
- Focused Pyrefly 1.1.1: exit 0; 0 errors
- Serena diagnostics: no warnings or errors in the four new Python files
- Gzip integrity: exit 0
- `git diff --check`: exit 0
- Native Sol final cold read: `CLEAN` after four wording fixes
- No commit, push, merge, or production-code change

## Readiness

- The extension's fixed settings, two search state machines, high-shot
  exception, evidence schema, OUT-list, tests, and halt conditions are pinned
- H1 through H4 are complete; the independent review returned `CLEAN`
- The user authorised one final non-video state commit on 2026-08-11

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

### H3/R8 extension planning

- User review of all 20 overlays exposed that the 15-frame halo removes useful
  traces and that the 4.0 jump limit rejects visually credible paths
- Fixed settings: three-frame halo and ratio 8.0, applied to both searches
- Incoming-only anchor: earliest accepted contact with positive incoming
  evidence; outgoing is ignored
- Predecessor admission: inclusive 60 base-30fps gap, or the measured high-shot
  exception with inclusive 12-base-30fps endpoint buffers
- Evidence order: save all per-contact GT-free measurements before any search or
  scoring
- State: planning saved only; no H3/R8 code, output, lint, type check, or test run

### Batch H1: H3/R8 helpers

- Files: `accepted_contact_trace_variants.py` and
  `test_accepted_contact_trace_variants.py`
- Change: guard reconstruction, fixed ratio-8 eligibility, three-way pre
  verdict, binary outgoing verdict, and both pure search state machines
- Gate: 38 focused tests and Ruff pass; pinned Pyrefly reports 0 errors
- Review: initial worker checks passed; independent end review pending
- Commit: included in the final H3/R8 state commit; see the branch tip for SHA

### Batches H2-H3: checked evidence and results

- Files: `analyse_accepted_contact_trace_variants.py`, three compressed outputs,
  and `h3_r8_report.md`
- Change: save every GT-free accepted-contact measurement, derive both searches
  from the saved rows, then append GT scoring
- Gate: write exit 0; final check exit 0 and exact-matched all 3,200 contact rows
  and 239 search rows
- Finding: first check exposed JSON integer-key normalisation in summary metadata;
  fixed with string keys and a round-trip regression test
- Commit: included in the final H3/R8 state commit; see the branch tip for SHA

### Batch H4: independent review and final gates

- Review: fresh read-only reviewer returned `CLEAN` after checking GT leakage,
  guard reconstruction, search semantics, timing, high-shot arithmetic,
  denominators, OUT-list compliance, and saved-row reproduction
- Report cold read: corrected one count, one causal overstatement, and one
  ambiguous unavailable denominator; result rows and conclusions are unchanged
- Focused gates: 38 H3/R8 tests pass; the combined baseline and H3/R8 run has
  58 passing tests; Ruff passes; Pyrefly reports 0 errors
- Repository gates: 1,456 tests pass and whole Pyrefly reports 0 errors;
  whole Ruff reports 661 existing findings outside the extension
- Overlay: same 20 identities rerendered at 1920x1080 with readable HUD; agent
  validation found all 2,120 expected frames and 20 decodable H.264 files
- Commit: included in the final H3/R8 state commit; see the branch tip for SHA
