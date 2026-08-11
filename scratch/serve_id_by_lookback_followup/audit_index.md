# Audit index

| Audit | Model and effort | Scope | Status | Result | Coordinator check |
| --- | --- | --- | --- | --- | --- |
| Accepted and serve-start source trace | `gpt-5.6-luna`, max | Read-only named source trace | Stopped for scope expansion | Ignored serve-start material; retained accepted-order references | Accepted-order claims checked against source |
| Corrected run-ending source trace | `gpt-5.6-luna`, max | Four named source and test files, read-only | Complete | Mapped retained run fields and lost end reasons | Every material claim checked with Serena, text search, and focused tests |
| Current decisions cold read | `gemini-3.6-flash-high`, high | `decisions.md` only, read-only | Complete | `READABLE`; Q1 choice placement and interval notation needed fixes | Genuine findings fixed; clause-count drift rejected because it combined separate sentences |
| Batch 0 rule review | fresh native Codex | Planning diff and OUT-list, read-only | Complete | Found stale gap vetoes, a dead conflict branch, omitted files, and an ineffective untracked-file gate | Findings checked locally and folded before staging |
| Batch 0b simplified-rule review | fresh native Codex | Final sequential-search planning diff and OUT-list, read-only | Complete | Found one omitted gate file, one stale unknown-state sentence, and overclaiming in the word `junk` | All three findings checked and fixed before staging |
| Batch 1 code review | fresh native Codex | Sequential helper and focused tests, read-only | Complete | Found three API and coverage gaps; re-review was `CLEAN` after fixes | All findings fixed; 17 tests and Ruff independently passed |
| Batch 2 data-path map | native Codex explorer | Named PR #82 source and population path, read-only | Complete | Confirmed the fixed masks and noted the deliberate span bound and GT-derived population crosswalk | Findings checked in the driver and tests |
| Batch 2 analysis review | fresh native Codex | Driver, focused tests, and new output schemas, read-only | Complete | Found a GT-rich input interface despite no actual leakage; re-review was `CLEAN` after projection fix | Full check mode still matches all 239 rows |
| Superseded findings cold read | `gemini-3.6-flash-high`, high | One findings document | Complete | `READABLE`; one abbreviation fix | Document later replaced; result has no bearing on current findings |
| Final causal audit | `claude-opus-4-6-thinking`, model-defined effort | Compact rule, counts, report, helper, driver, and tests; read-only | Complete | `PASS`; no result or wording findings | Assumptions checked against prior source map, focused tests, and saved-row rebuild |
| Final causal audit | `gemini-3.1-pro-high`, high | Compact rule, counts, report, helper, driver, and tests; read-only | Complete | `PASS`; no findings | Assumptions checked against prior source map, focused tests, and saved-row rebuild |
