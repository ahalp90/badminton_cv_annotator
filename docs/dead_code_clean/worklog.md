# Worklog — dead/parallel/duplicate code audit

## Resume block

- AUDIT COMPLETE. Nothing pending. Sol (high) audited report.md and returned
  11 defects (3 fidelity, 4 omission, 1 proportionality, 1 readability set,
  2 count errors); all folded. Four previously delegate-only absorb claims
  were verified first-hand during the fold (WP4-3, WP4-7, WP5-6, WP6-10);
  WP6-11 was restored to the ledger. The refactor itself is a separate,
  unstarted pass awaiting Ariel's rulings (report.md "Decisions" section).
- State: all eight sweeps returned (161 ledger rows, raw returns archived
  under findings/wp1-8.md); merged ledger written (findings.md) with
  verification verdicts; readable report written (report.md).
- Verified so far: root manifest; all delete/absorb recommendations in the
  report re-checked first-hand (git grep over tracked content + delegate
  pyrefly refs). PyCharm call hierarchy would not resolve Python callables
  (tried three FQN forms on detect_and_track), so the promised find-usages
  pass fell back to grep + pyrefly, recorded in findings.md's legend.
- Two delegate claims refuted (WP5 outward notes), two amended (WP4-2,
  WP4-6); details in findings.md "Refuted / amended".
- Runbook line in play: audit_plan.md "Order of execution" step 5.

## Concerns and observations

- (verify) WP6-4 (write-only bric cache fields) changes an on-disk contract;
  flagged in report as needing its own pass before action.
- (verify) WP7's 30 archive classifications rest on doc citations; I
  spot-checked the scripts/archive boundary sweep only. Re-verify per file at
  refactor time.
- (process) max-effort raw codex exec runs read-only; full ruff could not
  create its cache there (WP6 noted it). No impact on findings.
- (resolved) planning note about WP4/WP7 cluster overlap: superseded by the
  rev-2 disjoint ownership; no double-reported findings survived the merge.
- (resolved) planning note that api never imports bst_x: WP6 confirmed the
  api's BST path lives in src/api/bst_x_inference.py with its own loading;
  no hidden import.

## Module state

Audit-only pass; no source modules touched. The docs under
docs/dead_code_clean/ are the entire output surface:
- audit_plan.md rev 2 (post Sol-medium review, ten criticisms folded)
- findings/wp1-8.md — raw delegate returns, verbatim
- findings.md — merged ledger with VERIFIED / DELEGATE / REFUTED / AMENDED
- report.md — readable summary, pending Sol-high audit
- decisions.md — five recorded deviations/rulings

## Execution log

- 2026-08-01 plan drafted (audit_plan.md, decisions.md). Import-map grounding
  run in-session: shared unused by bst_x; flat-namespace imports; scraper
  split across scraper/ and annotator/.
- 2026-08-01 Sol (gpt-5.6-sol, medium, read-only) plan review: ten criticisms,
  all folded; plan rev 2 with disjoint WP ownership, split dead taxonomy,
  root manifest, trimmed deliverables.
- 2026-08-01 ~17:52 WP1-WP8 launched, all gpt-5.6-luna read-only. xhigh via
  companion: WP1 (vendor mirrors), WP7 (census), WP8 (tests). max via raw
  codex exec: WP2 (cross-package), WP3 (annotator+courtkeynet), WP4
  (scraper+shared), WP5 (bst_x), WP6 (bric+api).
- 2026-08-01 all eight returned; finals archived to findings/. Row counts:
  WP1 2, WP2 11, WP3 17, WP4 10, WP5 12, WP6 16, WP7 77, WP8 16.
- 2026-08-01 verification pass: three cross-sweep contradictions resolved
  (two WP5 outward notes refuted, WP4-8 confirmed); zero-caller greps for all
  D/T candidates; SPLITS_V2 claim amended; _select_device and clip-index
  duplications confirmed identical; PyCharm call-hierarchy fallback recorded.
- 2026-08-01 findings.md and report.md written.
- 2026-08-01 Sol (gpt-5.6-sol, high, read-only) report audit: 11 defects, all
  folded. Fixes: mirror rationale re-attributed to BRIC; census fraction
  corrected to 40 of 77; verification claims split into re-checked-by-hand vs
  sweep-verified; WP6-11 restored; theme 4 split wrappers/helpers/probes;
  the two annotator consolidations added to theme 7; numbers recounted from
  ledger sections; TrackNet collapse cost caveated; WP6-2/3 and WP3-17 added;
  jargon replaced with plain terms.
