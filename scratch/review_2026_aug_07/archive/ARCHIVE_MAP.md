# Archive map (tidy and culls of 2026-08-07)

The live entry point is `../REVIEW.md`. This archive keeps only what REVIEW.md cites
as evidence: the two source reports and the readable delegate outputs
(`result_extracted.md` per evidence job). Everything else from the review was deleted
in two culls on Ariel's ruling and remains recoverable from the pre-tidy snapshot
`scratch/review_2026_aug_07.pre-tidy-20260807.tar.zst` until that is removed.

Kept:

| File | Role |
|---|---|
| archive/report.md | first-pass report (tombstoned; superseded by REVIEW.md; holds the fuller per-finding arguments) |
| archive/calibration-report.md | calibration report (tombstoned; superseded by REVIEW.md) |
| archive/evidence/jobs/*/result_extracted.md | 10 first-pass census/verification results |
| archive/evidence/calibration-jobs/*/result_extracted.md | 6 calibration evidence results |

Deleted (snapshot only):

- process prompts and templates: review-prompt.md, next-session-brief.md, the
  calibration brief, four templates, all per-job brief.md files, evidence/briefs/
- raw delegate mechanics under evidence/jobs/ (~24M: codex-output.jsonl, stderr logs,
  launch files, raw result.md, launchlogs, run_wave scripts)
- worklog.md (first-pass process log), calibration-independent-positions.md
  (pre-report positions), REVIEW-first-draft.md (earlier wording of REVIEW.md)
- collected_tests.txt and tests_per_file.txt (regenerable via pytest --collect-only)

Known dangling pointers, accepted: the archived reports cite worklog.md,
calibration-independent-positions.md and per-job brief.md files, all deleted above.
