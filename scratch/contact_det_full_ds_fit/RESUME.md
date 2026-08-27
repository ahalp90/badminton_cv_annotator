# Resume

- Current work and status: the nine-run baseline and missed-contact check are complete; `hgb_reference_raw_more_negatives` is the leading first contact model
- Next action: write down and review the size and coverage limits for one small rally-start candidate list
- Active reviewer: none
- Branch: `contact-det-feasibility`; last code commit `6ef171d0 Score whole rallies after fixing the predictions`
- Last useful check: 96 focused tests and all 1,893 project tests pass; a fresh reviewer confirmed the missed-contact hash and matching fixes
- Important result: 0.8625 timing F1 at five frames; 99 fully correct sections out of 609 accepted at ten frames
- Important error: first-contact recall is 41.8%, later-contact recall is 89.0%, and 94 otherwise-good single-rally sections are one contact short
- Missed-contact result: all 94 one-short sections have a nearby candidate; 81 are missing the first contact and 39 of those have candidates only before the detected section starts
- Scoring note: the old whole-rally score checks 677 detected sections rather than one row per labelled rally; `baseline_report.md` explains the counts
- Blockers: none for local work
- Important files: `baseline_report.md`, `baseline_summary.json`, `missed_contact_report.md`, `missed_contact_summary.json`, `missed_contact_check_plan.md`, `contract.md`, `decisions.md`, `plan.md`, `worklog.md`
