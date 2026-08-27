# Resume

- Current work and status: the rally-start candidate list passes all four fixed limits; it covers 56 of 81 target first contacts at ten frames
- Next action: write and audit a separate plan for choosing one candidate, including the video-held-out first-model scores required for training
- Active reviewer: none
- Branch: `contact-det-feasibility`; latest commit `9ceb1823 Build the rally-start candidate list`
- Last useful check: 96 focused tests and all 1,893 project tests pass; a fresh reviewer confirmed the missed-contact hash and matching fixes
- Important result: 0.8625 timing F1 at five frames; 99 fully correct sections out of 609 accepted at ten frames
- Important error: first-contact recall is 41.8%, later-contact recall is 89.0%, and 94 otherwise-good single-rally sections are one contact short
- Missed-contact result: all 94 one-short sections have a nearby candidate; 81 are missing the first contact and 39 of those have candidates only before the detected section starts
- Candidate-list result: 56 of the 81 target first contacts are covered at ten frames; 30 only by frames before the detected section; 1,230 earlier entries give 21.96 per covered contact
- Scoring note: the old whole-rally score checks 677 detected sections rather than one row per labelled rally; `baseline_report.md` explains the counts
- Blockers: none for local work
- Important files: `baseline_report.md`, `missed_contact_report.md`, `rally_start_candidate_report.md`, their compact JSON summaries, `rally_start_candidate_plan.md`, `contract.md`, `decisions.md`, `plan.md`, `worklog.md`
