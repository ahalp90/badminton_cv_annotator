# Resume

- Current work and status: the four-group scorer is committed, independently checked and ready to run; no full fit has started
- Next action: run group A twice and compare both saved files before groups B, C and D
- Active reviewer: none
- Branch: `contact-det-feasibility`; latest commit `3b6297ca Score training videos without training on them`
- Last useful check: all 107 experiment tests and all 1,893 project tests pass; the pinned type check reports 0 errors; the two new files pass Ruff
- Important result: 0.8625 timing F1 at five frames; 99 fully correct sections out of 609 accepted at ten frames
- Important error: first-contact recall is 41.8%, later-contact recall is 89.0%, and 94 otherwise-good single-rally sections are one contact short
- Missed-contact result: all 94 one-short sections have a nearby candidate; 81 are missing the first contact and 39 of those have candidates only before the detected section starts
- Candidate-list result: 56 of the 81 target first contacts are covered at ten frames; 30 only by frames before the detected section; 1,230 earlier entries give 21.96 per covered contact
- Scoring note: the old whole-rally score checks 677 detected sections rather than one row per labelled rally; `baseline_report.md` explains the counts
- Blocker: none; no full fit has started
- Important files: `training_video_score_plan.md`, `training_video_score_groups.json`, `training_video_score_inputs.json`, `scripts/score_training_videos.py`, its tests, `baseline_report.md`, `missed_contact_report.md`, `rally_start_candidate_report.md`, `contract.md`, `decisions.md`, `plan.md`, `worklog.md`
