# Resume

- Current work and status: the rally-start training inputs are complete for all 32 training videos and independently checked
- Next action: write and audit a small fixed plan for the candidate selection model before opening human labels
- Active reviewer: none
- Branch: `contact-det-feasibility`; latest commit `40109b57 Save rally-start inputs for training videos`
- Last useful check: all 116 experiment tests and all 1,893 project tests pass; the pinned type check reports 0 errors; the new files pass Ruff
- Important result: 0.8625 timing F1 at five frames; 99 fully correct sections out of 609 accepted at ten frames
- Important error: first-contact recall is 41.8%, later-contact recall is 89.0%, and 94 otherwise-good single-rally sections are one contact short
- Missed-contact result: all 94 one-short sections have a nearby candidate; 81 are missing the first contact and 39 of those have candidates only before the detected section starts
- Candidate-list result: 56 of the 81 target first contacts are covered at ten frames; 30 only by frames before the detected section; 1,230 earlier entries give 21.96 per covered contact
- Scoring note: the old whole-rally score checks 677 detected sections rather than one row per labelled rally; `baseline_report.md` explains the counts
- Training-score result: 1,193,927 rows across all 32 training videos; each video was scored by a model trained on the other 24 videos in its fixed group split
- Rally-start input result: 2,621 section lists, 7,863 entries and 5,242 earlier candidates; 2,449 earlier candidates have no player-side answer
- Important files: `training_rally_start_input_report.md`, `training_rally_start_input_summary.json`, `training_rally_start_input_plan.md`, `training_video_score_report.md`, `contract.md`, `decisions.md`, `plan.md`, `worklog.md`
