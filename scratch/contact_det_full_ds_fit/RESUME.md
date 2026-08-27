# Resume

- Current work and status: the label-free validation input is recorded and the fixed candidate-model runner has passed its final review
- Next action: commit and run the fixed six-choice model comparison
- Active reviewer: none
- Branch: `contact-det-feasibility`; latest commit `16f7228e Record the validation rally-start inputs`
- Last useful check: all 137 experiment tests pass; the pinned type check reports 0 errors; this directory passes Ruff
- Important result: 0.8625 timing F1 at five frames; 99 fully correct sections out of 609 accepted at ten frames
- Important error: first-contact recall is 41.8%, later-contact recall is 89.0%, and 94 otherwise-good single-rally sections are one contact short
- Missed-contact result: all 94 one-short sections have a nearby candidate; 81 are missing the first contact and 39 of those have candidates only before the detected section starts
- Candidate-list result: 56 of the 81 target first contacts are covered at ten frames; 30 only by frames before the detected section; 1,230 earlier entries give 21.96 per covered contact
- Scoring note: the old whole-rally score checks 677 detected sections rather than one row per labelled rally; `baseline_report.md` explains the counts
- Training-score result: 1,193,927 rows across all 32 training videos; each video was scored by a model trained on the other 24 videos in its fixed group split
- Rally-start input result: 2,621 section lists, 7,863 entries and 5,242 earlier candidates; 2,449 earlier candidates have no player-side answer
- Validation input result: 615 section lists and 1,230 earlier candidates; 601 earlier candidates have no player-side answer; the repeated files match byte for byte
- Selection plan: compare logistic regression and shallow HGB at cut-offs 0.5, 0.7 and 0.9, using held-out scores for every training video
- Important files: `rally_start_selection_plan.md`, `rally_start_model_runs.json`, `scripts/run_rally_start_model.py`, `validation_rally_start_input_report.md`, `contract.md`, `decisions.md`, `plan.md`, `worklog.md`
