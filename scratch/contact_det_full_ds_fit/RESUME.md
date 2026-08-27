# Resume

- Current work and status: the failed rally-start addition is recorded; the final all-40 contact fit plan is under review
- Next action: commit the fixed five-group plan, then add the scorer for A, B, C, D and V
- Active reviewer: none; the final plan review found no blocker
- Branch: `contact-det-feasibility`; latest commit `0948c8ee Record the rally-start contact result`
- Last useful check: 18 focused model tests pass; the full saved row build has 5,242 training candidates and 1,230 validation candidates
- Important result: 0.8625 timing F1 at five frames; 99 fully correct sections out of 609 accepted at ten frames
- Important error: first-contact recall is 41.8%, later-contact recall is 89.0%, and 94 otherwise-good single-rally sections are one contact short
- Missed-contact result: all 94 one-short sections have a nearby candidate; 81 are missing the first contact and 39 of those have candidates only before the detected section starts
- Candidate-list result: 56 of the 81 target first contacts are covered at ten frames; 30 only by frames before the detected section; 1,230 earlier entries give 21.96 per covered contact
- Scoring note: the old whole-rally score checks 677 detected sections rather than one row per labelled rally; `baseline_report.md` explains the counts
- Training-score result: 1,193,927 rows across all 32 training videos; each video was scored by a model trained on the other 24 videos in its fixed group split
- Rally-start input result: 2,621 section lists, 7,863 entries and 5,242 earlier candidates; 2,449 earlier candidates have no player-side answer
- Validation input result: 615 section lists and 1,230 earlier candidates; 601 earlier candidates have no player-side answer; the repeated files match byte for byte
- Rally-start model result: all six choices failed the 80% correct-action rule; shallow HGB at 0.9 reached 51.7%, gained 30 fully correct sections and lost none
- Validation boundary: no candidate score file was written for validation and no validation label row was read
- Final-fit plan: rerun A–D with V included in training, score V from A–D, choose only from the original 57 setting pairs, then fit HGB once on all 40
- Important files: `final_contact_fit_plan.md`, `final_video_score_groups.json`, `final_contact_score_inputs.json`, `rally_start_model_report.md`, `contract.md`, `decisions.md`, `plan.md`, `worklog.md`
