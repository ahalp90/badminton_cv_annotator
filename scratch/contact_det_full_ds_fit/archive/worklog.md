# Worklog

## Pick up from here

- Current work: the fixed 47-video ShuttleSet22 score is complete
- Next action: finish the repository checks and commit the tracked result record
- Required check: the independent recount must reproduce the saved timing and player-side totals
- Current blocker: none
- Plan: `shuttleset22_test_plan.md`

## Things to remember

- The older handover ends at commit `6732d15`. Later pilot work continues through `5f6da72`.
- The repository's other work-tracking file describes unrelated work. This directory holds the state for the full-data contact experiment.
- The ShuttleSet22 source list has 47 prepared non-overlapping videos. Eight known overlaps and three unresolved sources are absent.
- The second code-reading task ran longer than planned. It was stopped and returned useful findings that it checked against the source.
- The exact list of no more than 12 full model runs must be committed before model training starts.
- Reports must use simple words and normal speech. They must not invent labels for ordinary ideas.

### Fixed the ShuttleSet22 inpaint run — 2026-08-28

- Decision: run the normal InpaintNet coordinate step from every saved no-inpaint TrackNet CSV
- Videos: the fixed 47 non-overlapping ShuttleSet22 IDs
- GPU check: exact-input frames differed by no more than one pixel on a complete original video
- Guard check: the fabricated count stayed unchanged and four extra frames became degraded
- Storage: keep the prepared extract read-only and write a sibling-file mirror in experiment scratch
- Restart: finish one video in a temporary directory, then rename it into place
- Label boundary: do not open ShuttleSet22 contact labels
- Planned commit: `Plan the ShuttleSet22 inpaint run`

### Added and checked the ShuttleSet22 inpaint runner — 2026-08-28

- Files: `scripts/inpaint_shuttleset22_tracks.py` and its focused tests
- Input check: all 47 base CSVs have contiguous frames, binary visibility and saved tracks that match at 1920 by 1080
- Model path: one GPU model stays loaded; every video uses non-overlapping 16-frame windows and batch size 16
- Save path: each complete mirror directory contains linked prepared files plus the new CSV, track, sidecar, guard files and receipt
- Restart: a complete video is reloaded and hash-checked; an unfinished video remains separate and stops a restart
- First video: video 8 completed 128,400 frames in 2.9 seconds and passed every semantic reload and hash check
- Focused checks: seven tests pass; Ruff passes for both new files; the pinned whole-project type check reports zero errors
- Whole tests: 1,892 passed and 29 skipped; one environment failure could not find `python` on `PATH` and passes when the project virtual environment bin is supplied
- Whole Ruff: exits 1 on 863 existing findings outside the new files
- Planned commit: `Add the ShuttleSet22 inpaint runner`

### Recorded the ShuttleSet22 fork — 2026-08-28

- Finding: the original 40 ShuttleSet videos used stride-8 TrackNet with InpaintNet
- Finding: the prepared 47 ShuttleSet22 videos used stride-8 TrackNet without InpaintNet
- Effect: testing the prepared files would combine dataset change with shuttle-processing change
- Other gap: the prepared test files have no annotation-stage output, so any path must run that stage before contact features can be built
- State: no ShuttleSet22 contact result has been calculated and no test choice has been changed
- Handover: `HANDOVER.md` records the completed experiment, the fork and the original test plan
- Planned commit: `Write the contact experiment handover`

### Fixed the rally-start contact selection plan — 2026-08-28

- File: `rally_start_selection_plan.md`
- Change: fixed the training answer, nine saved model inputs, two models, three cut-offs, four held-out training runs and one validation check
- Model boundary: every training video has both first-model and candidate-model predictions from models that did not train on that video
- Contact change: add one earlier contact or add nothing; never remove or replace a baseline contact
- Player side: a candidate must match both the first-contact time and labelled side to be a positive training answer
- Review: the first independent read found five contract gaps; all five are fixed before labels or model code are opened
- Planned commit: `Plan the rally-start contact choice`

### Added the missing validation input step — 2026-08-28

- Finding: the frozen validation list has frames and scores but no player-side answer for unkept candidates
- Decision: replay the existing Top/Bot rule before labels and save one checked validation input
- Limit: the new step cannot change a candidate, train a model or read a label row
- Files: `scripts/save_validation_rally_start_inputs.py` and its focused tests
- Checks: 121 experiment tests and all 1,893 project tests pass; changed files pass Ruff; the pinned type check reports 0 errors
- Review: an independent read found no code blocker and asked for one real hash-failure test; that test now stops before video inputs load and leaves a clear `running` result
- Planned commit: `Save the validation rally-start inputs`

### Saved the validation rally-start inputs — 2026-08-28

- Result: eight videos, 615 section lists, 1,845 entries and 1,230 earlier candidates
- Player side: 629 earlier candidates have an answer and 601 have no answer
- Label boundary: every saved label flag is false; no human label row was opened by the saver
- Repeat: the two full saves are equal byte for byte
- Privacy: the saved file contains no machine path or server detail
- Raw result: ignored by Git under `raw/validation_rally_start_inputs/`
- Planned commit: `Record the validation rally-start inputs`

### Added the fixed candidate-model runner — 2026-08-28

- Menu: logistic regression and shallow histogram gradient boosting, each at cut-offs 0.5, 0.7 and 0.9
- Separation: each A–D group is scored by a candidate model trained on the other three groups; the exact first contact model training set is also checked for every video
- Label order: read training timing, check the label-file hash, read training player side, and check the hash again
- Validation boundary: save all validation candidate scores before either validation label loader can run
- Contact change: keep every baseline contact and add at most one earlier contact to a section
- Saved checks: write every held-out score and training answer; rebuild the contact streams twice; recount the action totals from the saved rows
- Review: the first read found a combined label read and an unstated cut-off tie; both are fixed, and the second read found no blocker
- Planned commit: `Train the rally-start contact model`

### Ran the fixed candidate-model comparison — 2026-08-28

- Outcome: all six choices stopped at the training rules; validation was not run
- Main failure: the correct-action rate ranged from 18.2% to 51.7%, below the fixed 80% minimum
- Safest result: shallow HGB at 0.9 gained 30 fully correct sections and lost none, but only 76 of its 147 actions were correct
- Label boundary: no validation candidate score file was written and no validation label was read
- Repeat: both the result file and held-out training score file match byte for byte across two runs
- Review: an independent recount reproduced all six gates from the 5,242 saved candidate rows and found no blocker
- Launch fixes: two setup checks stopped safely before labels; small reviewed commits fixed the saved group order and valid search windows inside long sections
- Raw result: ignored by Git under `raw/rally_start_model/`
- Planned commit: `Record the rally-start contact result`

### Fixed the final contact fit — 2026-08-28

- Groups: keep A, B, C and D, then add the eight former validation videos as group V
- Separation: train each fit on the other 32 videos and score the held-out eight
- Fresh scores: rerun A–D because the old files came from 24-video fits; do not mix them into the final result
- Setting choice: use all 40 held-out score rows and only the original 19 cut-offs, three nearby-contact distances and fixed tie order
- Label meaning: all-40 labels may choose the final settings, so this is model fitting rather than an independent test
- Final model: fit HGB once on all 40 after the pair is fixed, save it outside Git and check it after loading
- Review: the first read asked for exact raw row order, source binding and a fixed model reload sample; the follow-up found no blocker
- Planned commit: `Plan the final contact fit`

### Added the final held-out scorer — 2026-08-28

- Files: `scripts/score_final_contact_groups.py`, its focused tests and one parameter added to the old training-row helper
- Separation: every fit accepts labels and training rows from exactly 32 videos, then scores the separate group of eight
- Group V check: its raw identities and probabilities must exactly repeat the chosen baseline's saved validation scores
- Final choice: combine A, B, C, D and V without old kept-contact flags, save the raw scores, then read all 40 labels and check the fixed 57 setting pairs
- Saved checks: require the fixed input hashes, exact row order, one source group per row, complete training counts, one source commit and repeatable files
- Review: a fresh read-only review found no blocker
- Checks: 144 experiment tests and all 1,893 project tests pass; Ruff passes for this directory; the pinned type check reports 0 errors
- Planned commit: `Score every development video`

### Added the final all-40 model fit — 2026-08-28

- File: `scripts/fit_final_contact_model.py` and its focused save-and-reload test
- Input check: binds the fit to the complete five-group result, raw score file, final kept-contact file, fixed inputs and both source commits
- Setting check: recounts all 57 fixed pairs from the raw scores before accepting the chosen cut-off and nearby-contact distance
- Training: uses the unchanged model and training-row rules on exactly all 40 development videos
- Reload check: saves the model outside Git, loads it again and requires equal probabilities for the first and last candidate row from every video
- Saved record: keeps the 80 row identities and probabilities, per-video feature hashes, training counts and library versions without machine paths
- Review: a fresh read-only review found no blocker
- Checks: 145 experiment tests and all 1,893 project tests pass; Ruff passes for this directory; the pinned type check reports 0 errors
- Planned commit: `Fit the final contact model`

### Fitted and checked the final contact model — 2026-08-28

- Setting: score cut-off 0.9 and nearby-contact distance six at 30 frames per second
- Training: all 40 development videos, 1,313,803 selected rows and 94,530 positive rows
- Model check: the saved model reproduced all 80 fixed probabilities after loading
- Result check: an independent read found no blocker and reproduced the input, count, model and hash checks
- Held-out result: 0.9050 precision, 0.8658 recall and 0.8849 F1 at five frames across all 40 videos
- Saved files: model and full result remain outside Git under `raw/final_contact_model/`
- Planned commit: `Record the final contact fit`

## Current files

- `scratch/contact_det/`: finished three-video pilot; unchanged by this work
- `scratch/contact_det_full_ds_fit/`: agreed plan, code map, 40-video list, feature-saving code and small tests

## Work completed

### Agreed the experiment — 2026-08-27

- Files: the initial planning files in this directory
- Change: recorded the goal, split, work limits, review points and commit sequence
- Check: the user confirmed the plan; a scan found no machine paths, hostnames or access details
- Commit: `e69aca0 Set up the full-dataset contact experiment`

### Inspected the three-video code — 2026-08-27

- Files: `current_system_map.md`, `plan.md`, `worklog.md`
- Change: identified the reusable one-video functions and the surrounding code that assumes three videos
- Check: the main agent checked the important split, label-order and Top/Bottom replay findings in the source
- Commit: will be included with the split change

### Added the user's safeguards — 2026-08-27

- Files: `contract.md`, `decisions.md`, `current_system_map.md`, `plan.md`, `worklog.md`
- Change: required out-of-fold predictions for later trained models, limited the first comparison, put removing extra contacts before adding missing contacts, and chose out-of-fold cut-offs across all 40 videos
- Check: the user accepted these changes and kept XGBoost outside the first baseline
- Commit: will be included with the split change

### Added the 40-video split — 2026-08-27

- Files: `shuttleset_development_split.json`, `scripts/experiment_config.py`, focused tests and package files
- Change: added a video list with no machine paths and code that checks video IDs, counts, training and validation roles, and saved metadata
- Check: 13 small tests pass; Ruff passes for this directory; whole-project Pyrefly passes
- Whole-project checks: Ruff reports 863 existing problems outside this directory. The first full test run passed 1,892 tests and failed one unrelated test because the shell could not find a command named `python`. With the project environment added to the shell path, all 1,893 tests pass and 29 are skipped.
- Review: two fresh read-only reviewers found no blocking problem; their code and wording suggestions were applied
- Commit: `bbbeb086 Add the full-dataset contact split`

### Added feature-saving code for any listed video — 2026-08-27

- Files: `scripts/freeze_contact_features.py`, its tests, this plan and the Git rule that keeps large feature files out of commits
- Change: saves one checked feature file per video, records the input file hashes, and marks a run as complete only after every requested video finishes
- Checks: 21 small tests, 37 reused pilot tests and all 1,893 project tests pass; Ruff passes for this directory; whole-project Pyrefly passes
- Review: a fresh read-only reviewer found three important problems; all three were fixed and the reviewer confirmed the fixes
- Commit: `6368d507 Freeze contact features for any video roster`

### Checked the three pilot videos — 2026-08-27

- Files: `feature_preparation_audit.md`, `pilot_feature_check.json`
- Change: ran the committed feature code on `sset_01`, `sset_15` and `sset_21` before starting the 40-video work
- Result: all 130,624 rows match the saved pilot exactly
- Check: the saved and new feature-file hashes were checked before row comparison; the saved records contain no machine paths
- Commit: `a4c8ec3b Record the pilot feature check`

### Fixed the first model comparison — 2026-08-27

- Files: `baseline_runs.md`, `baseline_runs.json`
- Change: fixed nine full model runs, 19 score cut-offs and three distances for merging nearby duplicate predictions before any validation score was read
- Limit: HGB and RF only; two motion choices; two class-weight choices; two small HGB changes; one change to the number of negative examples
- Status: the raw-motion feature run finished all 40 videos with 1,496,146 rows; the common-30 run is now preparing the matching files; model fitting has not started
- Commit: `c11f2062 Fix the first contact model runs`

### Added the fixed training and validation code — 2026-08-27

- Files: `scripts/baseline_config.py`, `scripts/feature_dataset.py`, `scripts/score_contact_baseline.py` and focused tests
- Change: checks the exact nine-run file, checks all 40 feature files before reading contact labels, trains on the 32 training videos only, and chooses the score cut-off and nearby-contact distance on the eight validation videos
- Saved result: keeps every validation score with its video, interval and frame; records the selected contacts, input hashes, model settings and per-video results without machine paths
- Checks: all 55 tests in this directory pass; Ruff passes for this directory; whole-project Pyrefly passes
- Review: a fresh read-only review found three repeatability problems; the code now marks a rerun as running before checks begin, rejects any change to the fixed nine runs, and checks that the written tie order matches the calculation
- Launch check: the compute environment keeps the repository source outside its normal Python search path; the command-line code now adds that checked source folder before loading features
- Commit: `4042c413 Score fixed contact train and validation splits`; the launch fix follows in a small separate commit

### Added one command for the nine fixed runs — 2026-08-27

- Files: `scripts/run_baseline_menu.py` and its focused tests
- Change: runs the exact nine comparisons in order, chooses the matching raw or common-30 feature record, and saves progress after each result
- Failure handling: clears every old child result before opening the menu, records setup or per-run failure without copying path-bearing error text, and accepts a child result only when its version, run ID, source commit and complete status match
- Review: a fresh read-only review found four gaps and one remaining corner after the first fixes; the follow-up review confirms that all are closed
- Commit: `681d630b Run the fixed contact menu in order`

### Ran the nine fixed comparisons — 2026-08-27

- Feature files: both motion choices finished all 40 videos; each has 1,496,146 saved rows
- Repeatability: the reference raw HGB run produced byte-for-byte equal score and result files twice
- Result: all nine planned runs completed; timing F1 at ±5 frames after scaling to 30 frames per second ranges from 0.8498 to 0.8625
- Leading timing result: reference raw HGB with more negatives, with 0.8924 precision, 0.8344 recall and 0.8625 F1
- Decision: no model has been chosen; player-side and whole-rally results come next

### Added strict checks for the completed menu — 2026-08-27

- Files: `scripts/baseline_results.py` and focused tests
- Change: checks every menu, result, feature and score hash; checks the fixed split and model settings; recomputes kept contacts from the saved scores; and confirms every saved prediction frame
- Label handling: hashes the contact-label file as bytes but does not parse its rows or import label-reading code
- Review: the first pass found excessive memory use, no check that both feature sets came from the same input files, unchecked invalid scores and incomplete result checks; all four are fixed
- Follow-up review: confirmed that the final small code move is correct and covered by tests
- Checks: all 66 tests in this directory pass; Ruff and the pinned Pyrefly check pass for this directory
- Full-file check: the experiment machine accepted all nine saved runs and all 1,496,146 raw feature rows
- Commit: `002f9a17 Check the finished contact runs before rally scoring`

### Rechecked the old whole-rally result — 2026-08-27

- Change: reran the existing three-video whole-rally scorer from its saved inputs before adapting it to the eight validation videos
- Result: the new output and the saved compressed result are byte-for-byte equal
- Next step: keep the same rally-matching functions and replace only the parts that assume three videos

### Added player-side prediction saving for the validation videos — 2026-08-27

- Files: `scripts/save_validation_rally_predictions.py` and focused tests
- Change: checks the saved track, pose, court and annotation files before reading them; applies the existing Top/Bottom rule once at each distinct predicted contact; and saves every run's frames, scores, sides and rally ranges
- Label handling: checks the label file hash as bytes through the earlier result checker but does not parse a contact or player-side row
- Memory: handles one validation video's large vision arrays at a time and releases them before loading the next video
- Review: a fresh read-only review found three issues; the final pass confirms all three are closed
- Checks: all 74 tests in this directory pass; Ruff and the pinned Pyrefly check pass for this directory
- Full run: completed all eight validation videos and all nine model runs; the saved contact counts agree with the earlier timing results and the file contains no machine paths
- Local copy: its SHA-256 hash matches the file on the experiment machine
- Commit: `ccdbbf73 Save player sides before opening rally labels`

### Added validation whole-rally scoring — 2026-08-27

- Files: `scripts/score_validation_rallies.py` and focused tests
- Change: fully checks the saved predictions first, reads contact timing without the player-side column, then reads and checks the player-side column separately
- Reused code: keeps the old one-to-one contact matching, half-open rally ranges, whole-rally check and confidence results
- Saved results: reports contact timing, player-side answers at three timing limits, whole-rally accuracy, per-video totals and failure counts for all nine runs
- Review: a fresh read-only review found an out-of-range frame gap, a label-file change gap and a missing order test; the follow-up confirms all three are closed
- Checks: all 85 tests in this directory pass; Ruff and the pinned Pyrefly check pass for this directory
- Full result: all nine runs scored across eight validation videos, 5,696 contacts and 668 labelled rallies
- Commit: `6ef171d0 Score whole rallies after fixing the predictions`

### Checked the first full-data baseline — 2026-08-27

- Files: `baseline_summary.json`, `baseline_report.md`, `decisions.md`, `plan.md`, this worklog
- Leading run: reference raw-motion HGB with balanced class weights and up to 24 negative examples per positive
- Timing result: 0.8924 precision, 0.8344 recall and 0.8625 F1 at five frames after adjustment to 30 frames per second
- Complete-rally result: 99 fully correct sections out of 609 accepted at the main ten-frame limit
- Error check: among 465 failed sections that line up with one rally, 266 have missing contacts without extras, 42 have extras without missing contacts, 65 have both, and 92 have complete timing but a wrong player side
- Narrow follow-up case: 94 sections are exactly one contact short with every predicted time and side otherwise correct
- Start contact result: 41.8% recall at five frames, compared with 89.0% for later contacts
- Review: a fresh read-only reviewer recalculated the headline totals and agreed with the chosen run
- Score boundary: the old score checks 677 detected sections rather than one row per labelled rally; the report states the exact counts and confirms that the chosen run remains best when only one-rally sections are compared
- Count wording: two one-contact rallies have no predictions; they raise the purely numerical one-short count from 94 to 96, but they are not counted as otherwise-good rallies
- Decision: keep the leading HGB run as the baseline; do not try removing extra contacts next
- Commit: `8e43e9ac Record the full-dataset contact baseline`

### Set the limit for the missed-contact check — 2026-08-27

- File: `missed_contact_check_plan.md`
- Input: the chosen HGB run and its unchanged validation scores
- Counts: first and later contacts at five and ten frames, plus the 94 otherwise-good sections that are one contact short
- Excluded: training, cut-off changes, contact changes, player-side changes, production code and ShuttleSet22 labels
- Purpose: decide whether a small rally-start selection test has candidate frames to work with
- Plan review: the saved score file has 283,363 unique video/frame rows and all 5,326 kept frames match the saved predictions; the four explanations now have a fixed order and keep their raw nearby-row counts
- Implementation commit: `Check where the baseline misses contacts`

### Added the missed-contact check — 2026-08-27

- Files: `scripts/check_missed_contacts.py`, its focused tests and two chosen-run hashes in `baseline_summary.json`
- Change: checks every saved input before reading contact-label rows, then explains each missed first or later contact at five and ten frames
- One-short check: reconstructs the 94 otherwise-good sections and records whether the nearby candidate is inside or outside the detected section
- Saved detail: joins by video and frame, keeps nearby row counts, and records both signed frame offset and absolute frame distance
- Review: a fresh read-only review found two blockers; the chosen run and score files are now bound by tracked hashes, and every nearby kept prediction is proved to have matched another label
- Checks: 96 focused tests pass; experiment Ruff and whole-project Pyrefly pass; the real saved inputs complete a path-free smoke run
- Whole-project checks: all 1,893 tests pass and 29 are skipped; Ruff reports the same 863 existing findings outside this experiment
- Commit: `e1c5eaa4 Check where the baseline misses contacts`

### Ran the missed-contact check — 2026-08-27

- Files: `missed_contact_summary.json`, `missed_contact_report.md`, `decisions.md`, `plan.md`, this worklog
- Saved result: `raw/missed_contact_check.json.gz`; complete, path-free and produced from commit `e1c5eaa4`
- First contacts at ten frames: 284 of 364 misses have a saved candidate nearby; 264 are below the 0.9 cut-off
- Later contacts at ten frames: 379 of 542 misses have no saved candidate nearby
- One-short sections: all 94 have a nearby candidate; 81 are missing the first contact and 13 a later contact
- Section boundary: 39 of the 81 missing-first sections have candidate frames only before the detected section starts
- Decision: test a small rally-start candidate list built without labels; do not lower the cut-off everywhere or remove extra contacts next
- Stop before code: set and review exact list-size, coverage and added-candidate limits first
- Commit: `4dc2a037 Record the missed-contact result`

### Set the rally-start candidate limits — 2026-08-27

- File: `rally_start_candidate_plan.md`
- List: the first kept contact and at most two earlier HGB score rows for each detected section
- Size limits: at most three candidates per section and 1,845 across validation
- Result limits: cover at least 50 of the 81 target first contacts and add no more than 25 entries per covered contact
- Label order: fix and reproduce the list before opening the saved missed-contact detail
- Excluded: the pilot's failed hand-written choice rule, model training, contact changes and ShuttleSet22 labels
- Review: an independent read found four clarity and counting problems; the follow-up confirms that all four are fixed
- Commit: `5d52dc10 Set the rally-start candidate limits`

### Added the rally-start candidate check — 2026-08-27

- Files: `scripts/check_rally_start_candidates.py` and its focused tests
- Change: builds the fixed list twice, saves equal candidate bytes, then opens the already checked missed-contact detail and measures the four limits
- Input checks: binds the model run, scores, predictions, complete-rally result, missed-contact result, split, feature record and contact-label file by hash
- Section handling: each broad missed-first-contact count uses the detected section assigned to that labelled rally; candidates cannot cross between sections
- Fixed distance: requires exactly six frames at 30 fps and uses the baseline frame-rate adjustment
- Checks: 104 tests in this experiment pass; Ruff passes for the changed files; whole-project Pyrefly reports zero errors
- Review: an independent code read found three blockers; the follow-up confirms the section assignment, input checks and exact distance are fixed
- Commit: `9ceb1823 Build the rally-start candidate list`

### Ran the rally-start candidate check — 2026-08-27

- Files: `rally_start_candidate_summary.json`, `rally_start_candidate_report.md`, this worklog and the other living plan files
- Saved list: 615 section lists, each with one fixed contact and two earlier candidates; 1,845 entries in total
- Main result: covers 56 of the 81 target first contacts at ten frames and 45 at five frames
- Earlier section boundary: 30 of the 56 contacts at ten frames are covered only by candidates before the detected section
- Cost: 1,230 earlier entries, or 21.96 per covered target contact
- Limits: all four fixed size and coverage checks pass
- Repeatability: a second full run produced the same compressed-file hashes
- Independent recount: found the same 81 targets, 56 covered contacts, 30 covered only before the section, 1,845 entries and 1,230 earlier entries
- Review: an independent result audit checked the input hashes, output contents and arithmetic and found no blocker
- Decision: keep the candidate list; plan a separate trained choice method using first-model scores made without training on the same video
- Commit: `fe17bd19 Record the rally-start candidate result`

### Set the training-video score groups — 2026-08-27

- Files: `training_video_score_groups.json`, `training_video_score_inputs.json` and `training_video_score_plan.md`
- Split: four groups of eight current training videos; each has four videos at each frame rate and the seven women's matches are spread 2, 2, 2 and 1
- Fits: exactly four chosen-HGB fits; each trains on the other 24 training videos and scores its held-out eight
- Validation boundary: the existing eight validation videos train none of the four models
- Fixed settings: chosen HGB, raw motion, balanced weights, up to 24 negatives per positive, seed 20260824, cut-off 0.9 and duplicate distance six
- Expected output: 1,193,927 score rows selected by the same seven search flags as the baseline
- Input record: tracked hashes pin the groups, split, settings, raw feature record, baseline summary/result and contact labels before fitting
- Launch check: run group A twice and require identical saved bytes before starting the other three fits
- Review: an independent plan audit found three blockers; the final pass confirms that the row counts, input hashes and exact 24-video label boundary are fixed
- Planned commit: `Set the held-out training score groups`

### Added the training-video scorer — 2026-08-27

- Files: `scripts/score_training_videos.py` and its focused tests
- Change: each fit receives contact labels and training examples from exactly 24 videos, then scores the separate group of eight videos
- Fixed boundary: the eight validation videos and the eight videos being scored cannot enter a fit's labels or training examples
- Saved checks: each group records its model, input hashes, training counts and scores; the final combination must match every expected candidate row in the fixed group order
- Failure handling: group and combined results say `running` before checks begin, so an old complete result cannot survive a failed rerun
- Launch audit: a test found that the first version overwrote group D's result while combining; separate path names fix it, and the repeat test now covers the failure
- Review: an independent second pass found no remaining blocker after that fix and the saved training-setting checks
- Checks: all 107 experiment tests and all 1,893 project tests pass; the pinned type check reports 0 errors; the new files pass Ruff
- Commit: `3b6297ca Score training videos without training on them`

### Checked the full-fit launch — 2026-08-27

- Local state: the tracked records are present, but the large feature files remain on the compute copy as intended
- Run state: no full fit started and no result file was created
- Next action: run group A twice in `tmux`, require equal score and result bytes, then run groups B, C and D and combine them twice

### Scored all training videos with held-out models — 2026-08-28

- Files: four group results, one combined score file, `training_video_score_summary.json` and `training_video_score_report.md`
- Rows: 1,193,927 score rows and 26,459 kept contacts across all 32 training videos
- Separation: each group of eight videos was scored by a model trained on the other 24 videos; validation videos were absent from every fit
- Repeat check: group A and the final combination produced identical files on their second runs
- Independent check: all input hashes, group lists, row identities, score bounds and kept-contact decisions passed
- Commit: `9b9bbbfe Record the training-video contact scores`

### Planned the rally-start training inputs — 2026-08-28

- File: `training_rally_start_input_plan.md`
- Need: the held-out score rows do not contain detected-section boundaries or predicted player sides
- Source: use the saved label-free video-pipeline result and its already checked shuttle, pose and court inputs
- Candidate rule: reproduce the frozen validation list exactly, then use the same first-kept-contact plus two-earlier-candidates rule for the 32 training videos
- Saved progress: one checked file per video, followed by one combined file in fixed group order
- Boundary: do not open human contact, rally or player-side label rows and do not train a choice method
- Review: the first read found missing checks for the training row shape, model separation and detected-section source; the revised plan covers all three

### Added the rally-start training input saver — 2026-08-28

- Files: `scripts/save_training_rally_start_inputs.py`, its focused tests and the shared one-video candidate-list function
- Change: checks the four held-out score groups, reproduces the frozen validation list, replays player sides and saves one restartable file per training video
- Label boundary: human contact and player-side files are checked only by filename and hash; no label row is parsed
- Restart rule: an interrupted video remains marked `running`; a complete child is reused only after its inputs and saved contents pass again
- Review: a fresh code audit found two integrity gaps; the final code compares combined scores with every group score and replaces stale complete markers before stage checks
- Checks: 116 experiment tests and all 1,893 project tests pass; the pinned type check has 0 errors; changed files pass Ruff
- Commit: `40109b57 Save rally-start inputs for training videos`

### Saved all rally-start training inputs — 2026-08-28

- Smoke replay: `sset_01`, `sset_02` and `sset_03` completed before the full run
- Full result: 32 videos, 2,850 detected sections, 2,621 candidate lists, 7,863 entries and 26,459 kept contacts
- Earlier candidates: 5,242 total; 2,419 before the section and 2,823 inside it
- Player-side limit: 2,449 earlier candidates have no answer from the existing rule
- Repeat check: the two combined files match byte for byte with SHA-256 `49236a091efde5ee9fcc6ac52616a716a276c992abe833c46830e30c5ec7e784`
- Independent check: all 1,193,927 score identities match the 40 raw feature files and the four group files in fixed A–D order
- Next action: write and audit the candidate selection plan before opening human labels

### Set the ShuttleSet22 output reload rule — 2026-08-28

- Full run: all 47 intended videos completed before the tmux session exited
- First reload: stopped at video 51 because its inpainted CSV has a Y coordinate outside the 1080-pixel frame
- Source check: the original TrackNetV3 code converts InpaintNet output straight to pixels without clipping it to the frame
- Loader check: the normal shuttle loader accepts those values and normalises them in the same way as other coordinates
- Decision: keep strict frame bounds for the saved TrackNet input and allow the original coordinate range when reloading InpaintNet output
- Final gate: count every outside-frame output row, then rerun the receipt, hash, track, sidecar and guard checks for all 47 videos

### Finished and checked the ShuttleSet22 inpaint run — 2026-08-28

- Completed set: all 47 fixed videos and 6,175,283 frames
- Inpaint selection: 2,916,960 frames; visible coordinates rose from 2,411,365 to 5,322,221
- Final reload: checked every saved hash, CSV, track, sidecar and fill count, then regenerated and exactly compared every shuttle guard output
- Coordinate range: one row in video 51 is one pixel below the picture at `(1909, 1080)`; the original InpaintNet code leaves output unclipped and the normal shuttle loader accepts it
- Outside-frame recount: one affected row in one video; all other output rows are inside the picture
- Label boundary: no ShuttleSet22 contact label was opened
- Runtime commit: `e6a16084 Allow original InpaintNet output range`
- Local checks: 9 focused runner tests pass; all 1,893 project tests pass and 29 skip; the pinned type check finds 0 errors
- Ruff: the changed runner files pass; the whole repository still has the same 863 existing findings outside this work
- Next action: write and check `shuttleset22_test_plan.md`, then prepare all contact predictions before opening labels

### Added the label-free ShuttleSet22 predictor — 2026-08-28

- Files: `shuttleset22_test_plan.md`, `scripts/prepare_shuttleset22_predictions.py` and its focused tests
- Boundary: the predictor has no label path or label-reader import; it freezes all features, probabilities, kept contacts and player-side answers before scoring exists
- Fixed setup: exact 47-video input set, prepared and inpaint identities, final model, 85 model fields, 0.9 cut-off and six-frame nearby-contact distance
- Saved state: one atomic directory per video, deterministic combined predictions and a restartable run-state file
- Checks: 163 experiment tests and all 1,893 project tests pass; changed files pass Ruff; pinned Pyrefly reports 0 errors
- Commits: `b4d01a2 Plan the ShuttleSet22 test` and `59f840f Prepare the ShuttleSet22 predictions`
- Label boundary: no ShuttleSet22 contact label has been opened
- Next action: run video 8 as a real-input smoke check, then freeze all 47 predictions

### Real-input smoke run stopped on stale saved records — 2026-08-28

- The predictor stopped before processing video 8 because the run environment has old copies of the final-fit and final-setting result JSON files
- The model file and recorded Python package versions match the fixed setup
- The local copies of both result files match the fixed hashes
- No ShuttleSet22 contact label was opened
- The two stale copies were replaced with the checked local files and their hashes now match
- Video 8 then completed and saved a restartable result
- Next action: run the remaining 46 videos in `tmux`, then reproduce the combined 47-video file byte for byte

### Tightened the prediction reload checks — 2026-08-28

- A read-only DeepSeek audit confirmed the label boundary and fixed detector settings
- The audit's claimed field-order risk was not present because inference selects columns by the ordered field list saved with the pinned model
- Change: a resumed video now compares its saved input records with the live files and requires the complete output-hash list
- Change: the combined file is read back byte for byte before the run is marked complete
- Review: a Luna xhigh read-only diff check found no problem
- Checks: 12 focused tests and 166 experiment tests pass; changed-file Ruff passes; pinned Pyrefly reports 0 errors; the full suite passes with 1,893 tests and 29 skips
- Whole-project Ruff still reports the same 863 existing findings outside this work
- Commit: `32d7087 Tighten the prediction reload checks`
- Next action: deploy this validator after the active long run, then freshly reload all 47 saved videos

### Froze all 47 ShuttleSet22 predictions — 2026-08-28

- The full label-free run finished with exit 0 and saved all 47 videos
- The tightened validator then freshly reloaded every video and rebuilt the combined file with exit 0
- The run state is complete with 47 unique videos and the combined file hash is `6199ab99fe2746f83b7f90cc2e2c02301acbd5f90dcf02c989af65ca6be5bd04`
- Independent recount: 3,982 detected spans, 39,994 contacts and 72 contacts without a player-side answer
- Saved contact scores range from 0.9000015372251916 to 0.9980202583145943
- Video 22 logged the normal unavailable state for its optional fabrication guard because the derived margin was below the accepted minimum; its saved guard inputs and prediction outputs still passed the fixed checks
- The final combined file was written at 2026-08-28 17:35:02 AEST
- No ShuttleSet22 contact label has been opened
- Next action: implement and check the scorer, then ask for the label root before the one-time label read

### Added and checked the ShuttleSet22 scorer — 2026-08-28

- File: `scripts/score_shuttleset22_test.py` and its focused tests
- Boundary: the scorer requires the exact frozen combined hash, complete run state and every saved child hash before annotation access can start
- Label rule: reproduces the preserved whole-rally cleaning and per-contact Top/Bot rule, including unknown human sides
- Result: reports timing at one, two, five and ten frames, first and later recall, player-side coverage, detected-section outcomes, unassigned contacts and the descriptive confidence curve
- Time check: records the first label-access time and requires the combined prediction file to predate it
- Review: a Luna xhigh contract read found the missing time check and confidence denominators; both are fixed and tested
- External audit: three read-only DeepSeek attempts ended during file inspection without returning a report
- Checks: 16 focused tests and 182 experiment tests pass; all 1,893 project tests pass with 29 skips; changed files pass Ruff; pinned Pyrefly reports 0 errors
- Whole-project Ruff still reports the same 863 existing findings outside this work
- Commit: `4835442 Add the ShuttleSet22 scorer`
- Label boundary: no ShuttleSet22 contact label has been opened
- Next action: ask for the annotation root, then run the scorer once

### Validated the full ShuttleSet22 annotation set — 2026-08-28

- Finding: the official annotation corpus contains all 58 match IDs
- Test set: 47 matches have separate downloadable, frame-aligned videos and do not overlap the base ShuttleSet development data
- Exclusions: eight matches overlap base ShuttleSet; IDs 14, 45 and 56 have unresolved frame-aligned public video sources
- Validation: authenticate the complete official corpus and match map, then parse labels only for the fixed 47 test matches
- Checksum: the scorer now reproduces the pinned historical annotation-corpus hash directly
- Review: a Luna xhigh read found no mismatch with the historical checksum or fixed 47/8/3 source split
- Commit: `4fd6c7a Validate the full ShuttleSet22 labels`

### Recorded the 58-to-47 dataset accounting — 2026-08-28

- Files: `shuttleset22_test_plan.md` and `decisions.md`
- Record: the plan explains the 47 downloadable matches, eight overlaps and three unresolved sources
- Unresolved sources: the plan preserves the three source-search findings without treating them as explanations from the dataset authors
- Privacy: the tracked note contains public provenance and no machine path or remote-working detail
- Commit: `1605a32 Explain the 47-video test set`

### Ran the fixed ShuttleSet22 test — 2026-08-28

- Boundary: the scorer accepted the frozen predictions before it started label access
- Identity: the source manifest, official 58-match annotation corpus and complete annotation tree matched their pinned hashes
- Labels: 43,159 source rows became 38,218 usable contacts in 3,422 rallies
- Predictions: 39,994 contacts across 3,982 detected sections
- Five-frame timing: 80.62% precision, 84.37% recall and 82.45% F1
- Five-frame player side: 92.02% accuracy when both sides are answered
- Whole rallies: 493 of 2,969 one-rally sections are fully correct at five frames; 537 are fully correct at ten frames
- Setting boundary: the result does not change the model, features, 0.9 score cut-off or six-frame nearby-contact distance
- Independent recount: a standalone standard-library implementation reproduced every per-video and aggregate timing and player-side field at all four tolerances with zero mismatches
- External audit: a fourth read-only DeepSeek pass independently reproduced the hashes and headline figures, then hung before returning a final report; it is not counted as a passed audit
- Saved record: `shuttleset22_test_report.md` and `shuttleset22_test_summary.json`; raw labels and detailed results stay outside Git
- Final tests: the required command exits 1 only because `python` is absent from the shell `PATH`; it passes all 1,893 tests with 29 skipped when the project virtual-environment bin is added to `PATH`
- Final types: pinned Pyrefly exits 0 with no errors and 21 suppressed messages
- Final lint: whole-project Ruff exits 1 on the same 863 existing findings outside this work
