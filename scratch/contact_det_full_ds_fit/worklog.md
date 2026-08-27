# Worklog

## Pick up from here

- Current work: check whether saved candidate frames can explain the missed rally starts
- Next action: review `missed_contact_check_plan.md`, then add the read-only calculation and focused tests
- Checked so far: the accepted split, saved inputs, all nine timing results, Top/Bottom replay and complete-rally totals
- Plan section: `plan.md`, “Record the first 40-video result”

## Things to remember

- The older handover ends at commit `6732d15`. Later pilot work continues through `5f6da72`.
- The repository's other work-tracking file describes unrelated work. This directory holds the state for the full-data contact experiment.
- The ShuttleSet22 overlap list and prepared input location still need to be recorded before the final test.
- The second code-reading task ran longer than planned. It was stopped and returned useful findings that it checked against the source.
- The exact list of no more than 12 full model runs must be committed before model training starts.
- Reports must use simple words and normal speech. They must not invent labels for ordinary ideas.

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
- Planned commit: `Set the rally-start candidate limits`
