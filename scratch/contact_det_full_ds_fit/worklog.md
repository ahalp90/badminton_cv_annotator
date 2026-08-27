# Worklog

## Pick up from here

- Current work: finish the common-30 feature files and run the fixed 32/8 model comparison
- Next action: commit the checked training code, then run one repeatability check before the nine planned runs
- Checked so far: the accepted split, saved ShuttleSet metadata, pilot feature code, label-loading order, Top/Bottom replay and complete-rally scoring
- Plan section: `plan.md`, “Current change: train and compare HGB and RF”

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
- Commit: pending
