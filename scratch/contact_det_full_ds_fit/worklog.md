# Worklog

## Pick up from here

- Current work: save the same checked contact features for every video in the 40-video list
- Next action: commit the feature-saving code, then compare its three pilot outputs with the saved pilot rows
- Checked so far: the accepted split, saved ShuttleSet metadata, pilot feature code, label-loading order, Top/Bottom replay and complete-rally scoring
- Plan section: `plan.md`, “Current change: prepare features for any listed video”

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
- Commit: will be included in `Freeze contact features for any video roster`
