1. Question investigated

Read-only history review of `main` at `8c67569`, covering `593eb32^..main`:

- 37 commits total.
- 16 non-merge commits touched `src/annotator/**` or `src/scraper/**`.
- 12 merge commits were present.

2. Files/symbols inspected

Inspected Git stats, diffs and follow-history for:

- `src/annotator/rally_segmentation.py`
- `src/annotator/run_video.py`
- `src/annotator/video_outcomes.py`
- `src/annotator/rally/*.py`
- renamed scraper modules, including `search_index.py`
- `manual_broadcast_timeline_annotator.py`
- related calibration, contract and test files

Serena symbol overviews, references and declarations covered `segment_video`, `run_video`, `find_rally_spans`, `build_contact_data`, and current rally modules.

3. Concrete evidence

- PR #55, merge `1836501`, tip `83f6284`: message was “Replace pipeline stage numbers with responsibility names”. It modified eight annotator files and seven scraper files. Scraper renames were:

  - `stage1_index.py` -> `search_index.py`
  - `stage2_transcripts.py` -> `transcript_acquisition.py`
  - `stage3_triage.py` -> `relevance_triage.py`
  - `stage10_clean.py` -> `commentary_cleaning.py`
  - `stage11_pairing.py` -> `commentary_pairing.py`

  `Stage8Thresholds` became `RallySegmentationThresholds`; the current declaration and uses are at `src/annotator/config.py:46`, `:123` and `:167`.

- PR #58, merge `73c31e3`, tip `db33b85`: message body says to move trajectory, serve, span, sticky-evidence, contact and CLI work into focused modules while retaining `rally_segmentation` as a compatibility facade. It added six responsibility modules plus `__init__.py` and reduced `rally_segmentation.py` (`+1662/-1439` across source and tests).

- PR #59, merge `21e9dc5`: source changes were `run_video.py` and new `video_outcomes.py` (`+938/-474`). Its commits were “Refactor run_video into staged helpers” and “Extract run_video outcome stages”.

- PR #60, merge `aa5e260`: message was “Centralise annotator shared rules”. Source files were 11 annotator files. The commit sequence:

  - `85853bf`: moved `ScalingKind` into `fps_constants.py`.
  - `45daaee`: shared calibration tolerances and `safe_f1`.
  - `c32c492`: derived run metadata from executable policy.
  - `4272cd0`: derived FPS override fields from `FpsConstants`.
  - `a6fafb9`: routed landing options through shared FPS rules.

  Current evidence includes `src/annotator/fps_constants.py:23-38`, `:74`, `src/annotator/calibration/selection.py:63`, `:147`, and `src/annotator/calibration/sweep.py:106`.

- PR #61 was a source-touching non-play-label merge. `d249466` renamed `non_play_labels.py` to `broadcast_timeline_labels.py` and `non_play_annotation.py` to `manual_broadcast_timeline_annotator.py`.

- PR #62, merge `784282b`, message “Bootstrap broadcast annotation from scene cuts”. The source change was confined to `manual_broadcast_timeline_annotator.py`. It added scene-partition reading, scene lookup and scene-based label commits at `src/annotator/manual_broadcast_timeline_annotator.py:252`, `:341` and `:366`, plus `--scene-csv` at `:302`.

- PR #63, merge `8c67569`, message “Clarify annotator contracts and terminology”. All listed source paths were modified, not renamed. Changes included:

  - typed calibration contract records in `gt_scoring.py`;
  - `StickyResult` contract documentation at `src/annotator/types.py:162-193`;
  - explicit run input annotations in `run_video.py`;
  - `_load_replay_mask` -> `_load_dead_mask` at `src/annotator/rally/cli.py:42`;
  - corner-order constants at `src/annotator/point_winner.py:93` and `src/annotator/court_evidence.py:37`;
  - calibration names `boundary_report_key_fewest_merges` -> `boundary_report_key_fewest_swallowed_rallies` and `contact_live_key_floored_f1` -> `contact_live_key_raw_f1`.

File churn, counting individual commits without following renames:

- `src/annotator/calibration/gt_scoring.py`: 4 commits
- `src/annotator/calibration/scoring.py`: 4 commits
- `src/annotator/run_video.py`: 4 commits

No other current target path reached four commits.

Rally split symbol mapping:

- `_rolling_mean`, `_nan_rolling_mean`, `apply_replay_mask`: `db33b85:src/annotator/rally_segmentation.py:126,142,868` -> `src/annotator/rally/trajectory.py:8,24,40`.
- `CourtGeo`, `court_scale_slots`, `tracker_segments`, `build_sticky_result`: old `:161,601,1071,1120` -> `src/annotator/rally/evidence.py:22,42,64,249`.
- Serve symbols (`ServeSetupInputs`, `series_drift`, `serve_setup_still`, `build_serve_setup_inputs`, `ServeStartMode`, `ServeStartClose`, `ServeStartOptions`, serve helpers): old `:177-749` -> `src/annotator/rally/serve.py:14-363`.
- Span helpers and `find_rally_spans`: old `:402-1228` -> `src/annotator/rally/spans.py:35-341`.
- Contact helpers and `assemble_contacts`: old `:915-1259` -> `src/annotator/rally/contacts.py:31-187`.
- CLI helpers and `main`: old `:1376-1435` -> `src/annotator/rally/cli.py:21-173`.
- `scale_thresholds` and `segment_video` stayed in the facade: old `:106,1283` -> `db33b85:src/annotator/rally_segmentation.py:111,131`.
- The post-split facade delegates `find_rally_spans` and `assemble_contacts` at `db33b85:src/annotator/rally_segmentation.py:197-214`.

Run-video staging symbol mapping:

- `scoring_filter`: `1d0e3ba:src/annotator/run_video.py:22` -> `f3cfbf5:src/annotator/video_outcomes.py:102`.
- `_record_rejection` and `_record_trusted_mask_contact_rejection`: old `:48,77` -> `video_outcomes.py:165,199`.
- `LandingHorizonRow` and `_first_stroke_half`: old `:166,187` -> `video_outcomes.py:19,111`.
- `_ContactData` and `_VerdictData`: `1cabe7e:src/annotator/run_video.py:230,243` -> `ContactData` and `VerdictData` at `video_outcomes.py:41,54`.
- `_build_contact_data`, `_build_verdict_data`, `_build_hit_heights`: `1cabe7e:src/annotator/run_video.py:532,578,754` -> `build_contact_data`, `build_verdict_data`, `build_hit_heights` at `video_outcomes.py:119,443,518`.
- New run-stage helpers remained in `run_video.py`: `_validate_landing_horizons`, `_span_options`, `_validate_run_inputs`, `_empty_result`, `_injected_contact_rows`, `_finalize_exclusion_mask`, `_run_court_optional_segmentation` and `_run_court_segmentation`.
- Final orchestration calls those stages at `f3cfbf5:src/annotator/run_video.py:503-557`.

4. Callers/consumers found

- `run_video` is imported or called by `src/annotator/calibration/gt_scoring.py:19,735`, `calibration/sweep.py:49,418`, `e2e_court_annotator.py:69,984`, and `rally/cli.py:215`.
- Current `run_video.py` calls `rally_segmentation.segment_video` at lines `325` and `417`, and `find_rally_spans` at `335`, `373` and `390`.
- `segment_video` references also occur in the rally tests and configuration tests.
- `rally_segmentation.py` re-exports split-module symbols through imports at `src/annotator/rally_segmentation.py:54-108`.
- `resolve.py:10` and `replay_mask.py:34` still import from the facade.
- Serena declaration lookup resolved the staged `segment_video` calls to `src/annotator/rally_segmentation.py:131-215`.

5. Counterevidence / surprises

- No pure target-file deletions remained after rename detection. Git detected seven source renames: five scraper stage files and two non-play annotator files.
- Current Python code and tests contain no matches for the old module names `stage1_index`, `stage2_transcripts`, `stage3_triage`, `stage10_clean`, `stage11_pairing`, `non_play_annotation`, `non_play_labels` or `Stage8Thresholds`.
- The `rally_segmentation` path itself remains active as a facade and test target.
- `replay_mask` remains an API parameter at `src/annotator/rally_segmentation.py:136` even though the CLI loader now calls the input a `dead_mask`.
- `tests/test_fps_constants.py:161` retains `stage8` in a test function name.

Test-heavy commits:

- `83f6284`: renamed ten test paths, including the scraper stage tests and three stage-numbered annotator tests; also modified `test_annotator_run_video.py` and `test_fps_cli_and_tracknet_modes.py`.
- `6749c32`: added `test_non_play_annotation.py` and `test_non_play_labels.py`.
- `8092450`: added `test_non_play_measurement.py`.
- `d249466`: renamed the three non-play test files to broadcast-timeline and replay/serve names.
- PR #60 modified eight calibration, FPS, court-evidence and sweep test files.
- PR #62 added `tests/test_manual_broadcast_timeline_annotator.py`.
- PR #63 modified `tests/test_calibration_selection.py`, `tests/test_court_evidence.py` and `tests/test_fps_cli_and_tracknet_modes.py`.
- PR #59 changed no test files.

6. Unresolved or dynamic surfaces

- Serena search found `getattr` in current field/schema access at `src/annotator/e2e_court_annotator.py:271,584`, `court_evidence.py:143,223`, `calibration/sweep.py:296-320` and `calibration/gt_scoring.py:701`. None referenced an old module path.
- No `project.scripts` or `console_scripts` matches were found in the checked packaging files. Module CLI entry points use `if __name__ == '__main__'`, including `rally_segmentation.py:223` and the current scraper modules.
- Current tests monkeypatch facade targets such as `tests/test_annotator_config_strategies.py:84` and `tests/test_annotator_run_video.py:272-281`; these are dynamic consumers of the compatibility path.
- The Serena launcher could not acquire its read-only state lock because the filesystem rejected `/home/ariel/.local/state/serena-pyrefly/launcher.lock`; the already-active Serena project served the requested queries.
- No files were changed. No tests, lint or type checks were run. Working-tree status showed existing `.gitignore` changes plus untracked `.serena/` and `scratch/review_2026_aug_07/`.
- Diffs were read directly from Git objects using the merge-parent ranges for `1836501`, `73c31e3`, `21e9dc5`, `aa5e260`, `784282b` and `8c67569`; no diff artefact was written.