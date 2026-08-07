## 1. Question investigated

Mapped the 27 scoped annotator-related test files to imported production symbols, visibility, patches, parametrisation, shared fixtures and production callers.

## 2. Files/symbols inspected

Inspected all 27 requested test files, `tests/conftest.py`, the collected-test inventories, and the related modules under:

- `src/annotator/`
- `src/annotator/rally/`
- `src/annotator/calibration/`
- `src/bst_x/preparing_data/`
- `src/shared/tracknetv3/`
- `scripts/`
- `docs/scraper_pipeline/broadcast_nonstandard_camera_id/`

Structural lookups used Serena symbol overviews, declarations and references. Text search cross-checked re-exports, dynamic imports, monkeypatch targets and CLI guards.

## 3. Concrete evidence

- `test_annotator_rally_segmentation.py:10-42` imports segmentation, contact, serve, replay-mask and configuration symbols. It directly tests public functions plus private helpers such as `_find_rally_spans`, `_serve_start_find_rally_spans`, `_rolling_mean` and `_nan_rolling_mean` at `:81-737`. No mocks or shared fixtures. The only parametrisation above four cases is two `nan`/`inf` cases at `:278-285`.

- `test_annotator_run_video.py:6-14` imports `run_video`, `AnnotatorResult`, `RunCapture`, serve options, point-winner symbols and scoring helpers. It exercises public `run_video` and patched internal decomposition at `:106-137`, `:242`, `:278-585`, `:613`, `:667-693`, `:772` and `:868`. Patches assert both call arguments and resulting masks/results. The eight-case missing-input matrix is at `:190-197`; fields represent distinct missing-input contracts. No shared fixtures.

- `test_point_winner.py:9-39` imports public point-winner classes, enums and functions plus private `_carried_terminal`. Direct coverage is at `:83-632`. No mocks, shared fixtures or parametrisation above four cases. The private helper is exercised at `:291-315`.

- `test_annotator_serve_setup.py:7-8` imports public `ServeSetupInputs`, `series_drift` and `serve_setup_still` through `rally_segmentation`. Its local fixture at `:11-35` depends on `serve_setup_defaults`. No mocks. Seven invalid dtype/count cases are at `:127-145`; six invalid slot cases are at `:121-124`. These are distinct validation forms.

- `test_annotator_serve_setup_b2.py:7-16` imports public serve setup symbols and private `_sticky_serve_setup_before`. Its local fixture at `:19-27` depends on `serve_setup_defaults`. It patches `np.median` at `:110-115` to test the empty-input early-return path. Five count values at `:30-34` exercise threshold and lane behaviour.

- `test_annotator_measurement.py:14-23` imports `e2e_court_annotator` as `runner`, court-evidence records and `AnnotatorResult`. It directly exercises private orchestration helpers including `_strict_metrics`, `_configuration_values`, `_validate_arrays`, `_write_raw_cuts`, `_write_scene_evidence`, `_write_annotations`, `_write_rows`, `_run_one_configuration` and `_score_configurations` at `:51-66`, `:214-259`, `:338-370`. Patches at `:129-152`, `:201`, `:274-294`, `:332-337`, `:439-594` assert stage order, call counts and returned artefacts. No parametrisation above four cases or shared fixtures.

- `test_replay_and_serve_measurement.py:17-24` dynamically loads `docs/scraper_pipeline/broadcast_nonstandard_camera_id/measure_replay_and_serve_behaviour.py`. It calls public measurement functions and private `_write_csv_gz`, `_write_json_gz` and `_build_report` at `:35-282`. It patches `detect_contact_flags` at `:104-109`; assertions concern output behaviour rather than mock call arguments. No parametrisation above four cases.

- `test_annotator_replay_mask.py:13-28` imports public replay-mask functions, private `_cli_non_evidence`, inpaint grading functions and mask constants. It patches `rolling_nanmedian` at `:200-206`, `BaseAnnotatorConfig` at `:477`, `grade_track` at `:489`, `combine_mask` at `:503` and `:515`, and `sys.argv` at `:506` and `:519`. Assertions cover both captured calls and mask/log/file outputs. Six validation combinations are at `:331-347`, covering invalid optional-mask shapes, dtypes and lengths.

- `test_annotator_fixtures.py:11-25` imports calibration fixture records, verification functions, private calibration sources/loaders and `perspective_shift_signal`. It directly tests private `_load_calibration_geometry` at `:192`, `:208`, `:226` and `:237`. No mocks or shared pytest fixtures. Eleven distinct shared-file paths are parametrised at `:121-125`.

- `test_annotator_types.py:9-18` imports configuration classes, FPS scaling, type enums/classes and `resolve`. It tests public/re-exported contracts at `:41-110` without mocks. Twelve cases at `:55-59` combine three scaling kinds with four invalid FPS values: zero, negative, NaN and infinity.

- `test_annotator_config_strategies.py:9-16` imports public configuration/resolution/segmentation symbols and private span helpers. It directly tests `_find_rally_spans_quiet_start` and `_gap_state_rest_mask`; patches `build_serve_setup_inputs` at `:84-85` and `_rest_mask` at `:182-191`. The patches test call shape and returned behaviour. No parametrisation above four cases. It dynamically imports `impulse_cell_candidates` at `:194-200`.

- `test_annotator_scoring.py:7-10` imports calibration fixtures, reference scores, public scoring/report functions and `GtRally` row builders. Tests are at `:34-179`. No mocks. The fixture matrix has three cases at `:34-41`, below the requested threshold.

- `test_annotator_experiment_records.py:14` imports the records module and dynamically exercises the measurement runner. It targets public `utc_run_directory`, `write_summary_and_report` and `clean_run`, plus private `_sanitise_path`, `_candidate_files` and `_run_cli_measurement` at `:68-107`, `:138`, `:163-307`. Patches at `:26-62`, `:91-106`, `:212`, `:272-305` assert scanner calls, event order and generated files. No parametrisation above four cases.

- `test_fps_constants.py:11-32` imports public FPS constants/scaling, configuration, point-winner, segmentation, replay-mask and sticky-result functions. It patches `subprocess.run` at `:149-158` for `probe_fps` error payloads and asserts raised errors. Seven FPS values are used by two tests at `:111-132` and `:188-208`; these are distinct numeric scaling values, including fractional FPS.

- `test_fps_cli_and_tracknet_modes.py:10-12` dynamically imports rally, composition-mask, replay-mask, run-video and shuttle-extractor CLI modules. It directly targets private `_load_dead_mask` at `:20-21` and public CLI/extractor entry points. Patches cover subprocesses and command construction at `:36-45`, `:81-128`, `:216`, `:273`, `:319-334`; assertions mainly inspect call plumbing and resolved options. Six retired-option names are tested at `:144-161`; six command/profile combinations at `:240-289`; five CLI profiles at `:293-325`. These represent distinct parser or mode branches.

- `test_court_evidence.py:8-13` imports court-evidence builders and records, private `_static_corners_refpx`, and point-winner geometry helpers. Patches at `:64`, `:86-87`, `:103`, `:173-174`, `:430-438` and `:524` capture detector, camera and consensus calls while asserting evidence records and output geometry. No parametrisation above four cases. Both public builders and the private corner helper are exercised.

- `test_scraper_composition_mask.py:10` imports only `build_composition_mask`. Tests at `:21-97` directly assert public mask and segment outputs. No mocks, fixtures or parametrisation. The filename says “scraper”, but the imported production module is `annotator.composition_mask`.

- `test_dead_mask.py:7-11` imports public `build_dead_mask`, replay-mask combination and `DeadMaskMode`. It patches `combine_mask` at `:133` and `:148`; one path captures arguments and another asserts the function is not reached. No parametrisation above four cases or shared fixtures.

- `test_inpaint_guard.py:7` imports public `grade_track` and status constants. Tests at `:37-109` assert returned grades, metadata and raised errors. No mocks, fixtures or parametrisation.

- `test_inpaint_sidecar.py:19-27` dynamically loads `write_inpaint_metadata.py`. It tests public `write_inpaint_metadata`, private `_read_source_provenance` at `:378-384`, and AST structure in TrackNet prediction files at `:502-565`. It patches module datetime at `:188` and `open` at `:405`; assertions concern payloads, layout, provenance and failures. Five distinct mask/frame patterns are tested at `:75-93`.

- `test_batch_report.py:10-16` imports public report/path functions and result records. It invokes the rally CLI at `:90`, `:171` and `:205`, patches `run_video` at `:81` and `:164`, and patches `publish_batch_report` at `:197`. Assertions cover report output and guard-call plumbing. It uses shared `write_doubles_flags` at `:148` and `:183`. No parametrisation above four cases.

- `test_first_last_stroke_buffered_search.py:9` imports `main` from `scripts/analyse_first_last_stroke_buffered_search.py`. The test calls the CLI entry point at `:41-111`, asserting CSV and stdout output. No mocks, shared fixtures or parametrisation.

- `test_sticky_anchor.py:19-33` imports public sticky-anchor functions, private `_pick_one_frame` and `_run_clip`, and private compatibility aliases. Direct tests are at `:135-508`. No mocks, shared fixtures or parametrisation. Alias identity is checked at `:455-460`.

- `test_sticky_result.py:7` imports public `build_sticky_result`, `segment_video` and wrist constants. Tests at `:79-200` assert result arrays, shapes and errors. No mocks, fixtures or parametrisation above four cases.

- `test_doubles_overcount.py:25-39` imports public player-count helpers plus private `_order_two_on_court`, `_pick_one_frame` and `_run_clip`. It patches `check_pos_in_court` at `:66`, `:81`, `:97`, `:247` and `:268`; tests assert both short-circuit calls and resulting counts/order. No parametrisation above four cases.

- `test_broadcast_timeline_labels.py:7-19` imports public interval, CSV and validation functions/classes. Five invalid metadata tuples are tested at `:33-45`, covering distinct empty, zero, NaN and boolean cases. It patches `read_label_csv` at `:138` and asserts destination preservation on failure. No shared fixtures.

- `test_manual_broadcast_timeline_annotator.py:8-17` imports public timeline classes/functions, parser helpers and `commit_label`. Tests are at `:31-285`. No production private helper is imported, no monkeypatch is used, and no parametrisation exceeds four cases. `_FakeCapture` is passed to `video_metadata` at `:253-284` as a test double.

## 4. Callers/consumers found

- `segment_video` is consumed by `run_video` decomposition in `src/annotator/run_video.py:335-390` and by the rally segmentation wrapper. `find_rally_spans` is consumed by the rally pipeline and `run_video`; its private span helpers are called by the public dispatcher.

- `build_serve_setup_inputs` is consumed by `run_video.build_serve_options`. `series_drift` is consumed by `serve_setup_still`. `_serve_distance_ratio_passes` is consumed by `_sticky_serve_setup_before`.

- `fit_alternation`, `next_server_half` and `build_hit_height_rows` have production consumers in `src/annotator/video_outcomes.py`. `filtered_descending_landing` and `is_net_ender` are consumed by `pick_landing_to_end`. `inout_verdict` is consumed by `geometric_verdict`.

- `build_sticky_result` is consumed by `src/annotator/run_video.py:364`. `tracker_segments` is consumed by `run_video`.

- `impulse_cell_candidates` is declared in `src/annotator/rally/contacts.py:80-118`; `detect_contact_flags` consumes it. `rally_segmentation.py:64-65` re-exports it.

- `build_raw_cut_intervals` is consumed by the measurement loader. `scene_sample_indices` is consumed by `detect_scene_evidence`. Static and detected court-evidence builders are consumed by the e2e measurement runner.

- `verify_file` is consumed by fixture, ground-truth scoring, sweep and e2e verification paths. `verify_fixture` is consumed by fixture-array loading. `build_run_video_inputs` is consumed by calibration scoring and sweep paths.

- `run_annotator_measurement`, `write_summary_and_report` and `clean_run` are consumed by the e2e CLI runner. `format_batch_report` is consumed by `publish_batch_report`.

- `build_composition_mask` is consumed by `dead_mask.build_dead_mask` and the composition-mask module entry point. `grade_track` and `code_counts` are consumed by calibration scoring and replay-mask CLI code.

- `write_inpaint_metadata` is imported by `src/shared/tracknetv3/predict.py:11` and called at `:361`. `_read_source_provenance` is called internally by the writer.

- `scripts/analyse_first_last_stroke_buffered_search.py:421-422` contains the only semantic reference returned for its `main`, namely its own module guard. The scoped test imports and calls it at `tests/test_first_last_stroke_buffered_search.py:9` and `:78`; no production caller was returned.

- `court_scale_slots` had no production function caller returned by Serena beyond its re-export in `rally_segmentation.py`; the scoped direct test is its consumer.

- The `_pick_one_frame` compatibility alias in `sticky_anchor.py:321` had no semantic references returned. The public `pick_one_frame` is called by `_run_clip`; the alias surface is dynamic.

## 5. Counterevidence / surprises

- `rally_segmentation.py` re-exports many symbols from `rally.contacts`, `rally.serve`, `rally.spans`, `rally.trajectory`, `rally.evidence` and `types`. An import from `annotator.rally_segmentation` does not establish local declaration.

- `test_scraper_composition_mask.py` targets `annotator.composition_mask`, despite its scraper-oriented filename.

- Several tests load modules by path or import them inside test bodies: the replay/serve measurement test, inpaint sidecar test and FPS/TrackNet CLI tests.

- CLI coverage uses module `main` functions, `sys.argv`, subprocess patches and `__main__` guards. No `[project.scripts]` entry-point table was found in `pyproject.toml`.

- `tests/conftest.py:13-26` defines `serve_setup_defaults`, used by the two serve-setup test files. `write_doubles_flags` at `:29-38` is used by `test_batch_report.py`. `validation_video` at `:41-61` had no reference in the scoped files.

## 6. Unresolved or dynamic surfaces

- Serena semantic references do not fully resolve dynamic imports, aliases, monkeypatch targets, re-exports or CLI invocation from shell/module guards.

- The `sticky_anchor._pick_one_frame` alias and the dynamically imported measurement modules require text-level interpretation; no additional caller was inferred.

- External calibration fixtures and environment-selected paths were not consumed during this read-only mapping.

- No files were changed. No diff exists. `git status --short --branch` showed the pre-existing modified `.gitignore` and untracked `.serena/` and `scratch/review_2026_aug_07/`; these were not touched.

- Commands/checks: read-only `sed`, `nl`, `rg`, `awk`, `wc`, `git status`; Serena symbol/declaration/reference/search tools. Lint, type checking and tests were not run because this was evidence collection with no code changes.

- The Serena launcher could not create `/home/ariel/.local/state/serena-pyrefly/launcher.lock` under read-only access, but the already available Serena MCP session remained usable.