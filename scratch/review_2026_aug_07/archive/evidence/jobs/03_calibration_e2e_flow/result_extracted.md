## 1. Question investigated

Trace calibration and e2e orchestration on `main`, including entry points, call chains, artefacts, production-pipeline calls, consumers and validation-overlay surfaces. Line references are 1-based.

## 2. Files/symbols inspected

- `src/annotator/e2e_court_annotator.py`
- `src/annotator/calibration/{fixtures,scoring,gt_scoring,selection,sweep,schemas,run_cli}.py`
- `src/annotator/run_video.py`
- `src/annotator/validation_overlay/**`
- Relevant tests and `scripts/plots/plot_stage8_tradeoffs.py`
- Serena symbol overviews, symbol bodies, references, declarations and pattern searches.

No files changed. No `result.md` was written because access is read-only.

## 3. Concrete evidence

### Entry points and call chains

- `run_cli.py:33-43` defines `--fixtures`, `--out` and `--no-replay-mask`.
- `run_cli.py:123-130` calls `build_parser()` then `run_manifest(...)`.
- `run_cli.py:95-120` validates the fixture root and registry, calls the runner per selected fixture, flattens and renders scores, and optionally writes `<out>/<fixture>_metrics.csv`.
- Default chain: `run_manifest` -> `gt_scoring.run_fixture` -> `build_run_video_inputs` -> `run_video` -> `score_video` -> `flatten_metrics`/`render_table` (`run_cli.py:22-27`, `gt_scoring.py:722-738`).
- `sweep.py:668-697` parses `--fixture`, `--out-dir`, `--workers`, `--phase`, optional winner JSON and mask, then calls `run_sweep`.
- `sweep.py:444-542` runs boundary and/or contact phases, applies selection functions, writes CSVs and writes `config_winner.json`.
- `e2e_court_annotator.py:1332-1348` parses `--manifest` and `--device`, then calls `_run_cli_measurement`.
- `e2e_court_annotator.py:1307-1329` creates a timestamped run directory, calls `run_annotator_measurement`, then writes summary/report data and calls `clean_run`.
- Module guards are present at `run_cli.py:133-134`, `sweep.py:699-700` and `e2e_court_annotator.py:1348-1349`.

### Fixed e2e measurement

- Parents are `static_shuttleset_homography` and `detected_ckn_opencv_consensus` (`e2e_court_annotator.py:73-76`).
- Four fixed cases are declared at `e2e_court_annotator.py:121-148`.
- `_setup` creates the Cartesian product of both parents and all cases, producing eight configuration states (`e2e_court_annotator.py:1200-1207`).
- `_setup` reads the source commit, requires a clean tracked tree, parses the input manifest, loads GT tables, loads cases, writes raw cuts, creates the detector and builds configuration states (`e2e_court_annotator.py:1145-1207`).
- Static configurations call `build_static_court_evidence`; detected configurations call `detect_scene_evidence` and `build_detected_court_evidence` (`e2e_court_annotator.py:941-957`).
- Each successful configuration calls `resolve(...)`, constructs `RunCapture`, then calls `run_video(...)` directly (`e2e_court_annotator.py:975-1006`).
- Scoring is deferred until all inference configurations finish. `_score_configurations` first verifies eligible GT files, then calls `_write_scoring_outputs` per successful inference state (`e2e_court_annotator.py:1210-1247`).

### e2e top-level classes

- `FixedCase` (`e2e_court_annotator.py:136-146`): fixed case identifier, fixture, stride, producer mode, FPS and frame count.
- `InputManifest` (`e2e_court_annotator.py:151-160`): pinned videos, track override, CourtKeyNet files and producer names.
- `CaseData` (`e2e_court_annotator.py:163-185`): loaded arrays, video path, raw cuts and case status.
- `ConfigurationState` (`e2e_court_annotator.py:187-205`): one parent/case execution state and retained outputs.
- `RunDriver` (`e2e_court_annotator.py:208-234`): run-wide inputs, detector, cases, configurations and failure paths.
- `VideoMetadata` (`e2e_court_annotator.py:236-243`): FPS, frame count and dimensions read from a video.

### e2e top-level functions

- `utc_now` (`:245`): returns an RFC 3339 UTC timestamp; `_md5_bytes` (`:250`) returns an MD5 digest; `_artifact_record` (`:254`) records relative path, MD5 and byte count.
- `_json_ready` (`:263`): converts enums, NumPy values, dataclasses, named tuples, mappings and sequences to JSON values; `_integer_like` (`:294`) tests integer parsing.
- `_write_json` (`:302`): serialises a value and writes formatted JSON; `_csv_value` (`:309`) normalises CSV values; `_write_rows` (`:323`) writes CSV headers and rows.
- `_relative_pin_path` (`:332`) returns a POSIX pin path; `_pin_from_json` (`:336`) validates one manifest pin; `parse_input_manifest` (`:356`) validates the complete fixed manifest schema.
- `verify_selected_pins` (`:405`) resolves and verifies selected pins; `_pin_path` (`:412`) resolves a pin against `fixtures_root()` or `REPO_ROOT`; `_pin_record` (`:423`) builds a pin record.
- `_array_pin_record` (`:439`) adds array shape and dtype; `_plain_pin_record` (`:445`) builds a non-array pin record.
- `_fixture_by_name` (`:451`) maps fixture names; `_load_array` (`:455`) loads a pinned NumPy array; `_validate_arrays` (`:459`) checks array shapes, dtypes, finiteness and detection counts.
- `probe_video` (`:488`) reads OpenCV video metadata; `validate_video_metadata` (`:503`) checks FPS, frame count and `512x288` dimensions.
- `_raw_cut_rows` (`:512`) converts cut tuples to CSV rows; `_save_mask` (`:519`) writes a one-dimensional boolean `.npy` mask.
- `_corner_values` (`:526`) flattens corner arrays; `_scene_row` (`:533`) serialises one court scene record; `_landing_fields` (`:562`) serialises one landing.
- `_horizon_row` (`:571`) serialises one landing-horizon record; `_strict_metrics` (`:592`) calculates contact counts, precision, recall, F1 and offsets.
- `_landing_metrics` (`:620`) aggregates the configured one-, two- and three-second horizons; `_configuration_values` (`:633`) returns fixed run settings.
- `_device_record` (`:654`) records requested and resolved devices; `_source_commit` (`:658`) runs `git rev-parse HEAD`.
- `_require_clean_source_tree` (`:668`) runs `git status --porcelain --untracked-files=no`; `validate_output_root` (`:677`) resolves an output path and requires it not to exist.
- `_configuration_path` (`:684`) joins run root, parent and case ID; `_failure_payload` (`:688`) builds failure metadata and traceback; `_write_failure` (`:710`) writes it.
- `_shared_pins_by_fixture` (`:721`) groups shared GT pins by fixture; `verify_eligible_gt_files` (`:730`) checks per-set CSV membership and hashes.
- `_input_records` (`:743`) records video, track, pose, resolution and homography inputs; `_load_case` (`:761`) verifies and loads arrays, probes the video and derives raw cuts.
- `_write_raw_cuts` (`:782`) writes shared raw cuts; `_write_scene_evidence` (`:789`) writes court CSVs and masks.
- `_write_annotations` (`:805`) writes `annotations.json`; `_write_landing_horizons` (`:809`) writes sorted landing rows.
- `_write_scoring_outputs` (`:818`) writes contact CSVs, calls scoring functions and writes `metrics.json`.
- `_gt_rallies_for_fixture` (`:854`) imports and calls `load_gt_rallies`; `_configuration_manifest` (`:860`) records configuration metadata and artefacts.
- `_write_terminal_configuration_manifest` (`:918`) writes one configuration manifest; `_run_one_configuration` (`:925`) runs court evidence, production annotation, capture and inference artefacts.
- `_write_scene_evidence_partial` (`:1024`) writes partial court evidence after consensus failure; `_configuration_summary` (`:1034`) returns status and artefact references.
- `_not_run_configuration_summaries` (`:1045`) creates eight `not_run` summaries; `_environment` (`:1058`) records runtime and package information.
- `_run_manifest` (`:1081`) builds the terminal run manifest and status; `_make_detector` (`:1136`) constructs the shared CourtKeyNet detector.
- `_setup` (`:1145`) prepares all run-wide state and configuration states; `_score_configurations` (`:1210`) performs deferred GT verification and scoring.
- `_write_initial_run_files` (`:1249`) writes the input manifest and run log; `run_annotator_measurement` (`:1260`) executes the fixed run.
- `_run_cli_measurement` (`:1307`) wraps the measurement with run reporting and cleaning; `build_parser` (`:1332`) defines the CLI; `main` (`:1339`) delegates to the CLI wrapper.

### e2e internal call graph and artefacts

- `main` -> `_run_cli_measurement` -> `run_annotator_measurement` -> `_write_initial_run_files` -> `_setup` -> `_run_one_configuration` eight times -> `_score_configurations` -> `_run_manifest` (`e2e_court_annotator.py:1283-1300`).
- `_run_one_configuration` -> court-evidence builder -> `_write_scene_evidence` -> `run_video` -> `_save_mask`, `_write_annotations`, `_write_landing_horizons` (`e2e_court_annotator.py:941-1013`).
- Shared artefacts: `shared/<case_id>/raw_cuts.csv` (`e2e_court_annotator.py:782-786`).
- Per-configuration artefacts: `court_scenes.csv`, `scene_rows.csv`, `keep_vote.npy`, `court_present.npy` (`:789-802`); `annotations.json` (`:805-806`); `landing_horizons.csv` (`:809-815`); `raw_replay_mask.npy` and `definitive_exclusion_mask.npy` (`:1008-1013`).
- Per-configuration scoring artefacts: `strict_contacts.csv`, `wide_edge_contacts.csv` and `metrics.json` (`:827-850`).
- Per-configuration terminal files: `manifest.json` and, on failure, `failure.json` (`:918-922`, `:961-972`, `:1014-1019`).
- Run-root files: `input_manifest.json`, `run.log`, optional `setup_failure.json`, optional `scoring_failure.json` and terminal `manifest.json` (`:1249-1257`, `:1287-1300`).

### Calibration flow and artefacts

- `fixtures.py:76-111` derives fixture paths for track, pose arrays, dead mask, court-present mask and scene rows.
- `fixtures.py:127-128` names the homography and resolution sources; `_CALIBRATION_GEOMETRY` is loaded at import (`:243-269`).
- `fixtures.py:356-370` declares the three fixtures and shared files: `shots_master.csv`, `homography.csv`, `my_raw_video_resolution.csv` and the per-set `set1.csv`/`set2.csv`/`set3.csv` files.
- `gt_scoring.py:411-429` verifies and loads fixture arrays plus the three shared tables.
- `gt_scoring.py:432-485` additionally loads `court_present.npy` and `scene_rows.csv`, grades the track and assembles `RunVideoInputs`.
- `scoring.py` is in-memory scoring: `score_boundaries` is at `:170-226`; `score_contacts` is at `:537-620`; neither writes files.
- `gt_scoring.py:564-682` calls `load_gt_rallies`, `load_set_tables`, `reconcile_sets`, `classify_all`, `score_boundaries`, `score_contacts` and `greedy_match`.
- `gt_scoring.py:741-783` writes optional `<out>/<fixture>.csv`, `<out>/<fixture>_geometric_verdicts.csv`, `<out>/<fixture>/codes.npy` and `<out>/<fixture>/rejections.csv`.
- `gt_scoring.py:799-824` runs all fixtures and writes those files when invoked with `--capture --out`.
- `selection.py:36-224` returns sort keys, filtered rows, winners and floor predicates; it has no file-writing calls.
- `sweep.py:97-127`, `:130-163` and `:348-405` turn scoring rows into boundary/contact reports.
- `sweep.py:462-531` writes `boundary_sweep.csv`, `best_config_comparison.csv`, `alignment_own_covered.csv`, `alignment_shared.csv`, `split_log.csv`, `contact_sweep.csv`, `contact_frontier.csv`, `contact_stability.csv` and `config_winner.json`.
- `sweep.py:592-655` reads and validates a prior `config_winner.json`; `sweep.py:658-665` reads an optional boolean mask `.npy`.
- `schemas.py:123-132` maps each sweep CSV filename to its columns; `schemas.py:170-205` builds the winner JSON document.

### Scoring overlap

- `gt_scoring.py:24-32` imports `RallyBoundary`, `classify_all`, `greedy_match`, `load_gt_rallies`, `safe_f1`, `score_boundaries` and `score_contacts` from `scoring.py`; these are shared functions, not duplicate definitions in `gt_scoring.py`.
- Base-30 scaling is duplicated in shape: `scoring.py:267-269` uses `int(ScalingKind.FRAME_COUNT.scale(base_frames, fps))`; `gt_scoring.py:403-404` uses `int(ScalingKind.FRAME_COUNT.scale(5.0, fps))`.
- F1 is centralised in `scoring.py:57-60`: `return 0.0 if denominator == 0 else 2 * precision * recall / denominator`.
- `scoring.py:439-451` computes recall, precision and F1 from matched/GT/candidate counts and calls `safe_f1`.
- `gt_scoring.py:399-400` has the similar ratio formula `return numerator / denominator if denominator else None`.
- `gt_scoring.py:706-709` independently forms contact precision and recall, then calls the imported `safe_f1`.
- `gt_scoring.py:571-601` calls `score_contacts` for the count gate but separately performs per-rally `greedy_match`, timing counts and contact matches.

### Production annotator invocation

- `gt_scoring.py:19` directly imports `run_video`; `run_fixture` calls it at `:735`.
- `sweep.py:49` directly imports `run_video`; `production_candidate_runner` calls it at `:408-419`.
- `e2e_court_annotator.py:69` directly imports `run_video`; `_run_one_configuration` calls it at `:984-1006`.
- The e2e `subprocess.run` calls are only `git rev-parse HEAD` and `git status` (`e2e_court_annotator.py:658-674`). Its stored `python -m annotator.e2e_court_annotator` command is metadata passed to `RunDriver` (`:1268-1279`), not a subprocess pipeline call.
- Sweep parallelism uses `ProcessPoolExecutor` around the direct production runner (`sweep.py:178-198`).

### Validation overlay

- The documented executable is `python -m annotator.validation_overlay.overlays.shuttle_track` (`shuttle_track.py:1-5`).
- `shuttle_track.main` calls `probe_video`, `read_segments`, `load_track`, `make_render_plan`, `make_draw` and `render` (`shuttle_track.py:99-130`).
- `core/cli.py:147-203` builds the timeline and HUD plan; `:259-303` composes decoded frames and draws overlays; `:311-352` encodes and atomically replaces the output.
- `core/decode.py:110-179` uses `ffprobe`; `:182-286` uses `ffmpeg` to decode spans. `core/encode.py:57-105` uses `ffmpeg` to encode.
- The source import graph contains overlay imports only within `src/annotator/validation_overlay/**`; no maintained annotator module imports it.
- Overlay tests import the core and shuttle-track surfaces (`tests/test_validation_overlay_core.py:15-17`, `tests/test_validation_overlay_assembly.py:11-14`).

## 4. Callers/consumers found

- e2e is imported by `tests/test_annotator_measurement.py:14` and called at `:296`, `:596`, `:630`, `:657` and `:680`.
- `_run_cli_measurement` is called by `tests/test_annotator_experiment_records.py:73`, `:95`, `:98` and `:106`.
- No `src/annotator` module imports `annotator.e2e_court_annotator`; the source search found only the module’s own definitions.
- `selection.py` is imported by `sweep.py:25`; its production calls are at `sweep.py:97-109`, `:130-145`, `:474-515`.
- `scripts/plots/plot_stage8_tradeoffs.py:67-69` reads `boundary_sweep.csv` and `boundary_crowns.csv`; `:235-236` reads `contact_sweep.csv` and `contact_frontier.csv`.
- Sweep tests read all current CSV outputs and `config_winner.json` (`tests/test_sweep.py:122-129`, `:207-208`, `:240-243`).
- Schema helpers are imported directly by schema and sweep tests (`tests/test_calibration_schemas.py:7-20`, `tests/test_sweep.py:13-17`).
- Package `__init__.py` files contain docstrings only and no observed symbol re-exports (`calibration/__init__.py:1`, `validation_overlay/__init__.py:1`, `core/__init__.py:1`, `overlays/__init__.py:1`).

## 5. Counterevidence / surprises

- The e2e module docstring describes a “conceptual skeleton” while the code runs the fixed eight-state flow and writes terminal manifests (`e2e_court_annotator.py:1-7`, `:1260-1304`).
- The current sweep producer writes `best_config_comparison.csv` (`sweep.py:462-464`), while the plotting consumer reads `boundary_crowns.csv` (`plot_stage8_tradeoffs.py:60-69`).
- e2e setup temporarily restricts `gt_scoring_module.SHARED_FILES` to the first three shared files and defers per-set GT membership verification until scoring (`e2e_court_annotator.py:1154-1164`, `:1214-1223`).
- The current checkout status includes a tracked `.gitignore` change. `_require_clean_source_tree` rejects non-empty tracked status output (`git status` result; `e2e_court_annotator.py:668-674`).

## 6. Unresolved or dynamic surfaces

- `ANNOTATOR_FIXTURES_ROOT` and all external fixture paths are runtime-configured (`fixtures.py:115-122`).
- `load_set_tables` and e2e GT verification use runtime `*.csv` directory globs (`gt_scoring.py:488-494`, `e2e_court_annotator.py:734-740`).
- e2e uses dynamic `getattr` for serialisation, verdict fields and detector device (`e2e_court_annotator.py:271-274`, `:571-585`, `:1140-1142`); `gt_scoring.py:701-703` dynamically selects aggregate fields.
- Tests monkeypatch imported e2e names including `run_video`, `score_video`, `flatten_metrics`, `load_gt_tables` and `detect_scene_evidence` (`tests/test_annotator_measurement.py:455-498`, `:529-594`).
- No lint, type check or tests were run; this was read-only evidence collection. `git diff --check` exited 0. Serena MCP structural calls succeeded; launcher status could not acquire its lock on the read-only filesystem.