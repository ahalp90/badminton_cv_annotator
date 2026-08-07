I’m using the Serena/Pyrefly navigation guidance because this evidence task explicitly requires symbol-level reference checks. I’ll keep the pass read-only and restrict it to the requested import and dynamic-reference surfaces.
The initial repository-wide search found facade consumers in `src/`, tests, calibration code, and several runnable/documented scripts. I’m now separating true imports from prose mentions and checking the facade’s exported names one by one, including dynamic string surfaces.
**Question**

The facade is used by five production modules, fourteen test files, one archived script, and three runnable Python files under `docs/`. Direct `annotator.rally.*` imports occur in the facade, two rally-package internals, and four test files.

**Scope inspected**

Inspected tracked `src/`, `tests/`, `scripts/`, `docs/` including archives, `.github/`, root configuration, `src/annotator/rally_segmentation.py:39-108`, and all `src/annotator/rally/*.py`.

No files were changed. No diff exists.

**Evidence**

Facade importers:

| File | Kind | Names imported or referenced |
|---|---|---|
| `src/annotator/run_video.py:12` | production | `SpanOpen`, `ServeStartOptions`, `build_serve_setup_inputs`, `segment_video`, `find_rally_spans`, `tracker_segments`, `build_sticky_result` at `:49`, `:62`, `:325`, `:335`, `:362`, `:365` |
| `src/annotator/replay_mask.py:34` | production | `compute_speed`, `rolling_nanmedian`, `true_runs` |
| `src/annotator/resolve.py:10` | production | `scale_thresholds` |
| `src/annotator/types.py:18` | production, type-only | `ServeStartClose`, `ServeStartMode` |
| `src/annotator/calibration/sweep.py:48` | production | `ServeStartClose`, `ServeStartMode` |
| `tests/test_annotator_config_strategies.py:12` | test | `segment_video`; module alias `seg` uses `find_rally_spans` at `:190`; `impulse_cell_candidates` at `:195`; indirect facade patch `build_serve_setup_inputs` at `:84` |
| `tests/test_annotator_rally_segmentation.py:16` | test | `CourtGeo`, `ServeSetupInputs`, `ServeStartClose`, `ServeStartMode`, `ServeStartOptions`, `SpanOpen`, `apply_replay_mask`, `compute_speed`, `contact_proximity_ok`, `court_scale_slots`, `detect_contact_flags`, `segment_video`, `span_impulses`, `suppress_contact_flags`, `wrist_contact_near` |
| `tests/test_annotator_run_video.py:7` | test | module alias; `ServeStartClose`, `ServeStartMode`, `StickyResult` at `:12`; patch/attribute names `find_rally_spans`, `segment_video`, `tracker_segments`, `build_sticky_result` at `:108`, `:127`, `:129`, `:133`, `:231`, `:242`, `:278`, `:281`, `:614` |
| `tests/test_annotator_serve_setup.py:7` | test | `ServeSetupInputs`, `series_drift`, `serve_setup_still` |
| `tests/test_annotator_serve_setup_b2.py:8` | test | `ServeSetupInputs`, `ServeStartMode`, `ServeStartOptions`, `StickyResult`, `build_serve_setup_inputs`, `find_rally_spans` |
| `tests/test_annotator_types.py:15` | test | `scale_thresholds` |
| `tests/test_batch_report.py:78` | test | `rally_segmentation` module; `main` at `:90`, `:172`, `:205` |
| `tests/test_fps_cli_and_tracknet_modes.py:66` | test | module alias; `main` at `:93`, `:133`, `:161`, `:176` |
| `tests/test_fps_constants.py:31` | test | `build_sticky_result`, `scale_thresholds`, `segment_video` |
| `tests/test_point_winner.py:37` | test | `ANKLE_L`, `ANKLE_R`, `WRIST_L`, `WRIST_R`, `StickyResult` |
| `tests/test_scraper_doubles_flag.py:157` | test | module alias `segmentation`; `main` at `:174` |
| `tests/test_sticky_result.py:7` | test | `WRIST_L`, `WRIST_R`, `build_sticky_result`, `segment_video` |
| `tests/test_sweep.py:18` | test | `ServeStartClose`, `ServeStartMode` |
| `tests/test_tracker_segments.py:6` | test | `tracker_segments`, `wrist_contact_near` |
| `scripts/archive/autoseg_trials/s28_sticky_pin_anchor_picks.py:78` | script, dynamic | `importlib.import_module`; attributes `tracker_segments`, `build_sticky_result`, `segment_video`, `SpanOpen` at `:111-119` |
| `docs/scraper_pipeline/broadcast_nonstandard_camera_id/measure_replay_and_serve_behaviour.py:52` | doc script | `BODY_UNIT_WRIST_THRESHOLD`, `build_sticky_result`, `compute_speed`, `detect_contact_flags`, `find_rally_spans`, `rolling_nanmedian`, `tracker_segments` |
| `docs/scraper_pipeline/inpaint_hallucination_fix/analysis/audit_tracks.py:41` | doc script | `detect_contact_flags` |
| `docs/tracknet/evidence/inpaint_fabrications_20260722/c11_landing_bisect/instrument_bisect.py:24` | doc script | `ANKLE_L`, `ANKLE_R`, `WRIST_L`, `WRIST_R`, `compute_speed`, `court_scale_boxes`, `rolling_nanmedian`; `court_scale_boxes` is not currently re-exported |

Direct rally-package imports:

| File | Names |
|---|---|
| `src/annotator/rally_segmentation.py:54-93` | Relative imports of `rally.cli`, `rally.contacts`, `rally.evidence`, `rally.serve`, `rally.spans`, and `rally.trajectory`; names are the full import bindings in lines `54-93` |
| `src/annotator/rally/contacts.py:8` | `.trajectory`: `_nan_rolling_mean`, `_rolling_mean` |
| `src/annotator/rally/spans.py:21` | `.serve`: `ServeStartClose`, `ServeStartMode`, `ServeStartOptions`, `_resolve_serve_gate` |
| `src/annotator/rally/spans.py:27` | `.trajectory`: `_rolling_mean` |
| `tests/test_annotator_config_strategies.py:11,183` | `spans`: `_find_rally_spans_quiet_start`, `_gap_state_rest_mask`; module alias uses `_rest_mask` at `:188` |
| `tests/test_annotator_rally_segmentation.py:33-40` | `serve`: `_serve_distance_ratio_passes`; `spans`: `_find_rally_spans`, `_find_rally_spans_span_open`, `_last_rest_close`, `_serve_start_find_rally_spans`; `trajectory`: `_nan_rolling_mean`, `_rolling_mean` |
| `tests/test_annotator_serve_setup_b2.py:7` | `serve`: `_sticky_serve_setup_before` |
| `tests/test_fps_cli_and_tracknet_modes.py:16` | `annotator.rally.cli`; uses `_load_dead_mask` at `:21-22` |

**Production references**

No other `src/` production importer of either module path was found. `.github/`, shell/config files, and root metadata contain no matching imports or invocations.

**Test references**

The test importers and exact names are listed in the Evidence table. No `patch("annotator.rally...")` string targets were found.

**Counterevidence**

Treating “unused” as “no executable user outside the facade”, these facade bindings have no such user:

- `src/annotator/rally_segmentation.py:39-51`: `BaseAnnotatorConfig`, `CONTACT_FRAMES_CSV`, `END_REST_FRAMES`, `PROXIMITY_MAX`, `RALLY_SPANS_CSV`, `REST_SPEED`, `REST_WINDOW`, `SMOOTH_WINDOW`, `START_MIN_FRAMES`, `START_SPEED`, `RallySegmentationThresholds`
- `:52-54`: `read_whole_video_flags`, `FpsConstants`, `scale_for_fps`, `log`, private `_cli_main`
- `:57-68`: `CONTACT_DEDUP_RADIUS_FRAMES`, `CONTACT_IMPULSE_MULTIPLE`, `CONTACT_SUPPRESSION_RADIUS_FRAMES`, `FLOOR_EPS`, `IMPULSE_FLOOR_HALF_WINDOW_FRAMES`, `assemble_contacts`, `rolling_floor`
- `:72`: `BODY_UNIT_HALF_WINDOW`
- `:79`: `PLAYER_PRESENT_MIN_FRAC`
- `:89-90`: `QUIET_START_REST_FRACTION`, `VISIBILITY_REST_FRAC`
- `:97-101`: `ContactCandidate`, `ReentryGuardVariant`, `Slot`, `SmoothingMode`

Serena references and text search independently verified at least three claims:

- `CONTACT_DEDUP_RADIUS_FRAMES`: only facade import `:57` and contacts implementation `src/annotator/rally/contacts.py:14,99`.
- `BODY_UNIT_HALF_WINDOW`: only facade import `:72` and evidence implementation `src/annotator/rally/evidence.py:39,254`.
- `QUIET_START_REST_FRACTION`: only facade import `:89` and spans implementation `src/annotator/rally/spans.py:32,227`.

Documentation-only counterevidence includes `PLAYER_PRESENT_MIN_FRAC` at `docs/scraper_pipeline/annotator_functionality_map.md:293`, `assemble_contacts` at `docs/scraper_pipeline/serve_prepend_lookback/serve_prepend_lookback_20260731-091227.md:341`, and historical shorthand references at `docs/archive/dead_code_clean/findings.md:105,116`. These are not executable users.

**Unresolved/dynamic surfaces**

- `scripts/archive/autoseg_trials/s28_sticky_pin_anchor_picks.py:78` is the only dynamic import of the exact facade path.
- Monkeypatch/module-string surfaces are covered at `tests/test_annotator_config_strategies.py:84,188` and `tests/test_annotator_run_video.py:107-134,242,278-281,613-615`.
- `python -m annotator.rally_segmentation` appears only in the facade docstring at `src/annotator/rally_segmentation.py:32`; no CI, Makefile, shell, or other runnable invocation was found.
- `src/annotator/rally/cli.py:16` contains the logger name string `annotator.rally_segmentation`.
- Archived `scraper.stage8_rally_segmentation` strings are different module paths and were excluded.
- Checks run: Serena `find_symbol`, `find_referencing_symbols`, `search_for_pattern`; `git grep`; `rg`; and a read-only AST import scan. No tests, lint, or type checks were run because no files were changed.
