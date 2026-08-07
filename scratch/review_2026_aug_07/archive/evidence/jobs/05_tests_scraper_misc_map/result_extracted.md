## 1. Question investigated

Mapped the in-scope scraper, calibration, validation, migration, data-access and extraction tests to production symbols and observable contracts on `main`.

`test_e2e.py` does not exist. `test_network.py` is excluded because it targets `bric.network`, not the scraper: `tests/test_network.py:1-15`.

## 2. Files/symbols inspected

Inspected the fresh indexes at:

- `scratch/review_2026_aug_07/evidence/collected_tests.txt`
- `scratch/review_2026_aug_07/evidence/tests_per_file.txt`

Inspected all 24 scoped test files, `tests/conftest.py:13-61`, the relevant `src/` and `scripts/` modules, `.github/AGENTS.md`, and `.codex/context.md`.

Used Serena symbol overviews, symbol/reference searches, declaration lookup and pattern searches. Cross-checked imports, CLI guards, monkeypatch targets, dynamic imports and string dispatch with `rg`.

## 3. Concrete evidence

### Scraper tests

- `tests/test_scraper_search_index.py:17,51-63,80-192`: exercises public `search_term_rows`, `flag_doubles`, `duration_out_of_band`, `upload_before_floor` and `build_candidates`; no private helper is directly targeted. Patches paths, `check_ytdlp`, sleep and subprocess behaviour. Assertions cover parsed rows, flags, candidate CSV output and subprocess exit behaviour. The eight `flag_doubles` cases at `:104-115` represent distinct token and keyword behaviours. No shared `conftest` fixture.

- `tests/test_scraper_relevance_triage.py:13,21-203`: directly exercises public `chunk_windows`, `call_triage_llm` and `run_relevance_triage`, plus private `_keep_decision` and `_write_keep_back`. `_call_once`, sleep, directories and `triage_video` are patched. Assertions cover decisions, retry counts, output flags and batch failure handling. No parametrisation over four cases and no shared fixture.

- `tests/test_scraper_transcript_acquisition.py:17,23-189`: directly exercises `parse_json3`, `parse_vtt`, `pull_subtitles`, `whisperx_fallback` and `run_transcript_acquisition`. The batch tests patch `acquire_transcript`; subprocess, directories, `check_ytdlp`, sleep and randomness are patched. Assertions cover caption parsing, sidecars, resume behaviour and failure thresholds. No parametrisation over four cases and no shared fixture.

- `tests/test_scraper_download_videos.py:12,31-79,108-796`: exercises `DownloadOutcome`, public `download_all_videos` and `main`, plus private `_completed_outputs`, `_download_one` and `_probe_audio`. Patches `shutil.which`, subprocess probing, `_download_one`, `_probe_audio`, `_check_ytdlp`, configuration and `sys.argv`. Assertions cover command arguments and timeouts, manifests, files, worker outcomes and CLI exit behaviour. The five manifest cases at `:453-476` are distinct malformed TOML/type/field behaviours. No shared fixture.

- `tests/test_scraper_commentary_cleaning.py:13,16-297`: exercises public `run_clean`, `call_clean_llm`, `load_fine_models` and `run_fine`, plus private `_bert_score_device`, `_padded_span` and `_extract_span`. Injects a fake `bert_score` module and patches `_clean_once`, sleep and ffmpeg subprocesses. Assertions cover cleaned sidecars, scores, filtering, retries and ffmpeg arguments. The three device cases at `:147-161` are distinct availability/architecture behaviours. No shared fixture.

- `tests/test_scraper_commentary_pairing.py:12,22-706`: exercises public `pair_video`, `build_video_fps_csv` and `main`, plus private `_believed_replay_in_rally_interior` and `_load_replay_mask`. Patches `sys.argv`, `cv2.VideoCapture` and replay-mask loading. Assertions cover paired CSV rows, source mapping, eligibility and error ordering. The six manifest cases at `:406-453` are distinct missing, duplicate, type and eligibility failures. No shared fixture.

- `tests/test_scraper_doubles_flag.py:18-218`: exercises public `doubles_flag`, `read_whole_video_flags` and the doubles CLI. Dynamically imports the rally segmentation modules at `:78-94,156-158`; patches `run_video.run_video` and `sys.argv`. Assertions cover CSV output, logs and processed IDs. The three invalid-literal cases are at `:138-146`. Uses shared `write_doubles_flags` from `tests/conftest.py:29-38`.

- `tests/test_scraper_composition_mask.py:10,21-97`: directly exercises public `build_composition_mask` and inspects `CompositionSegment` values. No mocks, parametrisation or shared fixture. `detect_cuts` is not directly targeted.

### Calibration tests

- `tests/test_calibration_scoring.py:5-183`: directly exercises `CONTACT_TOLERANCES_BASE30`, `RallyBoundary`, `GtRally`, `safe_f1`, `load_gt_rallies`, `merged_span_indices`, `classify_rally_boundary`, `score_boundaries` and `score_contacts`, plus `greedy_match`. The five `safe_f1` cases at `:27-35` cover zero-denominator and ordinary numeric inputs. Assertions are returned enums, records and metric dictionaries; no mocks or shared fixtures.

- `tests/test_calibration_selection.py:5-209`: directly exercises all imported selection functions, including boundary keys, contact floors, coverage allowance and winner selection. Assertions are pure key and selection results. No mocks, parametrisation over four cases or shared fixture.

- `tests/test_calibration_schemas.py:7-105`: directly exercises schema constants, CSV tuples, `winner_spec` and `winner_document`. Assertions pin exact tuples, mappings and JSON shape. Four collected tests; no mocks or shared fixture.

- `tests/test_calibration_run_cli.py:8-120`: exercises public `build_parser` and `run_manifest`; `_validate_registry`, `_validate_environment` and `_write_metrics` are reached indirectly. The local `fixture_root` context manager edits `ANNOTATOR_FIXTURES_ROOT`. Registry, runner, flattener and renderer dependencies are injected. Assertions cover parser errors, metrics files, return codes and stderr. No parametrisation over four cases or shared fixture.

- `tests/test_sweep.py:13-397`: exercises `CandidateSpec`, grid builders, `shipped_spec`, `serialise_spec`, `run_sweep`, `load_boundary_winner` and `main`, plus private row, frontier, CSV, provenance and candidate helpers. Patches grid builders, candidate runners, `sys.argv`, input construction, digesting and file verification. Assertions cover CSV/JSON outputs, winner documents, provenance, call counts and errors. The seven malformed-winner cases at `:306-323` are distinct schema violations. No shared fixture.

### Validation tests

- `tests/test_validation_overlay_core.py:15-241`: exercises `make_render_plan`, `render`, `probe_video`, `iter_span_frames` and timeline classes; `compose_frames` is reached through `render`. Uses actual ffmpeg/video files rather than mocks. The six aspect-ratio cases at `:65-90` cover distinct SAR metadata values. The four rate cases at `:108-125` are below the requested parametrisation threshold. Uses shared `validation_video` from `tests/conftest.py:41-61`.

- `tests/test_validation_overlay_assembly.py:11-117`: exercises `compose_frames`, `make_render_plan`, `render`, `iter_span_frames`, `probe_video`, timeline classes, `make_draw` and `BOX_COLOUR`. Uses actual frame composition, encoding and overlay callbacks. No mocks or parametrisation. Uses shared `validation_video`.

### Remaining scoped tests

- `tests/test_namespace_migration.py:1-14,39-74,91-283,371-486`: imports no production implementation for the tested migration checks. It dynamically imports `bst_x_common`, reads YAML/gzip artefacts, calls `git ls-files`, and scans tracked text. It pins T6 retained weights, T8 sidecar schemas, T10 Chang baseline filenames and T11 legacy namespace/module/extra/environment/venv scans. Four six-directory T8 parametrisations at `:162-203` use six retained run directories, repeating the same schema checks for distinct artefact paths. No mocks or shared fixtures.

- `tests/test_rename_videos.py:8-101`: exercises public `scripts.rename_videos.main`; CSV loading, exclusions and filename parsing are indirect. Patches `sys.argv`. Assertions cover dry-run output, existing-target errors, duplicate IDs and filesystem state. No parametrisation or shared fixture.

- `tests/test_video_metadata.py:8-106`: directly exercises `find_video_files` and `build_resolution_csv`; `main` is not called. Patches `cv2.VideoCapture` and `SET_INFO_DIR`. Assertions cover file resolution, DataFrame contents and unreadable-video reporting. No parametrisation over four cases or shared fixture.

- `tests/test_video_io.py:9,17-44`: directly exercises `get_video_info` with the local `synth_video` fixture. Assertions cover round-trip metadata and missing-file errors. No mocks or parametrisation.

- `tests/test_extract_failure_guard.py:28-328`: exercises `raw_extract.main`, `prepare_dataset_npy_from_raw_video` and constants including `FAILED_CLIPS_LOG`, `RAW_SUFFIXES` and `COCO_N_JOINTS`. Injects `preparing_data.rtmlib_pose` and patches `sys.argv`/`sys.modules`. Assertions cover NPY shapes, failure logs, thresholds, resume denominators and exceptions. No parametrisation or shared fixture.

- `tests/test_download_adapter.py:8,26-57`: exercises public `download_shuttleset_videos`; `_candidate_rows` is indirect. Patches `download_all_videos`. Assertions cover candidate CSV rows and downloader keyword arguments. One collected test; no shared fixture.

- `tests/test_data_access.py:19-450`: directly exercises `DataPaths`, `ClipRecord`, `taxonomy_lookup`, `_derive_class_label`, `get_clip_records`, `summarise`, `_menu` and `interactive`. Patches `builtins.input`; assertions cover labels, filters, paths, exceptions and terminal output. No parametrisation over four cases or shared fixture.

- `tests/test_integration.py:52-142`: exercises `Dataset_npy_collated`, `BST_CG_AP`, `TAXONOMIES` and `taxonomy_lookup` through a real DataLoader/model forward pass. It is skipped when `BST_X_DATA_DIR` is absent at `:60-67`. No mocks or shared fixture.

- `tests/test_remote_preflight.py:44-233`: exercises `taxonomy_lookup`, `derive_npy_collated_dir_basename`, `env_path_or_none` and `load_repo_dotenv`. The two six-way parametrisations at `:104-155` cover distinct taxonomy/split data cells. Assertions cover directory naming, prediction NPZ fields and cell availability. No mocks or shared fixture.

## 4. Callers/consumers found

- Scraper search helpers are called by `build_candidates` at `src/scraper/search_index.py:211,246-249`; `build_candidates` is called by the module entry point at `:276-277`.

- Relevance helpers are called through `triage_video` and `run_relevance_triage` at `src/scraper/relevance_triage.py:194-242`; the batch entry point is `:299-300`.

- Transcript parsers and subtitle fallback are called by `acquire_transcript` at `src/scraper/transcript_acquisition.py:263-273`; the CLI entry point is `:353-354`.

- `download_all_videos` is called by its CLI at `src/scraper/download_scraped_videos.py:467,574-592` and by `src/bst_x/pipeline/download_adapter.py:9,41-55`.

- Cleaning, pairing and doubles helpers are called internally or by their batch/CLI functions at `src/scraper/commentary_cleaning.py:173-214,452-467`, `src/scraper/commentary_pairing.py:330,354,367-368` and `src/annotator/doubles_flag.py:99-130,179`.

- `build_composition_mask` is consumed by `src/annotator/dead_mask.py:8,82`; its CLI guard is `src/annotator/composition_mask.py:173-174`.

- Calibration scoring is consumed by `src/annotator/calibration/gt_scoring.py:566-571` and `src/annotator/calibration/sweep.py:349-362`. Selection functions are consumed by `sweep.py:97-107,486,514`; schema helpers are consumed at `sweep.py:278,522-533`.

- `run_manifest` is called by `src/annotator/calibration/run_cli.py:128`; its module guard is `:133`.

- Overlay functions are consumed by `src/annotator/validation_overlay/core/cli.py:191,240,270,331` and `src/annotator/validation_overlay/overlays/shuttle_track.py:20,112`.

- `find_video_files` and `build_resolution_csv` are consumed by `src/bst_x/pipeline/video_metadata.py:68,135` and `src/bst_x/pipeline/build_dataset.py:35,201-206`.

- `get_video_info` is consumed by `src/bric/preprocessing/extract_shuttle.py:47,111`.

- `prepare_dataset_npy_from_raw_video` is called by `src/bst_x/preparing_data/prepare_train_on_shuttleset.py:998` and the validation smoke script at `src/bst_x/validation_scripts/refactoring/smoke_prepare_2d_bit_exact.py:134-143`.

- `get_clip_records` and `summarise` are consumed by `src/bst_x/pipeline/data_access.py:424,529-539,631-639`. `derive_npy_collated_dir_basename` is consumed by training, inference and validation at `src/bst_x/bst_x_train.py:42,1175`, `src/bst_x/bst_x_infer.py:36,161` and `src/bst_x/validation_scripts/verify_bst_x_train_target.py:23,56`.

- `scripts/rename_videos.main` has only its module guard as a non-test caller: `scripts/rename_videos.py:209-210`. The tested scraper, calibration, adapter, data-access and raw-extraction `main` functions similarly have module guards listed in their respective source files; no additional `src/` or non-archive `scripts/` callers were found.

## 5. Counterevidence / surprises

- Several `test_scraper_*` files target `src/annotator` rather than `src/scraper`: doubles flags and composition masks.
- Batch tests replace core workers: `triage_video`, `acquire_transcript`, `_clean_once`, `_download_one` and `run_video` are patched in the cited test files.
- `src/bst_refactor` is absent while `src/bst_x` exists: `tests/test_namespace_migration.py:335-339`.
- Legacy migration strings still occur in active test/design material, including old module paths and `bst-runtime`: `tests/test_namespace_migration.py:390-414`, `docs/architecture_notes/namespace_migration_test_design.md:249-254`.
- The exact old environment-name strings occur in the migration test and active architecture notes, including `BST_SHUTTLE_CSV_DIR`: `tests/test_namespace_migration.py:417-441`, `docs/architecture_notes/collation_taxon_pin_w_preds_refactor_log.md:307`.
- The model symbol `BST_CG_AP` still exists as an alias and registry key: `src/bst_x/model/bst.py:425-446`. The mixed-case filename text `bst_CG_AP` also remains in active architecture notes and the migration test.

## 6. Unresolved or dynamic surfaces

- `test_remote_preflight.py` can read `.env` through `load_repo_dotenv`; it was not executed, and `.env` was not read.
- `test_integration.py` depends on the external `BST_X_DATA_DIR` tree and may skip without it.
- yt-dlp, ffprobe, ffmpeg, WhisperX, BERTScore, CUDA availability and subprocess behaviour are patched, unavailable or environment-dependent in the cited tests.
- Migration scans depend on dynamic imports, `git ls-files`, tracked artefacts and stage-gating conditions.
- No files were changed. No diff exists. The pre-existing worktree status showed `.gitignore` modified plus untracked `.serena/` and `scratch/review_2026_aug_07/`; none were modified.
- Static inspection commands and Serena queries completed. Ruff, Pyrefly and pytest were not run because this was a read-only evidence census.