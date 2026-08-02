# WP4 raw return (automated read-only sweep, 2026-08-01)

1. CLUSTER SHAPE: `src/scraper` owns documented stages 1, 2, 3 and 11 (`stage1_index.py:13`, `stage2_transcripts.py:10`, `stage3_triage.py:9`, `stage11_pairing.py:13`), plus runnable-undocumented stage 10 and downloader entry points (`stage10_clean.py:437-467`, `download_scraped_videos.py:543-579`). `src/shared` supplies live court, dataset, taxonomy, video-I/O, plotting and player-mapping utilities used by annotator, API, BRIC and `scripts/build_shots_master.py`; `shared.temporal` is only exercised by tests (`tests/test_temporal.py:14`).

2. LEDGER:

WP4-1 | D-unreach | `src/shared/court.py:to_court_coordinate/check_pos_in_court` | The legacy coordinate/filter chain is unreachable from tracked roots because `check_pos_in_court` has no caller and `to_court_coordinate` is called only by it. | evidence: `src/shared/court.py:101-146`, `src/shared/README.md:38-45`; roots/callers checked: CI pytest, Docker FastAPI routes, documented `python -m` roots, `__main__` blocks, `getattr`/`importlib`/registry/`__all__` dispatch, and all `shared.court` imports | delete both functions and their unreferenced documentation | confidence high

WP4-2 | U | `src/shared/dataset.py:SPLITS_V2/_load_splits_v2` | `SPLITS_V2` is computed during import but has no tracked production consumer, while `build_shots_master.py` rereads `SPLITS_V2_PATH` directly. | evidence: `src/shared/dataset.py:160-203`, `scripts/build_shots_master.py:52-62,164-166`; roots/callers checked: CI pytest, Docker FastAPI routes, documented roots, dynamic dispatch, and all `shared.dataset` imports | delete `_load_splits_v2` and `SPLITS_V2`, retaining the used `SPLITS_V2_PATH` | confidence high

WP4-3 | U | `src/shared/taxonomy.py:PLAYERS/UNPREFIXED_TYPES` | `PLAYERS` and `UNPREFIXED_TYPES` have no tracked caller or internal use, unlike the taxonomy registry and stroke mappings around them. | evidence: `src/shared/taxonomy.py:108-111,229-235`, `src/shared/player_mapping.py:21,113`; roots/callers checked: CI pytest, Docker FastAPI routes, documented roots, dynamic dispatch, and all taxonomy imports | delete both unused constants | confidence high

WP4-4 | D-unreach | `src/shared/temporal.py:clip_window_seconds/clip_window_frames` | These helpers have no tracked call sites and are not reached by production or test roots. | evidence: `src/shared/temporal.py:26-88`, `src/shared/README.md:18,44`, `tests/test_temporal.py:14,24-103`; roots/callers checked: CI pytest, Docker FastAPI routes, documented roots, `__main__` blocks, and dynamic dispatch | delete both helpers and their README references | confidence high

WP4-5 | T | `src/shared/temporal.py:subsample_indices` | `subsample_indices` is a test-only surface: tests import and call it, but no production module imports `shared.temporal`. | evidence: `src/shared/temporal.py:91-142`, `tests/test_temporal.py:14,24-103`, `src/shared/README.md:18,44`; roots/callers checked: CI pytest, Docker FastAPI routes, documented roots, `__main__` blocks, and dynamic dispatch | park as an archive candidate until a production adopter is named | confidence high

WP4-6 | D-unreach | `src/shared/video_io.py:iter_frames/read_frames` | `read_frames` has no caller and `iter_frames` is used only by that unreachable helper, so the pair is unreachable from tracked roots. | evidence: `src/shared/video_io.py:46-109`, `src/shared/README.md:17,43`, `tests/test_video_io.py:17-21,54-64`; roots/callers checked: CI pytest, Docker FastAPI routes, documented roots, `__main__` blocks, and dynamic dispatch | delete both helpers and their README references | confidence high

WP4-7 | T | `src/shared/video_io.py:read_frame_at/write_frame_thumbnail` | These functions have no production caller and are retained by tests alone, despite `write_frame_thumbnail` claiming to back an API contract. | evidence: `src/shared/video_io.py:81-157`, `src/bric/perception/players.py:41,239`, `src/bric/preprocessing/extract_shuttle.py:47,111`, `tests/test_video_io.py:17-21,62-146`; roots/callers checked: CI pytest, Docker FastAPI routes, documented roots, and dynamic dispatch | remove the test-only surface and stale API claim if no API owner exists; retain `get_video_info` | confidence high

WP4-8 | U | `src/scraper/config.py:CONCURRENT_FRAGMENTS/DOWNLOAD_WORKERS` | These configuration constants have no consumers, while the downloader hardcodes the same values in its command and worker default. | evidence: `src/scraper/config.py:179-188`, `src/scraper/download_scraped_videos.py:31,315-330,473-479,552-553`; roots/callers checked: CI pytest, documented scraper roots, undocumented `__main__` roots, and dynamic dispatch | delete the two unused constants and their comments | confidence high

WP4-9 | D-prod | `src/scraper/download_scraped_videos.py:download_video` | `download_video` has no production caller; the CLI reaches `download_all_videos`, while only scraper tests call this convenience wrapper. | evidence: `src/scraper/download_scraped_videos.py:429-455,473-540,543-567`, `tests/test_scraper_download_videos.py:99-105`; roots/callers checked: CI pytest, Docker FastAPI routes, documented `python -m` roots, undocumented `__main__` roots, and dynamic dispatch | delete the wrapper and its direct tests; retain the CLI path | confidence high

WP4-10 | C | `src/scraper/stage11_pairing.py:VIDEO_EXTENSIONS/_read_sources_manifest; src/scraper/download_scraped_videos.py:VIDEO_EXTENSIONS/_read_manifest` | The stages duplicate the manifest boundary and extension set, but their validation contracts differ and the local copies deliberately keep pairing independent of downloader internals. | evidence: `src/scraper/stage11_pairing.py:41-44,239-253`, `src/scraper/download_scraped_videos.py:31,126-208`; roots/callers checked: documented `scraper.stage11_pairing`, downloader `__main__`, CI tests, and dynamic dispatch | leave; extracting a shared reader or constant would add coupling without a proven contract reduction | confidence high

3. OUTWARD NOTES:

WP2: `src/shared/court.py:1-20` mirrors `src/bst_x/pipeline/court_utils.py:1-180`; compare drift and isolation rationale.

WP2: `src/shared/dataset.py:207-215`, `src/shared/player_mapping.py:1-5` and `src/shared/taxonomy.py:1-10` mirror BST pipeline modules.

WP2: `src/shared/eval_plots.py:22-137` overlaps `scripts/plots/confusion_matrix.py:41-154` and `src/bst_x/result_utils.py:98-154`.

WP2: `src/scraper/download_scraped_videos.py:1-6` shares downloader shape and throttle literals with `src/bst_x/pipeline/download_videos.py:1-41`.

4. NOT CHECKED: No scraper network, Gemini, WhisperX, BERTScore, CUDA, yt-dlp, ffprobe or video runtime was executed; full lint, type and test gates were not run because this was read-only. Cross-package P/S/C adjudication and directories owned by other work packages were not audited. `.env`, credentials, untracked files and ignored callers were excluded.
