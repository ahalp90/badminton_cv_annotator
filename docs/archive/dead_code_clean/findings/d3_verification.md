# D3 raw return (automated read-only check, 2026-08-02): renderer history, api probe dead-certainty, downloader config check

1. VERDICT: Confirmed. BST-X and BRIC have live callers; the script is a CLI-only renderer; `annotate_cells` and `render_panel` duplicate the shared helper internals.

- Shared renderer: production caller is `src/bric/eval.py:17` (`from shared.eval_plots import plot_confusion_matrix`) and `src/bric/eval.py:187-191` (`plot_confusion_matrix(...)`). No test caller was found. Documentation references include `README.md:52`, `src/shared/eval_plots.py:3-5`, and the audit notes `docs/archive/dead_code_clean/findings/wp2.md:21-23`.
- BST-X renderer: live production caller is confirmed. `src/bst_x/bst_x_train.py:38` imports it, and `src/bst_x/bst_x_train.py:943-951` contains:

  > `if show_confusion_matrix:`  
  > `    plot_confusion_matrix(`

  Its only other executable call is the module self-demo at `src/bst_x/result_utils.py:137-153`. Documentation references are `src/bst_x/data_pipeline_to_model_train.md:505,558` and `docs/architecture_notes/architecture_2_research_10_April.md:228`.
- Script renderer: `annotate_cells` is called only by `render_panel` at `scripts/plots/confusion_matrix.py:90`; `render_panel` is called only by `main` at `scripts/plots/confusion_matrix.py:142-149`; `main` is reached only through the module guard at `scripts/plots/confusion_matrix.py:159-160`. The script usage is documented at `scripts/plots/confusion_matrix.py:12-15` and `README.md:52`. No production import caller or test caller was found.
- The BRIC claim is confirmed. `src/bric/eval.py:17` imports `shared.eval_plots`, and `src/bric/eval.py:187-191` calls it. The contrary audit note at `docs/archive/dead_code_clean/findings/wp5.md:38` is stale.
- The duplicate internals are confirmed. Both implementations contain the same threshold and cell loop: `src/shared/eval_plots.py:24-31` and `scripts/plots/confusion_matrix.py:48-55`. Both render the same heatmap, labels, subtitle and annotations: `src/shared/eval_plots.py:50-65` and `scripts/plots/confusion_matrix.py:74-90`. Their F1 sorting and zero-safe normalisation also match at `src/shared/eval_plots.py:95-110` and `scripts/plots/confusion_matrix.py:114-131`.
- History:
  - Shared renderer: last substantive code commit `4c83fff` (authored 2026-05-24, committed 2026-06-18). `git log -L` identifies it as the body introduction. Later touches `12b30c9`, `fc6aa62` and `2369971` changed source paths or provenance text.
  - Script renderer: last substantive renderer change `ea494e2` (authored 2026-05-31, committed 2026-06-18). `2369971` moved `scratch/presentation_prep/confusion_matrix.py` to `scripts/plots/confusion_matrix.py`; `45f88d5` only changed the usage path in the docstring.
  - BST-X renderer: last substantive code commit `12b30c9` (authored 2026-06-11, committed 2026-06-18). `fc6aa62` was a 100% rename and `68870e5` added only the licence header at `src/bst_x/result_utils.py:1-5`.
- Derivation is determinable. `src/shared/eval_plots.py:3-5` explicitly says it was “Adapted from `src/bst_x/result_utils.py` and `scripts/plots/confusion_matrix.py`”. The script lineage predates that shared file: `d3f5c4b` introduced the presentation script, while `4c83fff` introduced `shared/eval_plots.py`.

NOT CHECKED: No plotting runtime, tests, lint or type checks were run. No files were edited.

2. VERDICT: Confirmed for tracked code. `_summary_live`, `_live_splits` and `is_available` have no tracked callers. `available_splits` is test-only apart from its dead wrapper.

- `_summary_live` is explicitly marked “Currently unused” at `src/api/registry.py:306-309`. Registry list/detail routes call `_summarise_model` at `src/api/registry.py:377-384`; clip listing calls `_summaries_for` at `src/api/registry.py:399-415`. The detail route explicitly says live inference is not called at `src/api/registry.py:469-478`.
- `_live_splits` is defined at `src/api/registry.py:75-85` and only calls `bst_x_inference.available_splits()` at `src/api/registry.py:81-82`. Current summary logic instead uses `_pred_splits` at `src/api/registry.py:198-203`.
- `is_available` is defined at `src/api/bst_x_inference.py:160-167` with no tracked caller. The live route imports and calls `predict` directly at `src/api/main.py:228-230`, then handles `BstXInferenceUnavailable` at `src/api/main.py:266-271`.
- `available_splits` is defined at `src/api/bst_x_inference.py:170-180`. Its only non-test reference is the dead wrapper at `src/api/registry.py:81-82`. The two direct tests are exactly `tests/test_api.py:125-136`.
- FastAPI route/decorator check: the registry router is mounted at `src/api/main.py:185-197`; registered routes are listed at `src/api/registry.py:377-512`; the library route is at `src/api/main.py:463-510`. None exposes or calls the four probe names.
- Dynamic-dispatch check: production dispatch is explicit direct importing and branching at `src/api/main.py:222-300`. Registry handling is data lookup through `_load_registry` and `_get_model_entry` at `src/api/registry.py:25-31,113-117`, not function dispatch. No target-name hit was found through `getattr`, `importlib` or `__all__` scans. The only relevant `importlib` hits are migration-test machinery at `tests/test_namespace_migration.py:20,50`.
- Documentation check: `docs/api_contract.md:21-24` describes browsing as precomputed data with no model loaded, and `src/api/README.md:3-5,28` describes the same registry surface. Neither `docs/api_contract.md`, the root README, or API READMEs names these probes. The stale handover document still mentions `is_available` at `scripts/api_fixtures/handoff_report.md:1133-1144,1174`; that is documentation, not a code caller.
- Frontend check: tracked frontend code calls registry endpoints at `frontend/src/configure-screen.jsx:37` and `frontend/src/hooks/useClipList.js:26`, and calls `/api/library_predict` at `frontend/src/progress-screen.jsx:204-215`. No tracked JS/JSX/TS/TSX file names any of the four Python probes.

NOT CHECKED: No FastAPI server was started and no external consumer outside the tracked repository was assessed.

3. VERDICT: The downloader does import `scraper.config`, but not the rate or worker constants. Wiring those constants is a trivial local change. There is no observed import cycle, value mismatch or loss of CLI override.

- Current import: `src/scraper/download_scraped_videos.py:24` contains `from . import config`. It uses `config.CANDIDATES_CSV`, `config.VIDEOS_DIR` and `config.SOURCES_MANIFEST_NAME` at `src/scraper/download_scraped_videos.py:27-29`, `config.read_candidates` at `src/scraper/download_scraped_videos.py:491`, and CLI path defaults at `src/scraper/download_scraped_videos.py:550-552`.
- Hardcoded yt-dlp values at `src/scraper/download_scraped_videos.py:319-329` are:

  > `--retries`, `3`  
  > `--sleep-interval`, `5`  
  > `--max-sleep-interval`, `15`  
  > `--sleep-requests`, `10`  
  > `--limit-rate`, `2M`  
  > `--concurrent-fragments`, `1`

  They duplicate `YTDLP_RETRIES`, `SLEEP_INTERVAL_S`, `MAX_SLEEP_INTERVAL_S`, `SLEEP_REQUESTS_S`, `LIMIT_RATE` and `CONCURRENT_FRAGMENTS` at `src/scraper/config.py:181-188`. The binary name also duplicates `YTDLP_BIN = 'yt-dlp'` at `src/scraper/config.py:68`.
- Worker duplication: `_DEFAULT_WORKERS = 2` at `src/scraper/download_scraped_videos.py:40`, the function default at `src/scraper/download_scraped_videos.py:473-479`, and the CLI default at `src/scraper/download_scraped_videos.py:553` duplicate `DOWNLOAD_WORKERS = 2` at `src/scraper/config.py:185-186`.
- The existing config helper is not a complete replacement for the video command. It returns retries, request sleep and rate flags at `src/scraper/config.py:223-243`, while explicitly excluding the pre-download sleep flags at `src/scraper/config.py:227-230`. Directly reading the constants in the downloader is therefore the simplest wiring.
- No direct cycle is visible: `scraper.config` imports annotator-owned constants at `src/scraper/config.py:21`; `annotator.config` imports only local `fps_constants` and `types` at `src/annotator/config.py:16-17`.
- The CLI override remains intact. `download_all_videos` accepts `max_workers` at `src/scraper/download_scraped_videos.py:473-479`, the parser accepts `--workers` at `src/scraper/download_scraped_videos.py:553`, and `main` passes `args.workers` through at `src/scraper/download_scraped_videos.py:562-566`.
- Recency: `git log --follow` shows downloader commit `d3090d5` dated 2026-07-27 and config commit `7a7337a` dated 2026-07-28. The quoted values are current in those tracked files.

NOT CHECKED: No yt-dlp, network download or scraper runtime was executed.
