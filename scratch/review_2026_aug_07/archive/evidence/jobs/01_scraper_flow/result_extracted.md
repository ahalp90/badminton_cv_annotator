## 1. Question investigated

Traced the maintained scraper flow on `main`:

`search/index -> candidates.csv -> transcripts/<video_id>.json -> chunks/<video_id>.json -> videos/<video_id>.<ext> + sources.toml -> cleaned chunks -> rally_commentary_pairs.csv`.

The stages are separate module CLIs. No scraper orchestrator was found. `src/scraper/__init__.py:1-7`.

## 2. Files/symbols inspected

- `src/scraper/config.py`: constants and `check_ytdlp`, `ensure_dirs`, `ytdlp_throttle_args`, `read_candidates`, `write_candidates`.
- `src/scraper/search_index.py`: public functions at `:47-266`; CLI at `:269-277`.
- `src/scraper/transcript_acquisition.py`: public functions at `:56-342`; CLI at `:346-354`.
- `src/scraper/relevance_triage.py`: public functions at `:47-287`; CLI at `:291-300`.
- `src/scraper/download_scraped_videos.py`: `DownloadOutcome`, `download_all_videos`, CLI at `:54-592`.
- `src/scraper/commentary_cleaning.py`: `CleanError`, `run_clean`, `load_fine_models`, `refine_timestamps`, `run_fine`, CLI at `:71-467`.
- `src/scraper/commentary_pairing.py`: `build_video_fps_csv`, `pair_video`, CLI at `:57-368`.
- Supporting definitions in `src/annotator/config.py:25-28` and `src/bst_x/pipeline/download_adapter.py:8-61`.
- Serena used: `get_symbols_overview`, `find_symbol`, `find_referencing_symbols`, `search_for_pattern`, and `find_declaration`.

## 3. Concrete evidence

### Shared paths and schemas

- `SCRAPE_DIR`, `MASKS_DIR`, `RALLY_SPANS_CSV`, and `CONTACT_FRAMES_CSV` are imported into scraper config from annotator config at `src/scraper/config.py:21`; their definitions are `src/annotator/config.py:25-28`.
- Scraper paths are `candidates.csv`, `videos/`, `sources.toml`, `transcripts/`, `chunks/`, and `rally_commentary_pairs.csv` at `src/scraper/config.py:30-35`.
- `candidates.csv` has 13 columns defined at `src/scraper/config.py:46-60`. `write_candidates` always writes that header at `src/scraper/config.py:260-273`.
- Transcript sidecars contain `{source, segments}`. Segment fields are `start`, `end`, and `text`; the writer is at `src/scraper/transcript_acquisition.py:250-277, 301-328`.
- Chunk sidecars contain a JSON list of `{chunk_id, start, end, text}` before cleaning; triage writes them at `src/scraper/relevance_triage.py:205-215, 257-260`.
- The downloader manifest contains a scalar `dataset` and a `videos` table. Entries include `video_id`, `title`, `url`, and `commentary_eligible`; validation is at `src/scraper/download_scraped_videos.py:147-179`.
- `video_fps.csv` has `video_id,fps` columns at `src/scraper/commentary_pairing.py:57-85`.
- Rally input rows require `video_id`, `rally_id`, `start_frame`, and `end_frame` at `src/scraper/commentary_pairing.py:206-216`.
- Replay masks are `<video_id>_replay.npy`, one-dimensional boolean arrays, at `src/scraper/commentary_pairing.py:228-236`.
- Pair output columns are defined at `src/scraper/commentary_pairing.py:46-51` and written at `:359-364`.

### Search/index stage

- Module and public symbols: `search_term_rows`, `enrich_row`, `flag_doubles`, `duration_out_of_band`, `upload_before_floor`, `build_candidates`, `main`; definitions at `src/scraper/search_index.py:47-277`.
- CLI: `python -m scraper.search_index`; argparse is at `:269-273`, module guard at `:276-277`.
- Reads no maintained file artefact. It invokes `yt-dlp` search and metadata subprocesses at `src/scraper/search_index.py:57-129`.
- Writes `SCRAPE_DIR/candidates.csv` through `write_candidates` at `src/scraper/search_index.py:239-266`.
- It also creates `SCRAPE_DIR`, `videos`, `transcripts`, `chunks`, and `masks` through `ensure_dirs` at `src/scraper/search_index.py:202-203` and `src/scraper/config.py:217-220`.
- Timeout, non-zero search, and malformed `--print` rows are logged and skipped at `src/scraper/search_index.py:64-90`.
- Invalid enrichment JSON, timeout, and non-zero metadata calls leave fields unchanged and continue at `src/scraper/search_index.py:109-129`.
- If every search term returns zero rows, `build_candidates` raises `RuntimeError` at `src/scraper/search_index.py:230-234`.

### Transcript stage

- Module and public symbols: `pull_subtitles`, `parse_json3`, `parse_vtt`, `whisperx_fallback`, `acquire_transcript`, `run_transcript_acquisition`, `main`; definitions at `src/scraper/transcript_acquisition.py:56-354`.
- CLI: `python -m scraper.transcript_acquisition`; argparse and module guard are at `src/scraper/transcript_acquisition.py:346-354`.
- Reads `candidates.csv` through `read_candidates` at `src/scraper/transcript_acquisition.py:280-301`.
- Temporary caption files use `<video_id>*.json3` or `<video_id>*.vtt`; selection is at `src/scraper/transcript_acquisition.py:66-94`.
- Temporary Whisper audio uses `<video_id>*` in a `TemporaryDirectory` at `src/scraper/transcript_acquisition.py:156-190`.
- Writes `transcripts/<video_id>.json` at `src/scraper/transcript_acquisition.py:301-328`.
- Existing sidecars are skipped without parsing at `src/scraper/transcript_acquisition.py:301-304`.
- Missing candidates raises `FileNotFoundError` through config at `src/scraper/config.py:247-257`; an empty candidate list raises `RuntimeError` at `src/scraper/transcript_acquisition.py:290-295`.
- Subtitle timeout, non-zero exit, or no caption file returns `None` at `src/scraper/transcript_acquisition.py:77-94`.
- Empty parsed captions and unavailable WhisperX/CUDA return `None` at `src/scraper/transcript_acquisition.py:211-247, 262-277`.
- Per-video failures are counted and skipped; more than 50% failures raises mid-run or at completion at `src/scraper/transcript_acquisition.py:311-343`.
- Corrupt JSON caption content is not caught by `run_transcript_acquisition`; `json.loads` is direct at `src/scraper/transcript_acquisition.py:106`.

### Relevance/chunk stage

- Module and public symbols: `TriageError`, `load_transcript`, `chunk_windows`, `build_triage_prompt`, `call_triage_llm`, `triage_video`, `run_relevance_triage`, `main`; definitions at `src/scraper/relevance_triage.py:43-300`.
- CLI: `python -m scraper.relevance_triage`; argparse and module guard are at `src/scraper/relevance_triage.py:291-300`.
- Reads `candidates.csv` and checks `transcripts/<video_id>.json` at `src/scraper/relevance_triage.py:218-239`.
- Reads each transcript with `json.loads` at `src/scraper/relevance_triage.py:47-56`.
- Writes `chunks/<video_id>.json` at `src/scraper/relevance_triage.py:255-261`.
- Rewrites `candidates.csv`, filling only `keep` for triaged rows, at `src/scraper/relevance_triage.py:275-288`.
- Missing transcript sidecars are skipped. Empty transcript segments produce an empty chunk list because `chunk_windows` returns `[]` at `src/scraper/relevance_triage.py:59-86`.
- Missing API key, SDK errors, and malformed LLM JSON are retried, then become `TriageError` at `src/scraper/relevance_triage.py:117-170`.
- `TriageError` is logged and skipped per video; all-failure conditions raise at `src/scraper/relevance_triage.py:241-267`.
- Corrupt transcript JSON, missing `segments`, malformed LLM item fields, and invalid duration conversion are not caught by the batch handler.

### Video/download and manifest stage

- Module and public symbols: `DownloadOutcome`, `download_all_videos`, `main`; definitions at `src/scraper/download_scraped_videos.py:54-72, 467-592`.
- CLI options are `--candidates-csv`, `--output-dir`, `--workers`, `--dataset`, `--allow-missing-audio`, and `--video-only` at `src/scraper/download_scraped_videos.py:547-572`.
- Reads the candidate CSV at `src/scraper/download_scraped_videos.py:486-490`.
- Reads or initialises `videos/sources.toml` at `src/scraper/download_scraped_videos.py:182-193`.
- Existing video files are scanned by `iterdir`, accepted extensions, exact stem, or `<video_id> ` stem at `src/scraper/download_scraped_videos.py:86-95`.
- Downloads use `<video_id>.%(ext)s` and accepted extensions are `.mp4`, `.mkv`, `.webm`, `.avi`, and `.mov` at `src/scraper/download_scraped_videos.py:32, 338-357`.
- Writes `sources.toml` atomically through a random `.sources.toml.*.tmp` file at `src/scraper/download_scraped_videos.py:237-257`.
- Non-zero downloads, timeouts, absent outputs, and failed audio checks become per-video failure outcomes at `src/scraper/download_scraped_videos.py:359-400`.
- Multiple matching outputs and unexpected worker exceptions are raised after sibling outcomes are written at `src/scraper/download_scraped_videos.py:366-371, 519-535`.
- Invalid or corrupt manifests raise during validation at `src/scraper/download_scraped_videos.py:147-193`.
- Empty selection writes the manifest and returns `[]` at `src/scraper/download_scraped_videos.py:492-495`.

### Commentary cleaning and fine timestamps

- Module and public symbols: `CleanError`, `call_clean_llm`, `run_clean`, `load_fine_models`, `refine_timestamps`, `run_fine`, `main`; definitions at `src/scraper/commentary_cleaning.py:71-467`.
- CLI options are `--clean-only`, `--fine-only`, `--force`, and `--video-dir` at `src/scraper/commentary_cleaning.py:437-463`.
- Clean pass reads `candidates.csv` and kept rows with `keep == 'True'` at `src/scraper/commentary_cleaning.py:173-201`.
- It reads and rewrites `chunks/<video_id>.json`, adding `text_clean`, `alt_phrasings`, `bert_f1`, and `clean_pass` at `src/scraper/commentary_cleaning.py:201-270`.
- Missing chunk sidecars are logged and skipped. Missing raw `text` raises `KeyError` at `src/scraper/commentary_cleaning.py:194-204`.
- LLM failures retry and become `CleanError`; the video is skipped, with partial cleaned chunks written when applicable, at `src/scraper/commentary_cleaning.py:207-226`.
- Fine pass searches `video_dir` with `<video_id>.*` and the video extension set at `src/scraper/commentary_cleaning.py:388-393`.
- Fine pass reads chunk sidecars and video media, writes temporary WAV spans, and rewrites chunk timestamps at `src/scraper/commentary_cleaning.py:285-310, 362-385, 411-425`.
- Missing WhisperX/CUDA skips the whole fine pass; missing sidecars/videos skip individual videos at `src/scraper/commentary_cleaning.py:313-332, 406-421`.
- `ffmpeg`, WhisperX, malformed JSON, and alignment errors propagate; cleanup runs in `finally` at `src/scraper/commentary_cleaning.py:426-434`.

### Rally/commentary pairing

- Module and public symbols: `build_video_fps_csv`, `pair_video`, `main`; definitions at `src/scraper/commentary_pairing.py:57-367`.
- CLI options are `--rally-spans`, `--chunks-dir`, `--masks-dir`, `--fps-csv`, `--pairs-csv`, `--video-dir`, and `--build-fps-from` at `src/scraper/commentary_pairing.py:304-314`.
- `--build-fps-from` reads downloaded video files and writes `video_fps.csv` before pairing at `src/scraper/commentary_pairing.py:67-85, 329-330`.
- Pairing reads FPS, rally spans, and, when eligible, `chunks/<video_id>.json` and `masks/<video_id>_replay.npy` at `src/scraper/commentary_pairing.py:332-354`.
- `sources.toml` is required when at least one rally video has FPS and is used to determine `commentary_eligible` at `src/scraper/commentary_pairing.py:334-338`.
- Missing FPS or chunk/mask sidecars cause per-file skips or empty inputs; missing rally spans/FPS/manifest raises at `src/scraper/commentary_pairing.py:198-256`.
- Invalid mask shape/type raises `ValueError` at `src/scraper/commentary_pairing.py:228-236`.
- The final CSV is always written, including when `all_rows` is empty, at `src/scraper/commentary_pairing.py:340-364`.

### Config consumers

- Path constants: `CANDIDATES_CSV` is used by search, downloader defaults, and config read/write; `TRANSCRIPTS_DIR` by transcript and triage; `CHUNKS_DIR` by triage, cleaning, and pairing; `VIDEOS_DIR` by downloader and pairing; `SOURCES_MANIFEST_NAME` by downloader and pairing; `PAIRS_CSV` by pairing. Definitions: `src/scraper/config.py:30-35`.
- Candidate schema: `CANDIDATES_COLUMNS` is used by `write_candidates` and the ShuttleSet adapter at `src/scraper/config.py:46-60, 260-273` and `src/bst_x/pipeline/download_adapter.py:30-55`.
- Search constants are consumed by `search_index.py:25-43, 57-187, 190-265`; definitions are `src/scraper/config.py:62-116`.
- Transcript constants are consumed by `transcript_acquisition.py:26-41, 67-181, 227-343`; definitions are `src/scraper/config.py:123-128, 180-199`.
- Triage constants are consumed by `relevance_triage.py:22-40, 59-267`; definitions are `src/scraper/config.py:133-159, 201-203`.
- Cleaning constants are consumed by `commentary_cleaning.py:33-44, 58-269, 313-369`; definitions are `src/scraper/config.py:162-170`.
- Pairing constants are consumed by `commentary_pairing.py:28-37, 119-174, 304-338`; definitions are `src/scraper/config.py:172-174`.
- Download/throttle constants are consumed by search, transcript, downloader, and `ytdlp_throttle_args`; definitions are `src/scraper/config.py:179-192, 223-244`.
- `SCRAPE_TRACKNET_STRIDE` and `SCRAPE_TRACKNET_LARGE_VIDEO` are consumed outside `src/scraper` by `src/bst_x/pipeline/shuttle_extractor.py:29, 40`.
- `CONTACT_FRAMES_CSV` is imported into scraper config but no consumer was found under `src/scraper`.

## 4. Callers/consumers found

- `search_index.build_candidates` is called by its own CLI only: `src/scraper/search_index.py:269-277`.
- `run_transcript_acquisition` is called by its own CLI only: `src/scraper/transcript_acquisition.py:346-354`.
- `run_relevance_triage` is called by its own CLI only: `src/scraper/relevance_triage.py:291-300`.
- `download_all_videos` is called by its CLI and by the ShuttleSet adapter: `src/scraper/download_scraped_videos.py:574-581`, `src/bst_x/pipeline/download_adapter.py:8-9, 55-61`.
- `run_clean` and `run_fine` are called by the cleaning CLI at `src/scraper/commentary_cleaning.py:455-463`.
- `build_video_fps_csv` and `pair_video` are called by the pairing CLI at `src/scraper/commentary_pairing.py:329-354`.
- Artefact edges are: search `-> candidates.csv`; candidates `-> transcript acquisition`, triage, downloader, cleaning; transcripts `-> triage`; chunks `-> cleaning`, pairing; videos and manifest `-> cleaning`, pairing; FPS CSV `-> pairing`; rally spans and replay masks `-> pairing`; pairs CSV is the final writer.

## 5. Counterevidence / surprises

- `pyproject.toml:82-84` defines a `scraper` optional dependency, not a console script. No `[project.scripts]` or console-entry declaration was found.
- `rally_commentary_pairs.csv` is written by pairing, but no maintained reader was found under `src/` or `scripts/`.
- `video_fps.csv` has an in-repo producer only when pairing receives `--build-fps-from`; otherwise pairing reads a pre-existing file.
- `rally_spans.csv` and replay masks are produced outside `src/scraper`; scraper pairing only reads them.
- The clean pass does not read `sources.toml`; manifest eligibility is applied by pairing at `src/scraper/commentary_pairing.py:345-353`.
- `ensure_dirs` creates `masks/` even when called by search, transcript acquisition, or relevance triage at `src/scraper/config.py:217-220`.

## 6. Unresolved or dynamic surfaces

- `BADMINTON_SCRAPE_DIR` can change all default paths at import time: `src/annotator/config.py:22-28`.
- yt-dlp stdout, downloaded media, ffprobe output, Google SDK responses, WhisperX output, and BERTScore results are external runtime surfaces.
- JSON sidecars are parsed with `json.loads`, but the maintained readers do not perform complete schema validation.
- Tests monkeypatch imported paths, subprocesses, private helpers, and config values, for example `tests/test_scraper_search_index.py:55-62`, `tests/test_scraper_transcript_acquisition.py:115-119`, `tests/test_scraper_relevance_triage.py:142-167`, and `tests/test_scraper_commentary_cleaning.py:47-57`.
- Module-level mutable values include `CANDIDATES_COLUMNS`, `FLAT_PRINT_FIELDS`, `SEARCH_TERMS`, and keyword lists at `src/scraper/config.py:46-60, 74-116`; extension sets and output-column lists at `src/scraper/download_scraped_videos.py:32` and `src/scraper/commentary_pairing.py:44-51`; `_VIDEO_EXTS` at `src/scraper/commentary_cleaning.py:46-47`. No runtime cache object was found.
- No files were changed. `git diff --quiet -- src/scraper pyproject.toml src/annotator/config.py src/bst_x/pipeline/download_adapter.py` exited `0`; diff location: none.
- Read-only census commands included `git status`, `nl`, `rg`, and `wc`; no tests or lint were run. The launcher captures this response; no `result.md` was written.