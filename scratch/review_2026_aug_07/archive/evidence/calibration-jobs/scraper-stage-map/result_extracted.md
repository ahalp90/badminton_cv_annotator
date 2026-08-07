I’ll inspect only the seven specified scraper modules, trace their literal artefact paths and control flow, and report line-numbered evidence. I’ll use read-only searches and file views; no files will be changed.
**Question**

Map each specified scraper module’s artefact flow, skip/resume behaviour, and per-item error handling.

**Scope inspected**

Read only:

- `src/scraper/config.py`
- `src/scraper/search_index.py`
- `src/scraper/relevance_triage.py`
- `src/scraper/transcript_acquisition.py`
- `src/scraper/commentary_cleaning.py`
- `src/scraper/commentary_pairing.py`
- `src/scraper/download_scraped_videos.py`

Commands: line-numbered file reads with `nl -ba`; scoped `rg -n` scans for paths, existence checks, loops, and exceptions. All completed with exit code 0. No files changed, no diff exists, and no tests or lint were run.

**Evidence**

`config.py`

- No CLI or stage entry point.
- Defines `SCRAPE_DIR`, `CANDIDATES_CSV`, `VIDEOS_DIR`, `SOURCES_MANIFEST_NAME`, `TRANSCRIPTS_DIR`, `CHUNKS_DIR`, and `PAIRS_CSV` at `config.py:21,30-35`. `MASKS_DIR` and `RALLY_SPANS_CSV` are imported from `annotator.config`, not defined here.
- `ensure_dirs()` creates `SCRAPE_DIR`, `VIDEOS_DIR`, `TRANSCRIPTS_DIR`, `CHUNKS_DIR`, and `MASKS_DIR` with `exist_ok=True` at `config.py:217-220`.
- `read_candidates()` reads `CANDIDATES_CSV` or a caller-supplied override at `config.py:247-257`. Missing input raises rather than skips: `config.py:254` `if not candidates_path.exists():`.
- `write_candidates()` overwrites `CANDIDATES_CSV` with the fixed header at `config.py:260-273`.
- No per-item loop, `try/except`, or final stage summary.

`search_index.py`

- CLI: `python -m scraper.search_index`, with no arguments (`search_index.py:13,269-277`).
- Reads no persistent artefact. It reads `SEARCH_TERMS`, `YTSEARCH_COUNT`, and the flat-print constants from `config.py`; yt-dlp supplies external metadata.
- Writes `CANDIDATES_CSV` through `write_candidates()` at `search_index.py:264-265`; path is `config.py:30`.
- No `.exists()` check and no resume. It skips enrichment when complete: `search_index.py:104` `if not needs_enrich:`. Malformed yt-dlp lines are skipped at `search_index.py:80-82`; duplicate IDs merge provenance at `search_index.py:219-227`.
- Exceptions:
  - Per-term yt-dlp timeout: `search_index.py:64-70`, catches `subprocess.TimeoutExpired`, prints to stdout, returns `[]`; the term loop continues.
  - Per-row enrichment timeout: `search_index.py:109-115`, same exception, prints and returns `True`; the row remains and processing continues.
  - Per-row enrichment non-zero exit: `search_index.py:116-118`, prints and returns `True`.
  - Per-row JSON parse failure: `search_index.py:119-123`, catches `json.JSONDecodeError`, prints a warning and returns `True`.
  - Invalid duration: `search_index.py:164-167`, catches `ValueError` and treats the duration as unflagged.
- Failed terms are visible through the `0 rows` count at `search_index.py:214` and the `terms_with_hits/total_terms` summary at `search_index.py:237`. If every term fails, the stage aborts before writing at `search_index.py:230-234`. Enrichment failures leave the candidate row in the final CSV but have no separate failure count.

`relevance_triage.py`

- CLI: `python -m scraper.relevance_triage`, with no arguments (`relevance_triage.py:9,291-299`).
- Reads `CANDIDATES_CSV` through `read_candidates()` at `relevance_triage.py:228-229`; constant `config.py:30`.
- Reads `TRANSCRIPTS_DIR/<video_id>.json` at `relevance_triage.py:53-56,238-239`; directory constant `config.py:33`.
- Writes `CHUNKS_DIR/<video_id>.json` at `relevance_triage.py:258-259`; directory constant `config.py:34`.
- Rewrites `CANDIDATES_CSV`, filling only successful `keep` decisions, at `relevance_triage.py:275-288`.
- Skip checks:
  - `relevance_triage.py:54` `if not path.exists():` returns `None` from `load_transcript()`.
  - `relevance_triage.py:238` `if not (TRANSCRIPTS_DIR / f'{video_id}.json').exists():` continues the video loop.
  - `relevance_triage.py:255-256` skips a `None` outcome.
  - No existing chunk-sidecar check; triage calls are repeated and the sidecar is overwritten.
- Exceptions:
  - Per-window LLM retry: `relevance_triage.py:160-170`, catches broad `Exception`, prints retry messages, retries, then raises `TriageError`.
  - Per-video wrapper: `relevance_triage.py:241-254`, catches `TriageError`, increments `failed`, appends to `retry_list`, prints the failure, then continues unless the circuit-break condition raises `RuntimeError` at `relevance_triage.py:249-253`.
  - Unexpected transcript/JSON errors are not caught and abort.
- Missing transcripts are absent from `keep_by_id`; `_write_keep_back()` leaves their `keep` field blank at `relevance_triage.py:284-287`, with no missing-count summary. Failed videos appear in the retry-list output at `relevance_triage.py:268-269`. If all calls fail, the stage raises before rewriting candidates at `relevance_triage.py:263-267`.

`transcript_acquisition.py`

- CLI: `python -m scraper.transcript_acquisition`, with no arguments (`transcript_acquisition.py:10,346-354`).
- Reads `CANDIDATES_CSV` through `read_candidates()` at `transcript_acquisition.py:290-291`; constant `config.py:30`.
- Uses a temporary directory containing yt-dlp caption files matching `<video_id>*.json3` or `<video_id>*.vtt` at `transcript_acquisition.py:66,88-94`; formats/languages come from `SUB_FORMAT` and `SUB_LANGS` (`config.py:123-124`). These files are parsed and then discarded.
- WhisperX fallback uses temporary `<video_id>*` audio files at `transcript_acquisition.py:170,189-190`.
- Writes `TRANSCRIPTS_DIR/<video_id>.json` at `transcript_acquisition.py:301,325`; constant `config.py:33`.
- Resume check: `transcript_acquisition.py:302` `if sidecar.exists():`; prints a skip and continues at `transcript_acquisition.py:303-304`.
- Exceptions:
  - Caption pull timeout: `transcript_acquisition.py:77-83`, catches `subprocess.TimeoutExpired`, prints and returns `None`.
  - Caption pull non-zero exit: `transcript_acquisition.py:84-86`, prints and returns `None`.
  - Audio pull timeout: `transcript_acquisition.py:179-185`, catches `subprocess.TimeoutExpired`, prints and returns `None`.
  - Audio pull non-zero exit: `transcript_acquisition.py:186-188`, prints and returns `None`.
  - WhisperX import failure: `transcript_acquisition.py:211-218`, catches `ImportError`, prints and returns `None`.
  - The WhisperX `try/finally` at `transcript_acquisition.py:232-239` catches nothing; model cleanup runs and other exceptions propagate.
  - `None` transcripts are recorded in `failures`, printed at `transcript_acquisition.py:312-314`, then either raise on the mid-batch threshold at `transcript_acquisition.py:317-323` or continue at `transcript_acquisition.py:324`. The final threshold is checked at `transcript_acquisition.py:336-343`.
- Existing sidecars are visible through the per-item skip and all-skipped summary at `transcript_acquisition.py:331-333`. Failed items have no sidecar but appear in the failure count at `transcript_acquisition.py:337`; a threshold abort prevents the final summary.

`commentary_cleaning.py`

- CLI: `python -m scraper.commentary_cleaning`; default runs clean and fine passes. Options are `--clean-only`, `--fine-only`, `--force`, and `--video-dir` (`commentary_cleaning.py:437-463`).
- Reads `CANDIDATES_CSV` through `read_candidates()` at `commentary_cleaning.py:185-187,402-404`; constant `config.py:30`.
- Reads and rewrites `CHUNKS_DIR/<video_id>.json` at `commentary_cleaning.py:196-201,225,269,414-424`; constant `config.py:34`.
- Fine pass reads `<video_id>.<ext>` from the caller-supplied `--video-dir`, accepting `.mp4`, `.mkv`, `.webm`, `.avi`, and `.mov` (`commentary_cleaning.py:46-47,388-393`). No config constant defines this argument.
- Temporary fine-pass artefacts are `<chunk_id>.wav` in a temporary directory, written by ffmpeg and read by WhisperX (`commentary_cleaning.py:298-310,362-369`).
- Skip checks:
  - Clean pass: `commentary_cleaning.py:197` `if not sidecar.exists():`; prints and continues at `commentary_cleaning.py:198-199`.
  - Blank raw chunk: `commentary_cleaning.py:209-210` continues.
  - Existing clean field: `commentary_cleaning.py:211-212` skips unless `force`.
  - Fine pass: `commentary_cleaning.py:415` `if not sidecar.exists():`; continues at `commentary_cleaning.py:416-417`.
  - Missing video: `commentary_cleaning.py:418-421` prints and continues.
  - No WhisperX/CUDA: `commentary_cleaning.py:406-409` skips the entire fine pass.
  - No aligned words: `commentary_cleaning.py:379-380` keeps coarse timestamps.
- Exceptions:
  - Per-chunk LLM retry: `commentary_cleaning.py:121-131`, catches broad `Exception`, prints retries, then raises `CleanError`.
  - Per-video clean wrapper: `commentary_cleaning.py:207-226`, catches `CleanError`, increments `failed`, prints, writes chunks cleaned before failure if any, then continues.
  - All-failure check raises `RuntimeError` at `commentary_cleaning.py:234-241`.
  - Fine model import failure: `commentary_cleaning.py:324-329`, catches `ImportError`, prints and returns `None`.
  - Fine-pass `try/finally` at `commentary_cleaning.py:411-434` catches nothing. BERTScore, ffmpeg, JSON, and WhisperX errors abort after cleanup.
- Clean failures are visible through `CLEAN FAILED` at `commentary_cleaning.py:219-225`; partial successful work can remain in the sidecar. The returned count only records successful IDs at `commentary_cleaning.py:230-232,270`. Missing fine inputs are printed at `commentary_cleaning.py:416-421`; there is no aggregate per-video fine failure summary. Existing clean chunks are only indirectly visible through `cleaned/to_clean` at `commentary_cleaning.py:232`.

`commentary_pairing.py`

- CLI: `python -m scraper.commentary_pairing` (`commentary_pairing.py:13,304-367`). All input/output paths have CLI overrides.
- Reads:
  - Rally spans from the file supplied by `RALLY_SPANS_CSV`, defaulted at `commentary_pairing.py:306`; existence failure raises at `commentary_pairing.py:208-216`. The constant is imported into `config.py` at `config.py:21`.
  - `CHUNKS_DIR/<video_id>.json`, default `CHUNKS_DIR` (`config.py:34`), at `commentary_pairing.py:219-225`.
  - `MASKS_DIR/<video_id>_replay.npy`, default `MASKS_DIR`, at `commentary_pairing.py:228-236`. `MASKS_DIR` is imported into `config.py:21`.
  - `SCRAPE_DIR/video_fps.csv`, via the local `VIDEO_FPS_CSV` at `commentary_pairing.py:43`, read at `commentary_pairing.py:198-203`.
  - `VIDEOS_DIR/<video_id>.<ext>`, default `VIDEOS_DIR` (`config.py:31`), when building fps and validating manifest entries. Accepted extensions are listed at `commentary_pairing.py:44`.
  - `VIDEOS_DIR/sources.toml`, using `SOURCES_MANIFEST_NAME` (`config.py:32`), at `commentary_pairing.py:337,239-256`.
- Writes:
  - `SCRAPE_DIR/video_fps.csv` at `commentary_pairing.py:57-85`.
  - `SCRAPE_DIR/rally_commentary_pairs.csv`, default `PAIRS_CSV` (`config.py:35`), at `commentary_pairing.py:359-364`.
- Skip/resume:
  - Non-video files are skipped while building fps at `commentary_pairing.py:71-73`.
  - Missing chunks: `commentary_pairing.py:222` `if not chunk_path.exists():`; returns `[]`, so rallies remain in output with blank commentary.
  - Missing replay mask: `commentary_pairing.py:231` `if not mask_path.exists():`; returns `None`, so pairing proceeds without masking.
  - Missing fps CSV, rally spans, or manifest raises rather than skips at `commentary_pairing.py:200-201,208-209,241-242`.
  - A video without fps is logged and skipped at `commentary_pairing.py:342-344`.
  - Manifest entries without matching IDs/files are continued over at `commentary_pairing.py:267-283`; required missing or duplicate entries then raise at `commentary_pairing.py:286-300`.
  - Replay-masked rallies are retained unpaired at `commentary_pairing.py:168-170`; claimed, out-of-window, and replay-start chunks are skipped at `commentary_pairing.py:176-185`.
  - No resume: fps and pair CSVs are rebuilt/overwritten.
- No `try/except` exists in this module. Validation, file, cv2, numpy, and pairing errors abort. Per-video missing-fps handling uses `log.warning` at `commentary_pairing.py:343`.
- Missing-fps videos are absent from `all_rows` and therefore the final CSV at `commentary_pairing.py:340-364`. Missing chunks, masks, ineligible videos, and masked rallies remain visible as blank/unpaired rally rows; per-video counts are logged at `commentary_pairing.py:351-357`.

`download_scraped_videos.py`

- CLI: `python -m scraper.download_scraped_videos`, with `--candidates-csv`, `--output-dir`, `--workers`, `--dataset`, `--allow-missing-audio`, and `--video-only` (`download_scraped_videos.py:542-593`).
- Reads:
  - `CANDIDATES_CSV` (`config.py:30`) through `config.read_candidates()` at `download_scraped_videos.py:468-489`.
  - Existing video files in `output_dir` matching exact `<video_id>.<ext>` or legacy `<video_id> <suffix>.<ext>`, excluding `.fNN` stems and accepting the local extension set (`download_scraped_videos.py:32,86-95`). Default directory is `VIDEOS_DIR` (`config.py:31`).
  - `output_dir/sources.toml`, using `SOURCES_MANIFEST_NAME` (`config.py:32`), at `download_scraped_videos.py:182-193,486-489`.
- Writes:
  - Downloaded/merged video files using `output_dir/<video_id>.%(ext)s` at `download_scraped_videos.py:338-345`.
  - `output_dir/sources.toml` atomically at `download_scraped_videos.py:237-257,526-532`.
  - Temporary manifest files named `.<manifest>.random.tmp` during the atomic write.
- Skip/resume:
  - Missing manifest is initialised, not skipped: `download_scraped_videos.py:185` `if not manifest_path.exists():`.
  - Non-kept candidates are excluded at `download_scraped_videos.py:457-458`.
  - Existing completed output triggers resume logic at `download_scraped_videos.py:298,305-336`: skip immediately when already ineligible, skip without probing in audio-bypass modes, otherwise verify audio.
  - No selected tasks prints and returns at `download_scraped_videos.py:492-495`.
  - `temporary_path.exists()` at `download_scraped_videos.py:256` is cleanup only.
- Exceptions:
  - ffprobe timeout is translated at `download_scraped_videos.py:105-120` from `subprocess.TimeoutExpired` to `_AudioProbeTimeout`.
  - New download timeout: `download_scraped_videos.py:339-361`, catches `subprocess.TimeoutExpired`, prints and returns a failed `DownloadOutcome`.
  - New download non-zero exit: `download_scraped_videos.py:362-364`, prints and returns failure.
  - New-file unreadable media or audio-probe timeout: `download_scraped_videos.py:386-395`, catches `_UnreadableMedia` or `_AudioProbeTimeout`, prints, deletes the file, and returns failure.
  - Existing-file probe errors: `download_scraped_videos.py:418-433`, catches the same two types, prints and returns a failed outcome while retaining the file and recording ineligibility.
  - Worker future wrapper: `download_scraped_videos.py:519-525`, catches broad `Exception`, records only the first unexpected exception, and continues collecting sibling outcomes. The manifest is written, then the first unexpected exception is re-raised at `download_scraped_videos.py:534-535`.
- Normal failures are visible in `DownloadOutcome.failed` at `download_scraped_videos.py:277-285`, the finished summary at `download_scraped_videos.py:537-539`, and the CLI failure summary/status at `download_scraped_videos.py:585-588`. Existing skips are printed at `download_scraped_videos.py:318,327,443` and count as accepted outcomes. Unexpected worker errors have no final failure count, but sibling manifest entries are still written before re-raising.

**Producer→consumer table**

| Artefact | Producing module | Consuming module(s) | Repeated filename/pattern locations |
|---|---|---|---|
| `SCRAPE_DIR/candidates.csv` (`CANDIDATES_CSV`) | `search_index` initially; `relevance_triage` rewrites `keep` | `relevance_triage`, `transcript_acquisition`, `commentary_cleaning`, `download_scraped_videos` | `config.py:30`; `search_index.py:1-7`; `relevance_triage.py:7,276-288` |
| `SCRAPE_DIR/transcripts/<video_id>.json` (`TRANSCRIPTS_DIR`) | `transcript_acquisition` | `relevance_triage` | `config.py:33`; `transcript_acquisition.py:301`; `relevance_triage.py:53,238` |
| `SCRAPE_DIR/chunks/<video_id>.json` (`CHUNKS_DIR`) | `relevance_triage`; rewritten by `commentary_cleaning` | `commentary_cleaning`, `commentary_pairing` | `config.py:34`; `relevance_triage.py:258`; `commentary_cleaning.py:196,414`; `commentary_pairing.py:221` |
| `VIDEOS_DIR/<video_id>.<ext>` | `download_scraped_videos` | `download_scraped_videos` resume, `commentary_cleaning` fine pass, `commentary_pairing` fps/manifest checks | `config.py:31`; `download_scraped_videos.py:86-95,338`; `commentary_cleaning.py:390`; `commentary_pairing.py:60-71` |
| `VIDEOS_DIR/sources.toml` | `download_scraped_videos` | `download_scraped_videos` resume; `commentary_pairing` | `config.py:32`; `download_scraped_videos.py:486`; `commentary_pairing.py:337` |
| `SCRAPE_DIR/video_fps.csv` (`VIDEO_FPS_CSV`, local to pairing) | `commentary_pairing.build_video_fps_csv` | `commentary_pairing` main | `commentary_pairing.py:43,57-64,200-203,309` |
| `SCRAPE_DIR/rally_commentary_pairs.csv` (`PAIRS_CSV`) | `commentary_pairing` | None in scope | `config.py:35`; `commentary_pairing.py:310,359-364` |
| `RALLY_SPANS_CSV` (`rally_spans.csv` in module documentation) | Outside scope | `commentary_pairing` | `commentary_pairing.py:8,208-215`; path value is imported by `config.py:21` |
| `MASKS_DIR/<video_id>_replay.npy` | Outside scope | `commentary_pairing` | `commentary_pairing.py:229-236`; `MASKS_DIR` is imported by `config.py:21` |

**Counterevidence**

- `candidates.csv` and chunk sidecars have dual writers.
- `sources.toml` is both read and written by the downloader.
- `video_fps.csv` is both produced and consumed by `commentary_pairing`.
- Missing chunks are a skip in cleaning and triage, but pairing retains the rally with blank commentary.
- Temporary caption, audio, wav, and manifest files are intra-stage artefacts, not persistent pipeline contracts.
- CLI path overrides can move every default artefact. The downloader also accepts legacy spaced video filenames.

**Unresolved/dynamic surfaces**

- `SCRAPE_DIR`, `MASKS_DIR`, and `RALLY_SPANS_CSV` values are outside scope because `config.py` imports them from `annotator.config` (`config.py:21`).
- `CONTACT_FRAMES_CSV` is imported by `config.py` but is not read or written by any inspected module.
- Rally-span and replay-mask producers are outside the inspected files.
- `commentary_cleaning` receives its fine-pass video directory from `--video-dir`; it has no config-defined default.
