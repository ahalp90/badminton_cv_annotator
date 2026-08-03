# D1 raw return (automated read-only check, 2026-08-02): can the scraper downloader replace the bst_x downloader?

1. VERDICT: viable with adapter

Viable with adapter, not as-is. BST expects spaced filenames and an `id,width,height` sidecar, while scraper consumes kept candidate rows, emits `<video_id>.<ext>`, and normally requires audio. `src/bst_x/pipeline/build_dataset.py:183-208`, `src/bst_x/pipeline/clip_generator.py:177-203`, `src/scraper/download_scraped_videos.py:315-385`

2. BST_X REQUIREMENTS

- Input is `set/match.csv`, with `id`, `url`, and `video` columns by default. Optional filtering uses a `keep` column containing the exact string `True`. `src/bst_x/pipeline/download_videos.py:105-151`

- The live pipeline uses the default `EXCLUDED_VIDEOS` filter before creating download tasks. `src/bst_x/pipeline/build_dataset.py:183-185`, `src/bst_x/pipeline/download_videos.py:135-144`, `src/bst_x/pipeline/config.py:266-293`

- Successful files are named `{id} {video_name}.{extension}`. Existing files matching that pattern are skipped. Clip generation searches for `{video_id} *`, so the post-ID filename is part of the contract. `src/bst_x/pipeline/download_videos.py:41-64`, `src/bst_x/pipeline/clip_generator.py:177-193`

- yt-dlp gets three retries, the throttle flags, and a 1,800-second timeout. Failed or timed-out workers return `None`; the batch returns only successful filenames and does not resubmit failed rows. `src/bst_x/pipeline/download_videos.py:66-102`, `src/bst_x/pipeline/download_videos.py:156-165`

- `build_resolution_csv` writes `id,width,height`, extracts the ID from the token before the first space, and writes a missing-ID report against `match.csv`. It scans independently of `download_all_videos`; its CLI can run with `--skip-download`. With no supported video files, it returns an empty frame without writing the CSV. `src/bst_x/pipeline/download_videos.py:168-238`, `src/bst_x/pipeline/download_videos.py:244-263`

- The resolution CSV is read by shuttle normalisation and by BST pose/court preparation. Court projection scales using the indexed `width` and `height` values. `src/bst_x/pipeline/shuttle_extractor.py:201-218`, `src/bst_x/pipeline/shuttle_extractor.py:234-272`, `src/bst_x/preparing_data/prepare_train_on_shuttleset.py:989-1004`, `src/bst_x/pipeline/court_utils.py:102-122`

3. SCRAPER CONTRACT

- The batch input is `candidates.csv`, not TOML. The fixed candidate schema includes `video_id`, `url`, `title`, metadata fields, `keep`, and `triage_verdict`; downloading actually uses `video_id`, `url`, `title`, and exact `keep == 'True'`. Duplicate kept IDs raise an error. `src/scraper/config.py:38-60`, `src/scraper/config.py:247-257`, `src/scraper/download_scraped_videos.py:458-486`

- `sources.toml` is state, not the selection input. It must contain `dataset = "scraped"` and a `[videos]` table. Entries are scalar-valued tables with optional string/integer `video_id`, string `title`/`url`, and boolean `commentary_eligible`. A missing manifest is initialised automatically. `src/scraper/download_scraped_videos.py:126-167`

- Outputs use `<video_id>.<ext>`, with `--merge-output-format mp4`; completed-output detection requires the file stem to equal `video_id`. Multiple matching files raise an error. `src/scraper/download_scraped_videos.py:31-36`, `src/scraper/download_scraped_videos.py:75-83`, `src/scraper/download_scraped_videos.py:271-315`

- The format selector requests H.264 video plus audio, or an H.264 premuxed file. It has no video-only alternative. Default mode requires `ffprobe` and deletes newly downloaded files that are unreadable, time out, or contain no audio. `src/scraper/download_scraped_videos.py:33-40`, `src/scraper/download_scraped_videos.py:91-115`, `src/scraper/download_scraped_videos.py:363-385`, `src/scraper/download_scraped_videos.py:499-501`

- `allow_missing_audio=True` skips probing and marks new files `commentary_eligible = false`; it does not itself add a video-only yt-dlp format. `src/scraper/download_scraped_videos.py:354-360`, `src/scraper/download_scraped_videos.py:473-486`

- The batch atomically rewrites `sources.toml`, retaining old entries and adding/updating `video_id`, `title`, `url`, and `commentary_eligible`. The no-task path still writes the manifest. `src/scraper/download_scraped_videos.py:211-248`, `src/scraper/download_scraped_videos.py:488-497`, `src/scraper/download_scraped_videos.py:527-540`

- Scraper stage 11 requires the manifest basename to match an existing file and uses `commentary_eligible` to leave commentary blank for ineligible videos. Exact scraper naming should therefore be preserved if the manifest is retained. `src/scraper/stage11_pairing.py:256-297`, `src/scraper/stage11_pairing.py:334-350`

4. RETARGET PLAN

1. Add a `download_shuttleset_videos(...)` adapter. Read `match.csv`, map `id -> video_id`, `url -> url`, and `video -> title`, set `keep = "True"`, filter `EXCLUDED_VIDEOS`, and write a temporary or dedicated CSV with the scraper’s fixed header. Call the scraper batch with `output_dir=RAW_VIDEO_DIR`. `src/bst_x/pipeline/download_videos.py:105-151`, `src/scraper/config.py:46-60`, `src/scraper/download_scraped_videos.py:458-492`

2. Change `build_dataset.py` to import the adapter instead of the old downloader. Keep `build_resolution_csv`, either in its current module or moved to a metadata-only module, because it has live callers and a distinct downstream contract. `src/bst_x/pipeline/build_dataset.py:23-36`, `src/bst_x/pipeline/build_dataset.py:183-197`, `src/bst_x/pipeline/download_videos.py:168-238`

3. Add an explicit scraper video-only mode. The mode should append the BST H.264 video-only selector and use the scraper’s `allow_missing_audio` path, recording `commentary_eligible = false`. Passing `allow_missing_audio=True` without adding that selector will still fail URLs for which only video-only H.264 is available. `src/bst_x/pipeline/download_videos.py:23-30`, `src/scraper/download_scraped_videos.py:33-36`, `src/scraper/download_scraped_videos.py:354-377`

4. Preserve scraper filenames and update BST consumers to accept numeric `<id>.mp4` stems: change the clip lookup from `{id} *` to an exact numeric stem, and make `build_resolution_csv` parse both the old spaced form and the scraper form. Renaming scraper files instead would also require manifest-key updates because scraper resume and stage 11 use exact basenames. `src/bst_x/pipeline/clip_generator.py:177-193`, `src/bst_x/pipeline/download_videos.py:192-205`, `src/scraper/download_scraped_videos.py:75-83`, `src/scraper/stage11_pairing.py:271-297`

5. Tighten the raw-video guard to test supported video extensions or inspect `DownloadOutcome.failed`. The current `glob('*.*')` check can count scraper’s `sources.toml` as a “video”, while the scraper writes that manifest even when there are no tasks. `src/bst_x/pipeline/build_dataset.py:189-197`, `src/scraper/download_scraped_videos.py:488-497`, `src/bst_x/pipeline/download_videos.py:181-188`

6. Retain these live BST-only contracts during removal of the old downloader: the width/height resolution builder and its CLI, the H.264 video-only selector, the `EXCLUDED_VIDEOS` filter, and the spaced-name lookup used by clip generation. The scraper-side metadata builder produces `video_id,fps`, so it cannot replace the BST resolution CSV. `src/bst_x/pipeline/download_videos.py:23-30`, `src/bst_x/pipeline/download_videos.py:168-238`, `src/bst_x/pipeline/download_videos.py:258-263`, `src/bst_x/pipeline/clip_generator.py:177-193`, `src/scraper/stage11_pairing.py:57-85`

5. NOT CHECKED

- No yt-dlp, ffprobe, OpenCV, lint, type checks, or tests were run.
- I did not verify which actual ShuttleSet URLs have or lack audio.
- `data/`, experiments, `scripts/archive/`, and `docs/**/*.py` were not inspected.
- Semantic symbol and file navigation was used. Caller references were cross-checked with text search because Python call hierarchy did not resolve these module-level FQNs.
