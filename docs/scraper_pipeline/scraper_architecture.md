# Scraper architecture

## Scope and current status

`src/scraper` contains the acquisition and commentary side of the dataset
builder. It searches for badminton video, acquires transcripts, asks a language
model whether the commentary is useful, downloads selected videos, cleans the
commentary and joins commentary chunks to rallies produced by `src/annotator`.

The modules are usable but do not yet form one in-process orchestrator. Each
stage is a separate `python -m scraper.<module>` command. GitHub issue
[#15](https://github.com/ahalp90/badminton_cv_annotator/issues/15) tracks the
end-to-end wiring and trial.

The current code is authoritative. This document replaces the private
`local_scratch/autograder_architecture/scraper_spec.md` draft cited by older
source comments.

## Data flow

The pipeline uses files under one scrape root as explicit stage boundaries:

1. `search_index` queries yt-dlp searches and writes `candidates.csv`.
2. `transcript_acquisition` writes one transcript JSON sidecar per video.
3. `relevance_triage` writes commentary chunks and fills the candidate `keep`
   and `triage_verdict` fields.
4. `download_scraped_videos` downloads kept videos and records provenance and
   commentary eligibility in `videos/sources.toml`.
5. `commentary_cleaning` cleans kept chunks and can refine their timestamps
   against downloaded audio with WhisperX.
6. The annotator writes rally spans and replay masks.
7. `commentary_pairing` joins the first eligible following commentary chunk to
   each rally and writes `rally_commentary_pairs.csv`.

No scraper stage invokes the next stage. A completed output file is the rerun
checkpoint for the next command.

## Module and command map

Run commands from the repository root with `src` on `PYTHONPATH`:

```bash
PYTHONPATH=src python -m scraper.search_index
PYTHONPATH=src python -m scraper.transcript_acquisition
PYTHONPATH=src python -m scraper.relevance_triage
PYTHONPATH=src python -m scraper.download_scraped_videos
PYTHONPATH=src python -m scraper.commentary_cleaning
PYTHONPATH=src python -m scraper.commentary_pairing
```

`scraper.config` owns the candidate schema, search and download settings,
commentary thresholds and scraper-side paths. `annotator.config` owns the
shared scrape root and the annotator outputs consumed during pairing.

`BADMINTON_SCRAPE_DIR` overrides the default root at
`data/scrape_output/`. `scraper.config.ensure_dirs()` creates the standard
subdirectories.

## File contracts

### `candidates.csv`

`search_index` writes these fixed columns:

```text
video_id,url,title,channel,duration_s,upload_date,search_term,substream,
doubles_suspect,duration_suspect,upload_date_suspect,keep,triage_verdict
```

Boolean cells use the literal strings `True` and `False`. Consumers compare the
strings explicitly. Search-time screens record suspicions without dropping the
row. `relevance_triage` later fills `keep` and `triage_verdict`.

### Transcript and chunk sidecars

`transcripts/<video_id>.json` has this shape:

```json
{
  "source": "youtube_asr",
  "segments": [{"start": 12.3, "end": 15.9, "text": "..."}]
}
```

`source` is `youtube_asr` or `whisper`. The implementation does not distinguish
human YouTube subtitles from automatic subtitles.

`chunks/<video_id>.json` is a list of commentary chunks. Triage creates
`chunk_id`, `start`, `end` and `text`. Cleaning adds `text_clean`, three
`alt_phrasings`, `bert_f1` and `clean_pass`.

### Downloaded videos and provenance

Downloaded media lives at `videos/<video_id>.<ext>`. New downloads select H.264
and merge to MP4. `videos/sources.toml` contains a top-level dataset name and a
`[videos]` table. Each video entry records its ID, title, URL and boolean
`commentary_eligible` value.

Default downloads require a readable audio stream. `--allow-missing-audio` and
`--video-only` retain video without that guarantee and record it as commentary
ineligible.

### Annotator hand-off

`commentary_pairing` consumes:

- `rally_spans.csv`, with `video_id`, integer `rally_id`, `start_frame` and
  `end_frame`;
- `video_fps.csv`, with `video_id,fps`;
- optional one-dimensional boolean `masks/<video_id>_replay.npy` arrays; and
- `videos/sources.toml` eligibility entries.

It writes `rally_commentary_pairs.csv` with:

```text
video_id,rally_id,rally_start,rally_end,chunk_id,commentary_start,commentary_end
```

Rally positions remain source frames. Commentary positions remain seconds.
The per-video FPS is the explicit conversion between those clocks.

## Commentary acquisition, cleaning and pairing

Transcript acquisition tries English YouTube subtitles first. It prefers JSON3
and falls back to VTT. If neither yields a transcript, it can download audio and
run the GPU-only WhisperX `large-v3-turbo` path.

Relevance triage divides transcripts into overlapping 600-second windows with
a 60-second overlap. The configured JSON language-model prompt extracts useful
qualitative, tactical or shot-quality commentary. A video is kept when it
meets any configured chunk-count or chunk-density rule.

Cleaning makes one language-model call per chunk. Each call returns cleaned
text and three alternative phrasings. BERTScore compares cleaned and raw text;
`clean_pass` records whether the result reaches `CLEAN_BERTSCORE_MIN`.

The optional fine-timestamp pass uses WhisperX against the downloaded audio.
Without WhisperX or CUDA, coarse timestamps remain unchanged.

Pairing sorts rallies and chunks by time. Each commentary chunk can be claimed
once. A rally receives the first unclaimed chunk whose start is after the rally
end and no more than `PAIR_WINDOW_S` seconds later. Replay evidence can hold
out a rally or a chunk after short replay runs have been filtered with an
FPS-scaled minimum duration. Commentary-ineligible video still produces rally
rows, with blank commentary fields.

## Failure and rerun behaviour

The pipeline fails loudly when a batch result is no longer trustworthy, while
keeping successful per-video outputs that can be resumed.

- Search logs individual query failures and continues. It fails if every search
  term returns no rows.
- Transcript acquisition skips existing sidecars. Failed videos have no
  sidecar and are retried next run. It stops when the failure fraction is above
  half, including an early check after ten attempted videos.
- Triage writes successful chunk sidecars immediately. It stops when all
  transcript-bearing language-model calls fail.
- Cleaning skips existing cleaned chunks unless `--force` is supplied. Earlier
  successful chunks are saved if a later chunk in the video fails.
- The downloader treats existing media as resumable. It writes the source
  manifest before re-raising an unexpected worker exception. The command exits
  with status 2 when at least half of normal download outcomes fail.
- Pairing rewrites its output CSV on every run. Missing required batch files or
  manifest entries raise. Missing per-video chunks or replay masks are non-fatal.

## Known limits and current issues

- [Issue #12](https://github.com/ahalp90/badminton_cv_annotator/issues/12):
  annotation thresholds scale from the supplied FPS, but each end-to-end input
  still needs reliable FPS metadata. Variable-frame-rate phone footage cannot
  be represented faithfully by one scalar FPS and remains a known limit.
- [Issue #14](https://github.com/ahalp90/badminton_cv_annotator/issues/14): the
  dataset-builder refactor and documentation work continue.
- [Issue #15](https://github.com/ahalp90/badminton_cv_annotator/issues/15): the
  separate commands still need one recorded end-to-end trial.
- [Issue #28](https://github.com/ahalp90/badminton_cv_annotator/issues/28) and
  [#30](https://github.com/ahalp90/badminton_cv_annotator/issues/30): serve
  lookback remains measure-first, then conditional implementation work.
- [Issue #31](https://github.com/ahalp90/badminton_cv_annotator/issues/31):
  TrackNetV3/Inpaint hallucination follow-up remains open.
- [Issue #38](https://github.com/ahalp90/badminton_cv_annotator/issues/38): VLM
  scene filtering is recommended for broadcast cases that deterministic court,
  cut and motion signals cannot resolve.
- [Issue #40](https://github.com/ahalp90/badminton_cv_annotator/issues/40): run
  output compression and automated retention remain open.

## Historical source note

This document was checked against the current code on 8 August 2026. The former
private specification is retained as project history under
`local_scratch/autograder_architecture/archive/source_docs/20260808/` and is not
required to understand or run the current scraper.

