Result: Claim C is PARTLY supported. Missing sidecars use documented fallbacks, but existing unreadable sidecars raise rather than falling back.

## 1. Claim under test

For an eligible video, absent chunk or replay-mask sidecars may produce blank or unmasked pairing output without a sidecar-specific diagnostic or pairing-stage failure.

## 2. Evidence SUPPORTING the claim

### C1: Eligibility does not come from chunk files

- The CLI reads FPS and rally-span inputs, then derives `fps_video_ids` from those inputs: `src/scraper/commentary_pairing.py:332-338`.
- `_manifest_pairing_index()` requires an existing video and a boolean `commentary_eligible` manifest field, but accepts no chunk directory and checks no sidecar: `src/scraper/commentary_pairing.py:259-300`.
- Chunk and mask loading happens only after the manifest flag is read: `src/scraper/commentary_pairing.py:341-349`.
- The downloader writes `commentary_eligible` into `sources.toml`: `src/scraper/download_scraped_videos.py:260-273`.
- Relevance triage writes `chunks/<video_id>.json` separately after transcript and triage processing: `src/scraper/relevance_triage.py:236-261`.

### C2: Absent-sidecar fallbacks

`src/scraper/commentary_pairing.py:219-225`:

```python
def _load_chunks(chunks_dir: Path, video_id: str) -> list[dict]:
    """Load `<video_id>.json` chunk sidecar, or [] if absent."""
    chunk_path = chunks_dir / f'{video_id}.json'
    if not chunk_path.exists():
        return []
    with chunk_path.open(encoding='utf-8') as handle:
        return json.load(handle)
```

`src/scraper/commentary_pairing.py:228-236`:

```python
def _load_replay_mask(masks_dir: Path, video_id: str) -> np.ndarray | None:
    """Load a one-dimensional boolean `<video_id>_replay.npy`, or None if absent."""
    mask_path = masks_dir / f'{video_id}_replay.npy'
    if not mask_path.exists():
        return None
    replay_mask = np.load(mask_path)
    if replay_mask.ndim != 1 or replay_mask.dtype != np.bool_:
        raise ValueError(f'{mask_path} must be a one-dimensional boolean array')
    return replay_mask
```

The absent branches contain only `return []` or `return None`. They do not log, print, raise, or increment a counter.

### C3: A missing mask disables replay filtering

`src/scraper/commentary_pairing.py:151-154` sets the filtered mask to `None` when no mask is loaded:

```python
duration_filtered_replay_mask = (
    None if replay_mask is None
    else filter_short_exclusion_runs(replay_mask, minimum_run)
)
```

Both replay checks are guarded by `is not None`: `src/scraper/commentary_pairing.py:162-185`.

With `None`, the timing join still runs. The direct test passes `None` and asserts a pair is produced: `tests/test_scraper_commentary_pairing.py:135-143`.

### C4: Pairing writes output without a sidecar guard

The pairing stage counts pairs and logs them, then writes all rows:

`src/scraper/commentary_pairing.py:354-364`

```python
rows = pair_video(video_id, rally_spans, chunks, replay_mask, fps_map[video_id])
all_rows.extend(rows)
paired = sum(1 for row in rows if row['chunk_id'])
log.info('%s: %d rallies, %d paired', video_id, len(rows), paired)

args.pairs_csv.parent.mkdir(parents=True, exist_ok=True)
with args.pairs_csv.open('w', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=PAIRS_COLUMNS)
    writer.writeheader()
    writer.writerows(all_rows)
log.info('wrote %d pair rows -> %s', len(all_rows), args.pairs_csv)
```

## 3. Evidence REFUTING or weakening the claim

- `_load_chunks()` does not convert an existing unreadable or malformed file to `[]`; `open()` and `json.load()` are uncaught: `src/scraper/commentary_pairing.py:222-225`.
- `_load_replay_mask()` does not convert an existing unreadable file to `None`; `np.load()` is uncaught, and invalid shape or dtype raises `ValueError`: `src/scraper/commentary_pairing.py:230-236`.
- The main loop emits an informational per-video count such as `N rallies, 0 paired`: `src/scraper/commentary_pairing.py:356-357`. No missing-sidecar warning is emitted.
- A replay-mask integration test indirectly covers an absent mask: it passes an uncreated `masks` directory and asserts normal pairing: `tests/test_scraper_commentary_pairing.py:256-293`.
- Direct mask-loader tests cover malformed present files, not an absent file: `tests/test_scraper_commentary_pairing.py:157-165`.
- No test directly calls `_load_chunks()` for a missing chunk file. `test_stale_missing_manifest_basename_is_ignored` omits `chunks_dir` and asserts a blank row, but its setup is a manifest-mapping test and relies on the default chunk directory: `tests/test_scraper_commentary_pairing.py:532-564`.
- Explicitly ineligible videos are deliberately assigned `chunks = []` and `replay_mask = None`, with an informational log: `src/scraper/commentary_pairing.py:350-353`.

## 4. Verdicts

- C1: CONFIRMED. Selection uses rally spans, FPS, the video file, and `sources.toml`; it does not require chunk files.
- C2: PARTLY. Absent files silently return `[]` or `None`. Existing unreadable or malformed files propagate exceptions.
- C3: CONFIRMED for unmasked output. `None` bypasses both replay checks, so available chunks can pair without replay filtering.
- C4 counter: CONFIRMED. A per-video `paired` count and batch row count are logged.
- C4 guard: REFUTED. No pairing-stage threshold, all-fail check, or missing-sidecar guard uses those counts.
- C5: PARTLY. Missing-mask behaviour is indirectly tested; no direct missing-chunk loader test was found.
- C6 fallback documentation: CONFIRMED. Both loader docstrings explicitly document the absent-file fallback, and `pair_video()` types `replay_mask` as nullable: `src/scraper/commentary_pairing.py:143`, `:220`, `:229`.
- C6 rationale: no explicit “mask optional by design” wording was found.

## 5. Unresolved or dynamic surfaces

- “Unreadable” is ambiguous between absent, malformed, permission-denied, and invalid-format files. The source distinguishes absent fallback from existing-file exceptions.
- The module documents `python -m scraper.commentary_pairing` and invokes `main()` under the module guard: `src/scraper/commentary_pairing.py:13`, `:367-368`.
- No `getattr`, import-based dispatch, or alternate pairing entry point was found under `src/scraper`; tests call `main()` directly and monkeypatch `_load_replay_mask`: `tests/test_scraper_commentary_pairing.py:341-378`.

Checks run: Serena `get_symbols_overview`, `find_symbol`, `find_referencing_symbols`, `search_for_pattern`, and `find_declaration`; `rg`; `nl -ba`; and `git diff`. The Serena launcher could not acquire its state lock because the filesystem is read-only, but the active Serena MCP answered all queries. No runtime tests were run.

Files inspected: `src/scraper/commentary_pairing.py`, `tests/test_scraper_commentary_pairing.py`, `src/scraper/config.py`, `src/scraper/download_scraped_videos.py`, `src/scraper/relevance_triage.py`, `src/scraper/transcript_acquisition.py`, `.github/AGENTS.md`, and `.codex/context.md`.

No files were changed. The target-file diff is empty.