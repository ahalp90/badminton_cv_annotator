"""Run TrackNetV3 on per-rally clips; accumulate into per-vid shuttle cache.

Operates on the rally clips produced by ``bric.preprocessing.slice_rallies``. For
each vid:

  1. Find all rally clips for that vid (``<vid>_<set>_<rally>.mp4``)
  2. For each clip: subprocess ``predict.py`` via the wrapper, parse the
     rally-local CSV (Frame, Visibility, X, Y in clip-frame coords)
  3. Re-key clip-frame -> source-frame using the rally's stored
     ``clip_start_frame`` from shots_master.csv
  4. Write a per-vid dense cache at ``training/bric/cache/shuttle/<vid>.npz``
     with arrays length = source video frame count. Non-rally frames
     have visibility=0, x=y=0 (placeholder; never read at training time
     since training only slices in-rally windows).

Why per-rally instead of full source video:
  - TrackNet's median-image computation gets a clean rally background
    (no replay/talking-head pollution)
  - TrackNet's memory accumulator stays bounded per-clip — full source
    videos OOM at ~25 GB per worker; rally clips peak at < 1 GB
  - Subprocess overhead amortises over hundreds of frames per rally,
    not 33k subprocess calls per-shot

Idempotency: skip only when the cache records the current rally bounds and
canonical source frame count. ``--force`` to redo.

Usage:
    uv run python -m bric.preprocessing.extract_shuttle              # all vids
    uv run python -m bric.preprocessing.extract_shuttle --vid 1
    uv run python -m bric.preprocessing.extract_shuttle --workers 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import tempfile
import zipfile
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from bric.perception.shuttle import extract_shuttle  # noqa: E402
from classifier_shared.video_io import get_video_info  # noqa: E402

SHOTS_MASTER_PATH = REPO_ROOT / 'training' / 'data' / 'shuttleset' / 'annotations' / 'shots_master.csv'
RAW_VIDEO_DIR = REPO_ROOT / 'training' / 'data' / 'shuttleset' / 'raw_video'
RALLY_CLIPS_DIR = REPO_ROOT / 'training' / 'data' / 'shuttleset' / 'rally_clips'
SHUTTLE_CACHE_DIR = REPO_ROOT / 'training' / 'bric' / 'cache' / 'shuttle'
BOUNDS_METADATA_VERSION = 1


def find_source_video(vid: int) -> Path | None:
    candidates = {RAW_VIDEO_DIR / f'{vid}.mp4'}
    candidates.update(RAW_VIDEO_DIR.glob(f'{vid} *.mp4'))
    candidates.update(RAW_VIDEO_DIR.glob(f'{vid}_*.mp4'))
    matches = sorted(path for path in candidates if path.is_file())
    if len(matches) > 1:
        raise RuntimeError(f'multiple canonical raw videos found for vid={vid}: {matches}')
    return matches[0] if matches else None


def compute_rally_bounds(
    strokes: pd.DataFrame,
    frame_count: int,
) -> dict[tuple[str, int], tuple[int, int]]:
    """Read validated per-rally ``(start_f, end_f)`` from shots_master.

    Must match the slicer's logic exactly so frame offsets line up.
    """
    out = {}
    for (set_id, rally), grp in strokes.groupby(['set_id', 'rally'], sort=True):
        starts = grp['clip_start_frame'].unique()
        ends = grp['clip_end_frame'].unique()
        if len(starts) != 1 or len(ends) != 1:
            raise ValueError(f'set={set_id} rally={rally}: inconsistent stored clip bounds')
        start_f = int(starts[0])
        end_f = int(ends[0])
        if not 0 <= start_f < end_f <= frame_count:
            raise ValueError(
                f'set={set_id} rally={rally}: clip bounds [{start_f}, {end_f}) '
                f'invalid for source frame count {frame_count}'
            )
        out[(set_id, int(rally))] = (start_f, end_f)
    return out


def clip_bounds_are_current(
    clip_path: Path,
    source_video: Path,
    source_frame_count: int,
    bounds: tuple[int, int],
    fps: float,
) -> bool:
    """Return whether the slicer's sidecar proves the expected mapping."""
    start_frame, end_frame = bounds
    expected = {
        'version': BOUNDS_METADATA_VERSION,
        'source_video': source_video.name,
        'source_frame_count': source_frame_count,
        'clip_start_frame': start_frame,
        'clip_end_frame': end_frame,
    }
    try:
        actual = json.loads(clip_path.with_suffix('.bounds.json').read_text(encoding='utf-8'))
        actual_fps = float(actual['fps'])
    except (KeyError, OSError, TypeError, ValueError):
        return False
    fields_match = all(actual.get(key) == value for key, value in expected.items())
    return fields_match and abs(actual_fps - fps) <= 0.01


def rally_bound_arrays(
    rally_bounds: dict[tuple[str, int], tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic cache metadata arrays for rally bounds."""
    keys = sorted(rally_bounds)
    rally_keys = np.asarray([f'{set_id}:{rally}' for set_id, rally in keys])
    starts = np.asarray([rally_bounds[key][0] for key in keys], dtype=np.int64)
    ends = np.asarray([rally_bounds[key][1] for key in keys], dtype=np.int64)
    return rally_keys, starts, ends


def shuttle_cache_is_current(
    cache_path: Path,
    rally_bounds: dict[tuple[str, int], tuple[int, int]],
    source_video: Path,
    source_frame_count: int,
) -> bool:
    """Return whether a shuttle cache was built for the current clip mapping."""
    if not cache_path.exists():
        return False
    expected_keys, expected_starts, expected_ends = rally_bound_arrays(rally_bounds)
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            data_arrays_match = all(
                cache[name].shape == (source_frame_count,)
                for name in ('frame', 'x', 'y', 'visibility')
            )
            return (
                data_arrays_match
                and int(cache['bounds_metadata_version'].item()) == BOUNDS_METADATA_VERSION
                and str(cache['source_video'].item()) == source_video.name
                and int(cache['source_frame_count'].item()) == source_frame_count
                and np.array_equal(cache['rally_keys'], expected_keys)
                and np.array_equal(cache['rally_clip_start_frames'], expected_starts)
                and np.array_equal(cache['rally_clip_end_frames'], expected_ends)
            )
    except (EOFError, OSError, KeyError, ValueError, zipfile.BadZipFile):
        return False


def parse_rally_clip_name(clip_path: Path) -> tuple[int, str, int]:
    """``<vid>_<set>_<rally>.mp4`` → (vid, 'setN', rally)."""
    stem = clip_path.stem
    parts = stem.split('_')
    if len(parts) != 3:
        raise ValueError(f'unexpected rally clip name: {clip_path.name}')
    return int(parts[0]), parts[1], int(parts[2])


def process_one_vid(vid: int, master: pd.DataFrame, force: bool = False) -> None:
    """Run TrackNet on every rally clip for one vid; write per-vid cache."""
    out_path = SHUTTLE_CACHE_DIR / f'{vid}.npz'
    source_video = find_source_video(vid)
    if source_video is None:
        print(f'vid={vid}: WARNING source video not found; cannot determine length, skipping',
              flush=True)
        return

    strokes = master[master['vid'] == vid]
    if strokes.empty:
        print(f'vid={vid}: no strokes in shots_master, skipping', flush=True)
        return

    info = get_video_info(source_video)
    rally_bounds = compute_rally_bounds(strokes, info.n_frames)
    if not force and shuttle_cache_is_current(out_path, rally_bounds, source_video, info.n_frames):
        print(f'vid={vid}: shuttle cache matches current clip bounds, skipping', flush=True)
        return
    if out_path.exists() and not force:
        print(f'vid={vid}: shuttle cache has stale or missing bounds metadata; regenerating', flush=True)

    rally_clips = sorted(RALLY_CLIPS_DIR.glob(f'{vid}_*.mp4'))
    if not rally_clips:
        print(f'vid={vid}: no rally clips found in {RALLY_CLIPS_DIR}; '
              f'run bric.preprocessing.slice_rallies first', flush=True)
        return

    clips_by_key = {}
    for clip_path in rally_clips:
        try:
            clip_vid, set_id, rally = parse_rally_clip_name(clip_path)
        except ValueError as e:
            print(f'  {clip_path.name}: {e}, skipping', flush=True)
            continue
        if clip_vid == vid:
            clips_by_key[(set_id, rally)] = clip_path

    missing_clips = sorted(set(rally_bounds) - set(clips_by_key))
    if missing_clips:
        print(f'vid={vid}: {len(missing_clips)} current rally clips are missing; run '
              f'bric.preprocessing.slice_rallies first', flush=True)
        return

    fps = info.fps
    stale_clips = [
        clips_by_key[key]
        for key, bounds in rally_bounds.items()
        if not clip_bounds_are_current(
            clips_by_key[key], source_video, info.n_frames, bounds, fps,
        )
    ]
    if stale_clips:
        print(f'vid={vid}: {len(stale_clips)} rally clips have stale or missing bounds metadata; '
              f'run bric.preprocessing.slice_rallies first', flush=True)
        return

    # Pre-allocate dense per-vid arrays sized to source video length.
    n_total = info.n_frames
    visibility = np.zeros(n_total, dtype=np.int32)
    x = np.zeros(n_total, dtype=np.float32)
    y = np.zeros(n_total, dtype=np.float32)

    print(f'vid={vid}: {len(rally_clips)} rally clips → {n_total:,}-frame dense cache',
          flush=True)
    n_done = n_skipped = n_failed = 0

    with tempfile.TemporaryDirectory(prefix=f'tracknet_{vid}_') as tmpdir:
        for (set_id, rally), bounds in rally_bounds.items():
            clip_path = clips_by_key[(set_id, rally)]
            rally_start_f, _ = bounds

            try:
                csv_path = extract_shuttle(clip_path, save_dir=Path(tmpdir))
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f'  {clip_path.name}: TrackNet failed: {type(e).__name__}: {e}',
                      flush=True)
                n_failed += 1
                continue

            # Re-key clip-frame → source-frame and write into the dense arrays.
            src_frames = df['Frame'].to_numpy(dtype=np.int64) + rally_start_f
            valid = (src_frames >= 0) & (src_frames < n_total)
            src_frames = src_frames[valid]
            visibility[src_frames] = df['Visibility'].to_numpy(dtype=np.int32)[valid]
            x[src_frames] = df['X'].to_numpy(dtype=np.float32)[valid]
            y[src_frames] = df['Y'].to_numpy(dtype=np.float32)[valid]

            # Free the per-rally CSV — temp dir would also clean it up at exit.
            csv_path.unlink(missing_ok=True)
            n_done += 1

    if n_failed:
        out_path.unlink(missing_ok=True)
        print(f'vid={vid}: not writing cache because {n_failed} rallies failed', flush=True)
        return

    rally_keys, rally_starts, rally_ends = rally_bound_arrays(rally_bounds)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        frame=np.arange(n_total, dtype=np.int32),
        x=x, y=y, visibility=visibility,
        bounds_metadata_version=np.int32(BOUNDS_METADATA_VERSION),
        source_video=np.asarray(source_video.name),
        source_frame_count=np.int64(n_total),
        rally_keys=rally_keys,
        rally_clip_start_frames=rally_starts,
        rally_clip_end_frames=rally_ends,
    )
    n_visible = int((visibility > 0).sum())
    print(
        f'vid={vid}: wrote {out_path.name} — {n_total:,} frames total, '
        f'{n_visible:,} visible ({100 * n_visible / n_total:.1f}%); '
        f'{n_done} rallies processed, {n_skipped} skipped, {n_failed} failed',
        flush=True,
    )


def _worker(vid: int, master: pd.DataFrame, force: bool) -> None:
    try:
        process_one_vid(vid, master, force=force)
    except Exception as e:
        print(f'vid={vid}: ERROR {type(e).__name__}: {e}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vid', type=int, nargs='*',
                        help='Specific vid(s). Default: all in shots_master.csv.')
    parser.add_argument('--force', action='store_true',
                        help='Re-run even if cache exists.')
    parser.add_argument(
        '--workers', type=int, default=1,
        help='Concurrent vids. Each spawns a TrackNet subprocess per rally clip; '
             'memory bounded per-rally so high concurrency is fine. 8 is sensible '
             'on a single Blackwell GPU. Default 1.',
    )
    args = parser.parse_args()

    master = pd.read_csv(SHOTS_MASTER_PATH)
    if args.vid:
        vids = sorted(set(args.vid))
    else:
        vids = sorted(master['vid'].unique().tolist())

    n_workers = max(1, min(args.workers, len(vids)))
    print(f'Processing {len(vids)} vid(s) with {n_workers} worker(s): {vids}')

    if n_workers == 1:
        for vid in vids:
            _worker(int(vid), master, args.force)
        return

    ctx = mp.get_context('spawn')
    with ctx.Pool(n_workers) as pool:
        pool.map(partial(_worker, master=master, force=args.force),
                 [int(v) for v in vids])


if __name__ == '__main__':
    main()
