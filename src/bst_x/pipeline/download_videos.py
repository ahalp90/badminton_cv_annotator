"""Download ShuttleSet match videos from YouTube and build resolution CSV.

Uses yt-dlp for downloading (YouTube requires specialised tooling, not plain
HTTP requests). Adapted from user's parallel image downloader pattern.

Usage:
    python -m pipeline.download_videos [--output-dir DIR] [--workers N]
"""
import argparse
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import cv2
import pandas as pd

from pipeline.config import SET_INFO_DIR, RAW_VIDEO_DIR, EXCLUDED_VIDEOS, RESOLUTION_CSV_PATH

_VIDEO_EXTS = {'.mp4', '.mkv', '.webm', '.avi'}


def _check_ytdlp() -> None:
    """Verify yt-dlp is installed before spawning worker threads."""
    if not shutil.which('yt-dlp'):
        raise RuntimeError(
            'yt-dlp not found in PATH. Install with: pip install yt-dlp'
        )


def download_video(
    url: str,
    video_id: int | str,
    video_name: str,
    output_dir: Path,
) -> str | None:
    """Download one video using yt-dlp.

    Saves as '{video_id} {video_name}.mp4'. Skips if any file matching that
    pattern already exists.

    :param url: YouTube video URL.
    :param video_id: ID from the source CSV (ShuttleSet int, or a YouTube string id).
    :param video_name: Match name from match.csv (used in filename).
    :param output_dir: Directory to save the downloaded video.
    :return: Filename on success, None on failure.
    """
    # Any extension, since yt-dlp may choose .mkv etc.
    existing = list(output_dir.glob(f'{video_id} {video_name}.*'))
    if existing:
        print(f'  Skipping video {video_id} (already exists: {existing[0].name})')
        return existing[0].name

    output_template = str(output_dir / f'{video_id} {video_name}.%(ext)s')
    try:
        result = subprocess.run(
            [
                'yt-dlp',
                # avc1 is the H.264 fourcc (not AV1, despite the name). YouTube
                # now serves AV1-in-mp4 by default and the HPC nodes' cv2 cannot
                # decode it (2026-07-08 pilot finding). No unpinned fallback; an
                # AV1 file is unusable downstream, so fail loudly over a dud.
                '--format', 'bestvideo[vcodec^=avc1][ext=mp4]/best[vcodec^=avc1][ext=mp4]',
                '--output', output_template,
                '--no-playlist',
                '--retries', '3',
                # D22 ban-avoidance throttle (D22 stack, scraper_spec.md s5). bst_x
                # cannot import from src/scraper, so these values are deliberately
                # duplicated from src/scraper/config.py rather than shared.
                '--sleep-interval', '5',
                '--max-sleep-interval', '15',
                '--sleep-requests', '10',
                '--limit-rate', '2M',
                '--concurrent-fragments', '1',
                url,
            ],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min per video (full matches can be large)
        )
        if result.returncode != 0:
            print(f'  ERROR video {video_id}: {result.stderr.strip()[:200]}')
            return None

        downloaded = list(output_dir.glob(f'{video_id} {video_name}.*'))
        if downloaded:
            print(f'  Downloaded video {video_id}: {downloaded[0].name}')
            return downloaded[0].name
        return None

    except subprocess.TimeoutExpired:
        print(f'  TIMEOUT video {video_id}: download exceeded 30 minutes')
        return None
    except Exception as e:
        print(f'  ERROR video {video_id}: {e}')
        return None


def download_all_videos(
    match_csv_path: Path = SET_INFO_DIR / 'match.csv',
    output_dir: Path = RAW_VIDEO_DIR,
    excluded: frozenset[int] = EXCLUDED_VIDEOS,
    max_workers: int = 2,  # D22: fewer parallel workers is part of the ban-avoidance stack
    id_col: str = 'id',
    url_col: str = 'url',
    name_col: str = 'video',
    keep_col: str | None = None,
) -> list[str]:
    """Download all ShuttleSet match videos from YouTube in parallel.

    Column-name params default to match.csv's names, so the ShuttleSet path is
    unchanged. Point them at a stage-3 keep-list CSV (video_id, url, title, keep)
    and pass keep_col to feed only the kept rows straight in.

    :param match_csv_path: Path to the source CSV with video URLs.
    :param output_dir: Directory to save downloaded videos.
    :param excluded: Video IDs to skip. Defaults to config.EXCLUDED_VIDEOS.
    :param max_workers: Number of parallel download threads.
    :param id_col: Column holding the video id (filename stem).
    :param url_col: Column holding the video URL.
    :param name_col: Column holding the video/match name (filename).
    :param keep_col: Optional keep-flag column; keeps only rows parsing 'True'.
    :return: List of successfully downloaded filenames.
    """
    _check_ytdlp()

    output_dir.mkdir(parents=True, exist_ok=True)

    match_df = pd.read_csv(match_csv_path)
    # Filter out excluded videos
    match_df = match_df[~match_df[id_col].isin(excluded)]

    if keep_col is not None:
        # astype(str) WHY: pandas parses an all-bool CSV column to bool dtype,
        # while a mixed or blank-bearing column stays string; casting makes both
        # compare identically against 'True'. Truth-testing a raw cell is banned:
        # any non-empty string is truthy, 'False' included.
        match_df = match_df[match_df[keep_col].astype(str) == 'True']

    tasks = [
        # YouTube ids are strings; ShuttleSet's int64 ids format identically in
        # the filename f-string, so pass row[id_col] as-is (no int() cast).
        (row[url_col], row[id_col], row[name_col], output_dir)
        for _, row in match_df.iterrows()
    ]

    print(f'Downloading {len(tasks)} videos (excluding {sorted(excluded)})...')
    print(f'Using {max_workers} parallel workers')

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_video, *t) for t in tasks]
        for future in futures:
            result = future.result()
            if result:
                results.append(result)

    print(f'\nFinished: {len(results)}/{len(tasks)} videos downloaded successfully.')
    return results


def build_resolution_csv(
    video_dir: Path = RAW_VIDEO_DIR,
    output_path: Path = RESOLUTION_CSV_PATH,
) -> pd.DataFrame:
    """Scan downloaded videos and write a resolution CSV (id, width, height).

    Uses OpenCV to read video properties. This replaces the need to manually
    create my_raw_video_resolution.csv.

    :param video_dir: Directory containing downloaded match videos.
    :param output_path: Output path for the resolution CSV.
    :return: DataFrame with columns id, width, height.
    """
    video_files = [
        f for f in sorted(video_dir.iterdir())
        if f.suffix.lower() in _VIDEO_EXTS
    ]

    if not video_files:
        print(f'  WARNING: No video files found in {video_dir}')
        return pd.DataFrame(columns=['id', 'width', 'height'])

    rows = []
    for video_file in video_files:
        # Extract video ID from filename pattern: '{id} {name}.ext'
        try:
            vid_id = int(video_file.name.split(' ')[0])
        except (ValueError, IndexError):
            continue

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f'  WARNING: Could not open {video_file.name}')
            continue
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        rows.append({'id': vid_id, 'width': width, 'height': height})

    df = pd.DataFrame(rows).sort_values('id').reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f'Resolution CSV written: {output_path} ({len(df)} videos)')

    # Compare found videos against the full match.csv source
    match_csv_path = SET_INFO_DIR / 'match.csv'
    if match_csv_path.exists():
        expected_ids = set(pd.read_csv(match_csv_path)['id'].astype(int))
        found_ids = set(df['id'].astype(int))
        missing_ids = sorted(expected_ids - found_ids)

        print(f'  Resolution CSV: {len(found_ids)}/{len(expected_ids)} '
              f'expected videos found', end='')
        if missing_ids:
            print(f' (missing: {missing_ids})')
        else:
            print()

        missing_txt_path = output_path.parent / 'my_raw_video_resolution_csv_missing.txt'
        if missing_ids:
            msg = (
                f'On {date.today()} build_resolution_csv produced a CSV of '
                f'{len(found_ids)} video resolution readings but '
                f'{len(expected_ids)} were expected from match.csv.\n'
                f'Missing video IDs: {missing_ids}\n'
            )
            missing_txt_path.write_text(msg)
            print(f'  Missing video report written: {missing_txt_path}')
        elif missing_txt_path.exists():
            missing_txt_path.unlink()

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Download ShuttleSet match videos from YouTube.',
    )
    parser.add_argument('--output-dir', type=Path, default=RAW_VIDEO_DIR,
                        help='Directory to save downloaded videos')
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of parallel download workers')
    parser.add_argument('--resolution-csv', type=Path, default=RESOLUTION_CSV_PATH,
                        help='Output path for resolution CSV')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip video download, only build resolution CSV')
    args = parser.parse_args()

    if not args.skip_download:
        print('=== Downloading videos ===')
        download_all_videos(output_dir=args.output_dir, max_workers=args.workers)

    print('\n=== Building resolution CSV ===')
    build_resolution_csv(video_dir=args.output_dir, output_path=args.resolution_csv)


if __name__ == '__main__':
    main()
