"""Adapt ShuttleSet match metadata to the scraper-owned video downloader."""

import argparse
import csv
import tempfile
from pathlib import Path

from scraper import config as scraper_config
from scraper.download_scraped_videos import DownloadOutcome, download_all_videos

from pipeline.config import EXCLUDED_VIDEOS, RAW_VIDEO_DIR, SET_INFO_DIR


SHUTTLESET_DATASET_LABEL = 'shuttleset'


def _candidate_rows(
    match_csv_path: Path,
    excluded: frozenset[int],
) -> list[dict[str, str]]:
    """Map included ShuttleSet matches to the scraper candidate schema."""
    with match_csv_path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))

    candidates: list[dict[str, str]] = []
    for row in rows:
        video_id = int(row['id'])
        if video_id in excluded:
            continue
        candidate = dict.fromkeys(scraper_config.CANDIDATES_COLUMNS, '')
        candidate.update(
            video_id=str(video_id),
            url=row['url'],
            title=row['video'],
            keep='True',
        )
        candidates.append(candidate)
    return candidates


def download_shuttleset_videos(
    match_csv_path: Path = SET_INFO_DIR / 'match.csv',
    output_dir: Path = RAW_VIDEO_DIR,
    excluded: frozenset[int] = EXCLUDED_VIDEOS,
    max_workers: int = scraper_config.DOWNLOAD_WORKERS,
) -> list[DownloadOutcome]:
    """Download included ShuttleSet matches through the scraper downloader."""
    candidates = _candidate_rows(match_csv_path, excluded)
    with tempfile.TemporaryDirectory(prefix='shuttleset-download-') as temporary_dir:
        candidates_path = Path(temporary_dir) / 'candidates.csv'
        with candidates_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=scraper_config.CANDIDATES_COLUMNS)
            writer.writeheader()
            writer.writerows(candidates)
        return download_all_videos(
            candidates_path=candidates_path,
            output_dir=output_dir,
            max_workers=max_workers,
            dataset=SHUTTLESET_DATASET_LABEL,
            video_only=True,
        )


def main() -> None:
    """Download ShuttleSet videos through the shared downloader."""
    parser = argparse.ArgumentParser(description='Download ShuttleSet match videos.')
    parser.add_argument('--match-csv', type=Path, default=SET_INFO_DIR / 'match.csv')
    parser.add_argument('--output-dir', type=Path, default=RAW_VIDEO_DIR)
    parser.add_argument('--workers', type=int, default=scraper_config.DOWNLOAD_WORKERS)
    args = parser.parse_args()
    download_shuttleset_videos(
        match_csv_path=args.match_csv,
        output_dir=args.output_dir,
        max_workers=args.workers,
    )


if __name__ == '__main__':
    main()
