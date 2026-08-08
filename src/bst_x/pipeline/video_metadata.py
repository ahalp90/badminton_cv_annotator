"""Build ShuttleSet resolution metadata from downloaded match videos.

Usage:
    python -m pipeline.video_metadata [--video-dir DIR]
"""

import argparse
import re
from datetime import date
from pathlib import Path

import cv2
import pandas as pd
from scraper.config import VIDEO_EXTENSIONS

from pipeline.config import RAW_VIDEO_DIR, RESOLUTION_CSV_PATH, SET_INFO_DIR


_YTDLP_FORMAT_STEM = re.compile(r'\.f\d+$')


def _video_id(video_path: Path) -> int | None:
    """Read an ID from ``{id}.ext`` or legacy ``{id} {name}.ext``."""
    try:
        return int(video_path.stem.split(' ', maxsplit=1)[0])
    except ValueError:
        return None


def find_video_files(video_dir: Path) -> dict[int, Path]:
    """Index supported ShuttleSet videos and reject duplicate ID matches."""
    matches: dict[int, list[Path]] = {}
    for video_path in sorted(video_dir.iterdir()):
        if (
            not video_path.is_file()
            or video_path.suffix.lower() not in VIDEO_EXTENSIONS
            or _YTDLP_FORMAT_STEM.search(video_path.stem)
        ):
            continue
        video_id = _video_id(video_path)
        if video_id is not None:
            matches.setdefault(video_id, []).append(video_path)

    ambiguous = {video_id: paths for video_id, paths in matches.items() if len(paths) > 1}
    if ambiguous:
        details = '; '.join(
            f'{video_id}: {", ".join(path.name for path in paths)}'
            for video_id, paths in sorted(ambiguous.items())
        )
        raise RuntimeError(f'multiple raw videos found for ShuttleSet ID: {details}')

    return {video_id: paths[0] for video_id, paths in matches.items()}


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
    video_files = find_video_files(video_dir)

    if not video_files:
        print(f'  WARNING: No video files found in {video_dir}')
        return pd.DataFrame(columns=['id', 'width', 'height'])

    rows = []
    for video_id, video_file in sorted(video_files.items()):
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f'  WARNING: Could not open {video_file.name}')
            continue
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        rows.append({'id': video_id, 'width': width, 'height': height})

    if not rows:
        raise RuntimeError(f'Could not read resolution from any video in {video_dir}')

    df = pd.DataFrame(rows, columns=['id', 'width', 'height'])
    if not df.empty:
        df = df.sort_values('id').reset_index(drop=True)
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


def main() -> None:
    """Build the ShuttleSet resolution CSV."""
    parser = argparse.ArgumentParser(
        description='Build resolution metadata from ShuttleSet match videos.',
    )
    parser.add_argument('--video-dir', type=Path, default=RAW_VIDEO_DIR,
                        help='Directory containing downloaded videos')
    parser.add_argument('--resolution-csv', type=Path, default=RESOLUTION_CSV_PATH,
                        help='Output path for resolution CSV')
    args = parser.parse_args()

    build_resolution_csv(video_dir=args.video_dir, output_path=args.resolution_csv)


if __name__ == '__main__':
    main()
