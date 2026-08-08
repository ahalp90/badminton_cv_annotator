"""Windowed doubles verdict from the per-frame in-court over-count.

The vision lane records a per-frame over-count bool (True where more than two
people project inside the court) beside each clip's pose output. A single noisy
frame is not doubles: a coach or ball-kid can cross the court for a moment. This
module turns the per-frame signal into a clip- or segment-level verdict.

The fraction-only rule has applied since 2026-07-07: over a span, raise
the flag when the over-count holds on more than ``DOUBLES_SPAN_FRACTION`` of
frames. A consecutive-run leg was considered and dropped: any passerby crossing
the court for half a second would trip it. Starting value lives in
``annotator.config``; tune it on the first labelled doubles sample.

Runnable as a small CLI to sweep a directory of ``<video_id>_overcount.npy``
arrays into ``doubles_flags.csv``::

    PYTHONPATH=src python -m annotator.doubles_flag \\
        --overcount-dir data/scrape_output/overcount \\
        --rally-spans data/scrape_output/rally_spans.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .config import DOUBLES_SPAN_FRACTION, SCRAPE_DIR

# Per-video over-count file naming and the CLI's output contract.
OVERCOUNT_SUFFIX = '_overcount.npy'
DOUBLES_FLAGS_CSV = SCRAPE_DIR / 'doubles_flags.csv'
DOUBLES_FLAGS_COLUMNS = ['video_id', 'rally_id', 'doubles_flag']


def read_whole_video_flags(csv_path: Path) -> dict[str, bool]:
    """Read unique whole-video doubles flags from a doubles flags CSV.

    :param csv_path: CSV written by this module, with the fixed three-column header.
    :return: Mapping from video ID to its whole-video doubles verdict.
    :raises ValueError: If the header, flag value or whole-video uniqueness is invalid.
    """
    with csv_path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != DOUBLES_FLAGS_COLUMNS:
            raise ValueError(
                f'{csv_path}: expected header {DOUBLES_FLAGS_COLUMNS}, got {reader.fieldnames}'
            )

        flags: dict[str, bool] = {}
        for row in reader:
            if row['rally_id'] != '':
                continue
            video_id = row['video_id']
            if video_id in flags:
                raise ValueError(f'{csv_path}: duplicate whole-video row for {video_id}')
            value = row['doubles_flag']
            if value not in {'True', 'False'}:
                raise ValueError(f'{csv_path}: invalid doubles_flag for {video_id}: {value!r}')
            flags[video_id] = value == 'True'
        return flags


def doubles_flag(overcount: np.ndarray, span: tuple[int, int] | None = None) -> bool:
    """Windowed doubles verdict for one span of the per-frame over-count.

    :param overcount: (F,) bool per-frame over-count; True where >2 people were
        in-court on that frame.
    :param span: ``(start, end)`` half-open slice bounds; ``None`` uses the whole
        array. Half-open (``overcount[start:end]``) keeps one slicing convention; a
        one-frame boundary is immaterial to the fraction rule.
    :return: True when the over-count holds on more than ``DOUBLES_SPAN_FRACTION`` of
        the span's frames.
    """
    window = overcount if span is None else overcount[span[0]:span[1]]
    if window.size == 0:
        return False
    return bool(window.mean() > DOUBLES_SPAN_FRACTION)


def _load_overcount(path: Path) -> np.ndarray:
    """Load a per-video over-count array as bool."""
    return np.load(path).astype(bool)


def _index_overcount_dir(overcount_dir: Path) -> dict[str, Path]:
    """Map ``video_id -> path`` for every ``<video_id>_overcount.npy`` in the dir."""
    return {
        path.name[: -len(OVERCOUNT_SUFFIX)]: path
        for path in sorted(overcount_dir.glob(f'*{OVERCOUNT_SUFFIX}'))
    }


def _whole_video_verdicts(overcount_paths: dict[str, Path]) -> list[dict]:
    """One whole-video verdict per over-count file; rally_id blank (no spans given)."""
    rows = []
    for video_id, path in overcount_paths.items():
        flag = doubles_flag(_load_overcount(path))
        rows.append({'video_id': video_id, 'rally_id': '', 'doubles_flag': str(flag)})
    return rows


def _span_verdicts(rally_spans_csv: Path, overcount_paths: dict[str, Path]) -> list[dict]:
    """One verdict per rally span, reading (video_id, rally_id, start_frame, end_frame).

    Over-count arrays are cached per video so a video with many spans loads once. A
    span whose video has no over-count file is logged and skipped (log-and-skip per
    item, per house convention).
    """
    if not rally_spans_csv.exists():
        raise FileNotFoundError(f'--rally-spans not found: {rally_spans_csv}')

    cache: dict[str, np.ndarray] = {}
    rows = []
    with rally_spans_csv.open(newline='', encoding='utf-8') as handle:
        for record in csv.DictReader(handle):
            video_id = record['video_id']
            path = overcount_paths.get(video_id)
            if path is None:
                print(f'  WARNING: no {OVERCOUNT_SUFFIX} for video {video_id!r}; skipping its span.')
                continue
            if video_id not in cache:
                cache[video_id] = _load_overcount(path)
            span = (int(record['start_frame']), int(record['end_frame']))
            flag = doubles_flag(cache[video_id], span)
            rows.append({
                'video_id': video_id,
                'rally_id': record['rally_id'],
                'doubles_flag': str(flag),
            })
    return rows


def _write_flags(rows: list[dict]) -> None:
    """Write the verdict rows to ``doubles_flags.csv`` with the fixed header.

    Bools serialise as the CSV strings 'True'/'False', matching the config
    contract: consumers parse (== 'True'), never truth-test a raw cell.
    """
    SCRAPE_DIR.mkdir(parents=True, exist_ok=True)
    with DOUBLES_FLAGS_CSV.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=DOUBLES_FLAGS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    """CLI: sweep a dir of over-count arrays into doubles_flags.csv (per span or whole-video)."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument(
        '--overcount-dir', type=Path, required=True,
        help=f'Directory of <video_id>{OVERCOUNT_SUFFIX} per-frame bool arrays.',
    )
    parser.add_argument(
        '--rally-spans', type=Path, default=None,
        help='Optional CSV (video_id, rally_id, start_frame, end_frame). Without it, '
             'one whole-video verdict per over-count file (rally_id blank).',
    )
    args = parser.parse_args(argv)

    if not args.overcount_dir.is_dir():
        parser.error(f'--overcount-dir not found: {args.overcount_dir}')

    overcount_paths = _index_overcount_dir(args.overcount_dir)
    if not overcount_paths:
        parser.error(f'no *{OVERCOUNT_SUFFIX} files under {args.overcount_dir}')

    if args.rally_spans is not None:
        rows = _span_verdicts(args.rally_spans, overcount_paths)
    else:
        rows = _whole_video_verdicts(overcount_paths)

    _write_flags(rows)
    print(f'Wrote {len(rows)} doubles verdict(s) to {DOUBLES_FLAGS_CSV}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
