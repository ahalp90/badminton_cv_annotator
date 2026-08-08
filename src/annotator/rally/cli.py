"""Command-line batch orchestration for rally segmentation."""

import argparse
import csv
from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np

from ..batch_report import VideoOutcome
from ..config import BaseAnnotatorConfig, CONTACT_FRAMES_CSV, RALLY_SPANS_CSV
from ..doubles_flag import read_whole_video_flags
from ..types import SpanOpen

log = logging.getLogger('annotator.rally_segmentation')

_SPAN_OPEN_CHOICES = {'region-start': SpanOpen.REGION_START, 'back-fill': SpanOpen.BACK_FILL}


def _format_bool(value: bool | None) -> str:
    """Serialise a guardrail bool for the CSV: 'True'/'False', blank when unmeasured.

    Matches the config's bool encoding note (consumers parse `== 'True'`).
    """
    if value is None:
        return ''
    return 'True' if value else 'False'


def _load_positions(pos_dir: Path | None, video_id: str) -> np.ndarray | None:
    """Load `<video_id>_pos.npy` from pos_dir if both are present, else None."""
    if pos_dir is None:
        return None
    pos_path = pos_dir / f'{video_id}_pos.npy'
    if not pos_path.exists():
        log.info('no positions for %s, proximity_ok left blank', video_id)
        return None
    return np.load(pos_path)


def _load_dead_mask(mask_dir: Path | None, video_id: str) -> np.ndarray | None:
    """Load `<video_id>_dead_mask.npy` from mask_dir if present, else None.

    A missing file means the video runs unmasked. A present-but-invalid mask hits
    `run_video`'s fail-loud checks, which the per-video log-and-skip in `main` catches.
    """
    if mask_dir is None:
        return None
    mask_path = mask_dir / f'{video_id}_dead_mask.npy'
    if not mask_path.exists():
        log.info('no dead mask for %s, running unmasked', video_id)
        return None
    return np.load(mask_path)


def _read_string_id_table(path: Path, label: str):
    """Read a table with unique string IDs and retain the indexed DataFrame."""
    import pandas as pd

    table = pd.read_csv(path, dtype={'id': str, 'video_id': str})
    id_column = 'id' if 'id' in table.columns else 'video_id' if 'video_id' in table.columns else None
    if id_column is None:
        raise ValueError(f'{label}: expected an id or video_id column')
    table[id_column] = table[id_column].astype(str)
    if table[id_column].duplicated().any():
        duplicate_ids = sorted(table.loc[table[id_column].duplicated(), id_column].unique())
        raise ValueError(f'{label}: duplicate IDs {duplicate_ids}')
    return table.set_index(id_column)


def _read_fps_table(path: Path):
    """Read a unique string-id FPS table written by commentary pairing."""
    table = _read_string_id_table(path, 'fps CSV')
    if 'fps' not in table.columns:
        raise ValueError('fps CSV: expected an fps column')
    return {str(video_id): float(row.fps) for video_id, row in table.iterrows()}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Rally segmentation: rally spans and contacts from shuttle tracks.')
    parser.add_argument('--shuttle-dir', type=Path, required=True,
                        help='Directory of <video_id>.npy (t, 3) shuttle tracks')
    parser.add_argument('--pos-dir', type=Path, default=None,
                        help='Optional directory of <video_id>_pos.npy court positions')
    parser.add_argument('--mask-dir', type=Path, default=None,
                        help='Optional directory of <video_id>_dead_mask.npy dead-time masks '
                             '(True = dead); a missing file runs that video unmasked')
    parser.add_argument('--doubles-csv', type=Path, default=None,
                        help='Optional doubles flags CSV; only whole-video False rows are processed')
    fps_group = parser.add_mutually_exclusive_group()
    fps_group.add_argument('--fps-csv', type=Path, default=None,
                           help='per-video id,fps table written by commentary pairing')
    fps_group.add_argument('--fps', type=float, default=None,
                           help='CFR override for every video in this run')
    parser.add_argument('--span-open', choices=tuple(_SPAN_OPEN_CHOICES), default=None,
                        help='optional span-opening rule: region-start (every active region '
                             'yields a span) or back-fill (a qualifying region opens at its '
                             'start). Default: open at the first qualifying burst')
    parser.add_argument('--rally-spans-csv', type=Path, default=RALLY_SPANS_CSV)
    parser.add_argument('--contact-frames-csv', type=Path, default=CONTACT_FRAMES_CSV)
    return parser


@dataclass
class _TrackFilterResult:
    """Track paths retained by doubles filtering and recorded exclusions."""

    included_paths: list[Path]
    outcomes_by_path: dict[Path, VideoOutcome]
    all_excluded_error: ValueError | None


def _filter_track_paths(track_paths: list[Path], doubles_csv: Path | None) -> _TrackFilterResult:
    """Apply whole-video doubles flags and record exclusions."""
    outcomes_by_path: dict[Path, VideoOutcome] = {}
    if doubles_csv is None:
        return _TrackFilterResult(track_paths, outcomes_by_path, None)

    whole_video_flags = read_whole_video_flags(doubles_csv)
    filtered_track_paths = []
    for track_path in track_paths:
        video_id = track_path.stem
        if video_id not in whole_video_flags:
            outcomes_by_path[track_path] = VideoOutcome(
                video_id, 'excluded', reason='no doubles row; not assuming singles',
            )
            log.warning('excluding %s: no doubles row; not assuming singles', video_id)
        elif whole_video_flags[video_id]:
            outcomes_by_path[track_path] = VideoOutcome(
                video_id, 'excluded', reason='flagged doubles',
            )
            log.warning('excluding %s: flagged doubles', video_id)
        else:
            filtered_track_paths.append(track_path)

    # One excluded clip is a log line; the whole batch excluded must block. A flags
    # CSV with no whole-video rows (e.g. this module's own per-rally CLI output)
    # would otherwise empty the batch and exit 0.
    all_excluded_error = None
    if track_paths and not filtered_track_paths:
        all_excluded_error = ValueError(
            f'{doubles_csv}: the doubles filter excluded every video in the batch; '
            'refusing to write empty outputs'
        )
    return _TrackFilterResult(filtered_track_paths, outcomes_by_path, all_excluded_error)


def _write_segmentation_csvs(
    rally_spans_path: Path,
    contact_frames_path: Path,
    span_rows: list[tuple[str, int, int, int]],
    contact_rows: list[tuple[str, int, int, str, str, str]],
) -> None:
    """Write the batch rally spans and raw contact candidates."""
    with rally_spans_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['video_id', 'rally_id', 'start_frame', 'end_frame'])
        writer.writerows(span_rows)
    # wrist_near is the pure body-unit gate verdict; suppressed is true only for a gate-passing
    # candidate that lost the suppression contest. Both are blank when a video ran without gate
    # inputs (the gate never ran), so its raw candidates stand. Every detected candidate is
    # written (the RAW set), so nothing recall-first loses its input.
    with contact_frames_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow([
            'video_id', 'rally_id', 'contact_frame', 'proximity_ok', 'wrist_near', 'suppressed',
        ])
        writer.writerows(contact_rows)
    log.info('wrote %d rally spans, %d contacts', len(span_rows), len(contact_rows))


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    if not args.shuttle_dir.is_dir():
        raise FileNotFoundError(f'shuttle dir not found: {args.shuttle_dir}')
    if args.fps is None and args.fps_csv is None:
        parser.error('one of --fps or --fps-csv is required before processing videos')

    fps_by_id = _read_fps_table(args.fps_csv) if args.fps_csv is not None else {}
    span_open = _SPAN_OPEN_CHOICES[args.span_open] if args.span_open is not None else None
    args.rally_spans_csv.parent.mkdir(parents=True, exist_ok=True)
    args.contact_frames_csv.parent.mkdir(parents=True, exist_ok=True)

    span_rows: list[tuple[str, int, int, int]] = []
    contact_rows: list[tuple[str, int, int, str, str, str]] = []
    input_track_paths = sorted(args.shuttle_dir.glob('*.npy'))
    from ..batch_report import publish_batch_report
    from ..run_video import run_video

    filter_result = _filter_track_paths(input_track_paths, args.doubles_csv)
    track_paths = filter_result.included_paths
    outcomes_by_path = filter_result.outcomes_by_path
    all_excluded_error = filter_result.all_excluded_error

    for track_path in track_paths:
        video_id = track_path.stem
        if args.fps is None and video_id not in fps_by_id:
            outcomes_by_path[track_path] = VideoOutcome(
                video_id, 'skipped', reason='absent from fps CSV',
            )
            log.warning('skipping %s: absent from fps CSV', video_id)
            continue
        try:
            if args.fps is not None:
                fps = args.fps
            else:
                fps = fps_by_id[video_id]
            track = np.load(track_path)
            positions = _load_positions(args.pos_dir, video_id)
            dead_mask = _load_dead_mask(args.mask_dir, video_id)
            result = run_video(
                track,
                fps=fps,
                base=BaseAnnotatorConfig(span_open=span_open),
                positions=positions,
                raw_exclusion_mask=(
                    dead_mask if dead_mask is not None
                    else np.zeros(len(track), dtype=bool)
                ),
                court_optional=True,
                stop_after_segmentation=True,
            )
            spans, contacts = result.spans, result.contacts
        except Exception as exc:  # log-and-skip per video: one bad track must not sink the batch
            exception_text = ' '.join(str(exc).split()) or type(exc).__name__
            outcomes_by_path[track_path] = VideoOutcome(
                video_id, 'skipped', reason=exception_text,
            )
            log.warning('skipping %s: %s', video_id, exc)
            continue
        outcomes_by_path[track_path] = VideoOutcome(
            video_id, 'processed', rallies=len(spans), contacts=len(contacts),
        )
        for rally_id, (start, end) in enumerate(spans):
            span_rows.append((video_id, rally_id, start, end))
        for contact in contacts:
            contact_rows.append((
                video_id, contact.rally_id, contact.contact_frame,
                _format_bool(contact.proximity_ok), _format_bool(contact.wrist_near),
                _format_bool(contact.suppressed),
            ))
        log.info('%s: %d rallies, %d contacts', video_id, len(spans), len(contacts))

    outcomes = [outcomes_by_path[track_path] for track_path in input_track_paths]
    processed_count = 0
    for outcome in outcomes:
        if outcome.status == 'processed':
            processed_count += 1
    none_processed = bool(outcomes) and processed_count == 0 and all_excluded_error is None
    none_processed_error = None
    if none_processed:
        none_processed_error = RuntimeError(
            f'batch processed 0 of {len(outcomes)} videos; refusing to write empty outputs'
        )
    terminal_error = all_excluded_error or none_processed_error

    if terminal_error is None:
        _write_segmentation_csvs(
            args.rally_spans_csv, args.contact_frames_csv, span_rows, contact_rows,
        )
    else:
        args.rally_spans_csv.unlink(missing_ok=True)
        args.contact_frames_csv.unlink(missing_ok=True)

    try:
        publish_batch_report(
            outcomes, args.rally_spans_csv,
            all_excluded=all_excluded_error is not None,
        )
    except Exception as publication_error:
        if terminal_error is not None:
            raise terminal_error from publication_error
        raise
    if terminal_error is not None:
        raise terminal_error
