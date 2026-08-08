"""Build the replay and off-rally mask.

See ``docs/scraper_pipeline/scraper_architecture.md`` for the pipeline context.

Three independent per-frame boolean signals unioned into one `(frames,)` mask,
true where the frame is a replay or otherwise off-rally. Saved per video to
`config.MASKS_DIR / f'{video_id}_replay.npy'`, the same 1-D bool-over-frames
convention as the pose `_failed.npy`.

The three signals are independent producers (court, homography, shuttle), so a
missing input contributes an all-False mask with a log line rather than killing
the union: an absent court mask must not veto a real perspective-shift replay.

Speed and its helpers come from the shared annotator declarations, re-exported
by rally segmentation. The slow-motion signal therefore reads the same per-frame speed as
the rally rules, not a second definition.

Run as `python -m annotator.replay_mask --video-id ...` with PYTHONPATH=src.
"""
import argparse
import csv
import logging
from pathlib import Path

import numpy as np

from .config import (
    BaseAnnotatorConfig,
    MASKS_DIR,
    PERSPECTIVE_SHIFT_THRESHOLD,
    RALLY_SPANS_CSV,
    SLOWMO_SPEED_FRAC,
)
from .fps_constants import scale_for_fps
from .inpaint_guard import code_counts, grade_track
from .rally_segmentation import compute_speed, rolling_nanmedian, true_runs

log = logging.getLogger(__name__)

# Corner columns of the per-segment homography CSV, in `[corner, xy]` order so
# the flat 8-vector reshapes straight to (4, 2). Read by name (DictReader), so
# the physical column order in the file is irrelevant.
HOMOGRAPHY_CORNER_COLS = [
    'upleft_x', 'upleft_y',
    'upright_x', 'upright_y',
    'downleft_x', 'downleft_y',
    'downright_x', 'downright_y',
]


# ---------------------------------------------------------------------------
# Signal 1: court absence
# ---------------------------------------------------------------------------
def court_absence_signal(court_present: np.ndarray | None, n_frames: int, fps: float) -> np.ndarray:
    """Fire across court-absent runs that reach the fps-scaled window.

    A sustained absence (>= the window) masks its whole run; a one- or two-frame
    detector blip does not. The run-length gate masks the entire sustained
    absence rather than a single window-sized slice.

    :param court_present: `(frames,)` bool court-present flag, or None.
    :param n_frames: video frame count (the mask length).
    :return: `(n_frames,)` bool signal.
    """
    mask = np.zeros(n_frames, dtype=bool)
    if court_present is None:
        log.info('court-present mask missing; court-absence signal all-False')
        return mask
    if len(court_present) != n_frames:
        raise ValueError(f'court-present length {len(court_present)} != n_frames {n_frames}')
    absent = ~court_present.astype(bool)  # (frames,)
    for start, end in true_runs(absent):
        window = scale_for_fps(fps).court_absent_window
        if end - start >= window:
            mask[start:end] = True
    return mask


# ---------------------------------------------------------------------------
# Signal 2: perspective shift
# ---------------------------------------------------------------------------
def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median: the smallest value whose cumulative weight reaches half.

    Ties resolve to the lower value (searchsorted left). Used to pick the
    dominant broadcast view, weighting each segment by its frame duration.

    :param values: `(k,)` values.
    :param weights: `(k,)` non-negative weights.
    :return: the weighted-median value.
    """
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    half = 0.5 * cumulative[-1]
    idx = int(np.searchsorted(cumulative, half))
    return float(sorted_values[min(idx, len(sorted_values) - 1)])


def perspective_shift_signal(homography_rows: list[dict] | None, n_frames: int) -> np.ndarray:
    """Fire on segments whose court corners deviate from the dominant view.

    The dominant broadcast view is the duration-weighted median of each corner
    coordinate across all segments; a segment whose mean corner displacement
    from it (normalised by the reference court's bounding-box diagonal) exceeds
    PERSPECTIVE_SHIFT_THRESHOLD is a replay or cutaway angle and its frames fire.

    Comparing every segment to the dominant view avoids masking the whole tail
    after one legitimate camera change. The deviant minority view is the replay,
    and the normalisation is self-contained (diagonal of the reference corners),
    so no frame resolution is needed.

    :param homography_rows: per-segment dict rows for this video, or None/empty.
    :param n_frames: video frame count (the mask length).
    :return: `(n_frames,)` bool signal.
    """
    mask = np.zeros(n_frames, dtype=bool)
    if not homography_rows:
        log.info('homography rows missing; perspective-shift signal all-False')
        return mask

    starts = np.array([int(row['start_frame']) for row in homography_rows])  # (n_seg,)
    ends = np.array([int(row['end_frame']) for row in homography_rows])  # (n_seg,)
    durations = (ends - starts).astype(float)  # (n_seg,) weight
    corners = np.array(
        [[float(row[col]) for col in HOMOGRAPHY_CORNER_COLS] for row in homography_rows]
    )  # (n_seg, 8)

    reference = np.array(
        [_weighted_median(corners[:, col], durations) for col in range(corners.shape[1])]
    )  # (8,) dominant view
    seg_xy = corners.reshape(len(homography_rows), 4, 2)  # (n_seg, 4, 2) [corner, xy]
    ref_xy = reference.reshape(4, 2)  # (4, 2)

    mean_displacement = np.linalg.norm(seg_xy - ref_xy, axis=2).mean(axis=1)  # (n_seg,)
    ref_span = ref_xy.max(axis=0) - ref_xy.min(axis=0)  # (2,) bbox width, height
    diagonal = float(np.hypot(*ref_span))
    if diagonal <= 0:
        log.info('degenerate reference court (zero diagonal); perspective-shift signal all-False')
        return mask

    shifted = (mean_displacement / diagonal) > PERSPECTIVE_SHIFT_THRESHOLD  # (n_seg,)
    for seg in np.flatnonzero(shifted):
        mask[starts[seg]:ends[seg]] = True
    return mask


# ---------------------------------------------------------------------------
# Signal 3: velocity drop (slow motion)
# ---------------------------------------------------------------------------
def _validate_optional_frame_mask(name: str, values: np.ndarray | None, n_frames: int) -> None:
    """Validate one optional frame-aligned boolean mask."""
    if values is not None and (
        not isinstance(values, np.ndarray)
        or values.ndim != 1
        or values.dtype != np.bool_
        or len(values) != n_frames
    ):
        raise ValueError(f'{name} must be a one-dimensional np.bool_ array of length {n_frames}')


def velocity_drop_signal(
    track: np.ndarray | None, rally_spans: list[tuple[int, int]] | None, n_frames: int, fps: float,
    *, non_evidence: np.ndarray | None = None, baseline_exclude: np.ndarray | None = None,
) -> np.ndarray:
    """Fire where visible shuttle speed drops well below the rally norm (slow-mo).

    The rally norm is the median per-frame speed across all rally-span frames of
    the video. A frame fires when its rolling median speed sits in the slow-mo
    band (at or above REST_SPEED but below SLOWMO_SPEED_FRAC of the norm) AND
    the shuttle is visible. Invisible frames are the court-absence signal's job.
    Genuine rest (below REST_SPEED) deliberately does NOT fire: a resting
    shuttle is the between-rallies state, and that is exactly where the
    commentary pairing joins with live spans; masking rest would hold every
    post-rally chunk out of pairing. Slow motion means moving, slowly.

    Speed steps touching a non_evidence frame are unmeasured: they feed neither the baseline nor the rolling median; edge frames of a graded run can still fire on neighbouring measured evidence, and a window with no measured step reads not-slow.

    :param track: `(t, 3)` whole-video shuttle track, or None.
    :param rally_spans: `[(start, end), ...]` rally spans for this video, or None/empty.
    :param n_frames: video frame count (the mask length).
    :return: `(n_frames,)` bool signal.
    """
    mask = np.zeros(n_frames, dtype=bool)
    if track is None or not rally_spans:
        log.info('shuttle track or rally spans missing; velocity-drop signal all-False')
        return mask
    if len(track) != n_frames:
        raise ValueError(f'shuttle track length {len(track)} != n_frames {n_frames}')
    _validate_optional_frame_mask('non_evidence', non_evidence, len(track))
    _validate_optional_frame_mask('baseline_exclude', baseline_exclude, len(track))

    speed = compute_speed(track)  # (t,) per-frame, NaN on non-visible steps
    if non_evidence is not None:
        unmeasured_steps = non_evidence[1:] | non_evidence[:-1]  # (t-1,) either endpoint is graded
        speed[1:][unmeasured_steps] = np.nan
    in_rally = np.zeros(len(track), dtype=bool)  # (t,) frames inside any rally span
    for start, end in rally_spans:
        in_rally[start:end] = True
    if baseline_exclude is not None:
        in_rally &= ~baseline_exclude

    rally_speed = speed[in_rally]
    if not np.any(~np.isnan(rally_speed)):
        log.info('no visible rally-span frames; velocity-drop signal all-False')
        return mask
    rally_median = float(np.nanmedian(rally_speed))
    if rally_median <= 0:
        log.info('rally median speed is zero; velocity-drop signal all-False')
        return mask

    values = scale_for_fps(fps)
    rolling_median = rolling_nanmedian(speed, values.rest_window)  # (t,)
    visible = track[:, 2] == 1  # (t,)
    below_norm = rolling_median < (SLOWMO_SPEED_FRAC * rally_median)  # NaN windows read not-slow
    moving = rolling_median >= values.rest_speed  # rest is not slow-mo (see docstring)
    mask[below_norm & moving & visible] = True
    return mask


# ---------------------------------------------------------------------------
# Combination
# ---------------------------------------------------------------------------
def combine_mask(
    court_present: np.ndarray | None,
    homography_rows: list[dict] | None,
    track: np.ndarray | None,
    rally_spans: list[tuple[int, int]] | None,
    n_frames: int,
    fps: float,
    *, non_evidence: np.ndarray | None = None,
) -> np.ndarray:
    """Union the three replay/off-rally signals using an any-of rule.

    :return: `(n_frames,)` bool mask, True where any signal fires.
    """
    court = court_absence_signal(court_present, n_frames, fps)
    perspective = perspective_shift_signal(homography_rows, n_frames)
    velocity = velocity_drop_signal(
        track, rally_spans, n_frames, fps,
        non_evidence=non_evidence, baseline_exclude=court | perspective,
    )
    return court | perspective | velocity


def filter_short_exclusion_runs(mask: np.ndarray, min_frames: int) -> np.ndarray:
    """Keep each raw detector run in full when it is long enough.

    The input contains raw detector flags. For each raw run ``[start, end)``,
    filtering keeps the whole run when ``end - start >= min_frames``. Applying
    this function to an already filtered mask therefore leaves it unchanged.

    :param mask: one-dimensional boolean raw detector flags.
    :param min_frames: positive number of consecutive flags needed to keep a run.
    :return: one-dimensional boolean mask of duration-filtered whole runs.
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 1 or mask.dtype != np.bool_:
        raise ValueError('mask must be a one-dimensional boolean array')
    if isinstance(min_frames, bool) or not isinstance(min_frames, (int, np.integer)) or min_frames < 1:
        raise ValueError(f'min_frames must be a positive integer, got {min_frames!r}')

    believed = np.zeros_like(mask)
    for start, end in true_runs(mask):
        if end - start >= min_frames:
            believed[start:end] = True
    return believed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _read_homography_rows(csv_path: Path | None, video_id: str) -> list[dict] | None:
    """Read the per-segment homography rows for one video, or None if unavailable."""
    if csv_path is None or not csv_path.exists():
        return None
    with csv_path.open(newline='', encoding='utf-8') as handle:
        rows = [row for row in csv.DictReader(handle) if row.get('video_id') == video_id]
    return rows or None


def _read_rally_spans(csv_path: Path, video_id: str) -> list[tuple[int, int]]:
    """Read `(start_frame, end_frame)` rally spans for one video from the spans CSV."""
    if not csv_path.exists():
        return []
    with csv_path.open(newline='', encoding='utf-8') as handle:
        return [
            (int(row['start_frame']), int(row['end_frame']))
            for row in csv.DictReader(handle)
            if row.get('video_id') == video_id
        ]


def _cli_non_evidence(track: np.ndarray | None) -> np.ndarray | None:
    """Grade a CLI track and return frames rejected by the resolved policy."""
    if track is None:
        return None
    codes, _guard_info = grade_track(track)
    log.info('inpaint grade counts: %s', code_counts(codes))
    return np.isin(codes, tuple(sorted(BaseAnnotatorConfig().rejected_grades)))


def main() -> None:
    parser = argparse.ArgumentParser(description='Replay masking: replay/off-rally mask for one video.')
    parser.add_argument('--video-id', required=True)
    parser.add_argument('--shuttle', type=Path, default=None,
                        help='<video_id>.npy (t, 3) shuttle track')
    parser.add_argument('--court-mask', type=Path, default=None,
                        help='<video_id> court-present (frames,) bool npy')
    parser.add_argument('--homography-csv', type=Path, default=None,
                        help='Per-segment homography CSV (video_id, start_frame, end_frame, corners)')
    parser.add_argument('--rally-spans', type=Path, default=RALLY_SPANS_CSV)
    parser.add_argument('--out-dir', type=Path, default=MASKS_DIR)
    parser.add_argument('--fps', type=float, required=True)
    parser.add_argument('--no-replay-mask', action='store_true',
                        help='skip detector computation and write an all-False mask')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    log.info('config: no_replay_mask=%s', args.no_replay_mask)

    track = np.load(args.shuttle) if args.shuttle and args.shuttle.exists() else None
    court_present = np.load(args.court_mask) if args.court_mask and args.court_mask.exists() else None
    homography_rows = _read_homography_rows(args.homography_csv, args.video_id)
    rally_spans = _read_rally_spans(args.rally_spans, args.video_id)

    # Frame count: court mask and shuttle track both span the whole video; use
    # whichever is present. Without either there is nothing to size the mask to.
    if court_present is not None:
        n_frames = len(court_present)
    elif track is not None:
        n_frames = len(track)
    else:
        raise FileNotFoundError('need a court-present mask or a shuttle track to size the frame mask')

    if args.no_replay_mask:
        mask = np.zeros(n_frames, dtype=bool)
    else:
        mask = combine_mask(
            court_present, homography_rows, track, rally_spans, n_frames, args.fps,
            non_evidence=_cli_non_evidence(track),
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f'{args.video_id}_replay.npy'
    np.save(out_path, mask)
    log.info('%s: %d/%d frames masked -> %s', args.video_id, int(mask.sum()), n_frames, out_path)


if __name__ == '__main__':
    main()
