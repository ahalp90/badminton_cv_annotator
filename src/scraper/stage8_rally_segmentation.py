"""Stage 8: rally segmentation and contact detection (scraper_spec.md section 6).

Trajectory rules over a whole-video TrackNetV3 shuttle track, the `(t, 3)`
`[x_norm, y_norm, visibility]` npy that `shuttle_extractor.py:244-249` writes
(x, y already normalised to [0, 1] by video resolution, visibility passed
through). Speed everywhere below is per-frame L2 displacement of `(x, y)` on
frames where visibility is 1.

Three primitives (`compute_speed`, `true_runs`, `rolling_nanmedian`) are public
because stage 9 reuses them: its slow-motion signal is defined against this
stage's per-frame speed, so re-deriving it there would be a second source of
truth. All per-frame arrays here share one frame-index space `[0, t)`; that
invariant is what lets rally spans, contacts and masks line up downstream.

Run as `python -m scraper.stage8_rally_segmentation --shuttle-dir ...` with
PYTHONPATH=src.
"""
import argparse
import csv
import logging
import warnings
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .config import (
    CONTACT_FRAMES_CSV,
    END_REST_FRAMES,
    MIN_CONTACT_SPEED,
    MIN_DIR_CHANGE_DEG,
    PROXIMITY_MAX,
    RALLY_SPANS_CSV,
    REST_SPEED,
    REST_WINDOW,
    SMOOTH_WINDOW,
    START_MIN_FRAMES,
    START_SPEED,
)

log = logging.getLogger(__name__)

# Fraction of a window that must be tracked (visibility 1) for the window to
# read as "seeing the shuttle". Below this the window is mostly untracked and
# counts as rest (spec s6 "visibility mostly 0 across the window"). Not in
# config: it is the numeric reading of "mostly", not a swept rule threshold.
VISIBILITY_REST_FRAC = 0.5


# ---------------------------------------------------------------------------
# Shared primitives (stage 9 imports these)
# ---------------------------------------------------------------------------
def compute_speed(track: np.ndarray) -> np.ndarray:
    """Per-frame shuttle speed, NaN where the step is not fully visible.

    Speed at frame i is the L2 displacement of `(x, y)` from frame i-1 to i.
    Frame 0 has no predecessor and both endpoint frames must have visibility 1,
    else the step is unmeasured and reads NaN (so nan-aware stats skip it).

    :param track: `(t, 3)` `[x_norm, y_norm, visibility]` whole-video track.
    :return: `(t,)` speed in norm-units/frame; NaN on frame 0 and on any step
        touching a non-visible frame.
    """
    xy = track[:, :2]                                  # (t, 2) normalised position
    visibility = track[:, 2]                           # (t,)
    step = np.diff(xy, axis=0)                          # (t-1, 2) frame i-1 -> i
    step_speed = np.linalg.norm(step, axis=1)          # (t-1,)
    both_visible = (visibility[:-1] == 1) & (visibility[1:] == 1)  # (t-1,) both ends of the step

    speed = np.full(len(track), np.nan)                # (t,) frame-indexed; frame 0 stays NaN
    speed[1:] = np.where(both_visible, step_speed, np.nan)
    return speed


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of True in a boolean mask, as half-open `[start, end)` ranges.

    Vectorised via edge detection on the zero-padded int mask: +1 marks a run
    start, -1 marks one-past a run end. Shared with stage 9's court-absence
    signal, which masks whole absent runs.

    :param mask: `(t,)` boolean.
    :return: list of `(start, end)` with `mask[start:end]` all True.
    """
    padded = np.concatenate([[0], mask.astype(np.int8), [0]])  # sentinels force edges at the ends
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def rolling_nanmedian(values: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling median that ignores NaN, one value per input frame.

    Pads both ends with NaN so every frame gets a full-width window and the
    output keeps length t; nanmedian drops the pad and any NaN steps. Shared
    with stage 9's slow-motion signal.

    :param values: `(t,)` values, may contain NaN.
    :param window: window width in frames.
    :return: `(t,)` centred rolling median; NaN only where a whole window is NaN.
    """
    left = window // 2
    right = window - 1 - left
    padded = np.concatenate([np.full(left, np.nan), values, np.full(right, np.nan)])
    windows = sliding_window_view(padded, window)      # (t, window)
    with warnings.catch_warnings():
        # An all-NaN window (e.g. a fully untracked span) is expected and yields
        # NaN by design; silence the RuntimeWarning rather than let it spam logs.
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.nanmedian(windows, axis=1)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling mean with a shrinking window at the edges (no NaN handling).

    Used only for the visibility fraction, which is a clean 0/1 array.

    :param values: `(t,)` values, no NaN.
    :param window: window width in frames.
    :return: `(t,)` centred mean; edge frames average their partial window.
    """
    kernel = np.ones(window)
    counts = np.convolve(np.ones_like(values), kernel, mode='same')  # samples per position
    sums = np.convolve(values, kernel, mode='same')
    return sums / counts


# ---------------------------------------------------------------------------
# Rally spans
# ---------------------------------------------------------------------------
def _rest_mask(speed: np.ndarray, track: np.ndarray) -> np.ndarray:
    """Per-frame rest flag: slow OR mostly untracked across the window (spec s6).

    :param speed: `(t,)` per-frame speed (NaN on non-visible steps).
    :param track: `(t, 3)` track, for the visibility column.
    :return: `(t,)` bool, True where the frame reads as rest.
    """
    speed_median = rolling_nanmedian(speed, REST_WINDOW)     # (t,)
    slow = speed_median < REST_SPEED                         # NaN windows read not-slow here...
    visible = (track[:, 2] == 1).astype(float)               # (t,) 1.0 where tracked
    frac_visible = _rolling_mean(visible, REST_WINDOW)       # (t,) fraction tracked in window
    mostly_untracked = frac_visible < VISIBILITY_REST_FRAC   # ...and the OR below catches them
    return slow | mostly_untracked


def _find_rally_spans(speed: np.ndarray, at_rest: np.ndarray) -> list[tuple[int, int]]:
    """Segment the video into rally spans between extended rest.

    A long rest (a rest run >= END_REST_FRAMES) separates rallies. Inside each
    stretch of non-long-rest frames, the rally starts at the first sustained
    burst of fast frames (START_MIN_FRAMES consecutive above START_SPEED, the
    acceleration-from-rest signature) and ends where the following long rest
    begins. A stretch with no such burst (e.g. a brief tracker twitch) yields
    no rally.

    :param speed: `(t,)` per-frame speed (NaN on non-visible steps).
    :param at_rest: `(t,)` per-frame rest flag.
    :return: list of `(start_frame, end_frame)` half-open rally spans.
    """
    fast = np.nan_to_num(speed, nan=0.0) > START_SPEED       # (t,) NaN steps are not fast

    long_rest = np.zeros(len(speed), dtype=bool)             # (t,) frames inside an extended rest
    for start, end in true_runs(at_rest):
        if end - start >= END_REST_FRAMES:
            long_rest[start:end] = True

    fast_runs = [(start, end) for start, end in true_runs(fast) if end - start >= START_MIN_FRAMES]

    spans: list[tuple[int, int]] = []
    for region_start, region_end in true_runs(~long_rest):
        # The first qualifying fast run that opens inside this active region is
        # the acceleration out of the preceding rest; the region's end is the
        # onset of the next extended rest (or the video end).
        burst_start = next(
            (start for start, _ in fast_runs if region_start <= start < region_end),
            None,
        )
        if burst_start is None:
            continue
        spans.append((int(burst_start), int(region_end)))
    return spans


# ---------------------------------------------------------------------------
# Contacts
# ---------------------------------------------------------------------------
def detect_contacts(track: np.ndarray, start: int, end: int) -> list[int]:
    """Contact frames inside one rally span, from smoothed-velocity reversals.

    Smooth `(x, y)` over SMOOTH_WINDOW to survive TrackNetV3 jitter, take the
    per-frame velocity vectors, and flag a junction where the incoming and
    outgoing velocity turn by more than MIN_DIR_CHANGE_DEG with both segment
    speeds above MIN_CONTACT_SPEED and the three frames around the reversal all
    visible. A real contact often trips several adjacent junctions once
    smoothed; de-dup keeps the sharpest-angle junction and drops any other
    within SMOOTH_WINDOW frames of it (the spec is silent on de-dup: the true
    contact vertex carries the largest reversal, so sharpest-angle-wins).

    :param track: `(t, 3)` whole-video track.
    :param start: rally span start frame (inclusive).
    :param end: rally span end frame (exclusive).
    :return: contact frames in whole-video frame indices, ascending.
    """
    span = track[start:end]                              # (n, 3) rally-local view
    if len(span) < SMOOTH_WINDOW + 2:
        return []                                        # too short to smooth and difference twice

    smooth_x = _rolling_mean(span[:, 0], SMOOTH_WINDOW)  # (n,)
    smooth_y = _rolling_mean(span[:, 1], SMOOTH_WINDOW)  # (n,)
    smoothed = np.column_stack([smooth_x, smooth_y])     # (n, 2)
    visibility = span[:, 2]                              # (n,)

    velocity = np.diff(smoothed, axis=0)                 # (n-1, 2) segment j spans local frames j -> j+1
    v_in = velocity[:-1]                                 # (n-2, 2) segment into junction k+1
    v_out = velocity[1:]                                 # (n-2, 2) segment out of junction k+1
    speed_in = np.linalg.norm(v_in, axis=1)              # (n-2,)
    speed_out = np.linalg.norm(v_out, axis=1)            # (n-2,)

    # Angle between incoming and outgoing velocity at each interior junction.
    # Guard the zero-speed denominator so a stalled segment reads as no turn.
    denom = speed_in * speed_out
    safe = denom > 0
    cos_angle = np.ones(len(denom))                      # (n-2,) default 1.0 -> 0 deg where unsafe
    cos_angle[safe] = np.sum(v_in[safe] * v_out[safe], axis=1) / denom[safe]
    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))  # (n-2,)

    # All per-junction arrays share index k in [0, n-3]; junction k sits at
    # local frame k+1, touching local frames k, k+1, k+2.
    around_visible = (visibility[:-2] == 1) & (visibility[1:-1] == 1) & (visibility[2:] == 1)
    sharp_turn = angle_deg > MIN_DIR_CHANGE_DEG
    fast_enough = (speed_in > MIN_CONTACT_SPEED) & (speed_out > MIN_CONTACT_SPEED)
    is_contact = sharp_turn & fast_enough & around_visible

    candidate_local = np.flatnonzero(is_contact) + 1     # local frame of each candidate
    candidate_angle = angle_deg[is_contact]

    kept: list[int] = []
    for idx in np.argsort(-candidate_angle):             # sharpest angle first
        frame = int(candidate_local[idx])
        if all(abs(frame - other) >= SMOOTH_WINDOW for other in kept):
            kept.append(frame)
    kept.sort()
    return [start + frame for frame in kept]


def contact_proximity_ok(
    track: np.ndarray, positions: np.ndarray | None, contact_frame: int
) -> bool | None:
    """Guardrail: does a tracked player sit near the shuttle at the contact frame?

    Never filters a contact; it annotates one. When no positions were supplied
    the check is unmeasured, which returns None (serialised blank downstream):
    a guardrail with no evidence must not read as a pass.

    :param track: `(t, 3)` whole-video track.
    :param positions: `(t, 2, 2)` `[slot, xy]` court positions, or None.
    :param contact_frame: whole-video frame index of the contact.
    :return: True/False when measured, None when no positions were supplied.
    """
    if positions is None:
        return None
    shuttle_xy = track[contact_frame, :2]                    # (2,)
    player_xy = positions[contact_frame]                     # (2, 2) [slot, xy]
    distances = np.linalg.norm(player_xy - shuttle_xy, axis=1)  # (2,) per slot
    if np.all(np.isnan(distances)):
        # Positions exist but both slots failed this frame: measured, unconfirmed.
        return False
    return bool(np.nanmin(distances) <= PROXIMITY_MAX)


def segment_video(
    track: np.ndarray, positions: np.ndarray | None = None
) -> tuple[list[tuple[int, int]], list[tuple[int, int, bool | None]]]:
    """Full stage-8 pass over one video's shuttle track.

    :param track: `(t, 3)` whole-video track.
    :param positions: optional `(t, 2, 2)` court positions for the proximity guardrail.
    :return: `(spans, contacts)` where spans is `[(start_frame, end_frame), ...]`
        (rally_id is the list index) and contacts is
        `[(rally_id, contact_frame, proximity_ok), ...]`.
    """
    speed = compute_speed(track)
    at_rest = _rest_mask(speed, track)
    spans = _find_rally_spans(speed, at_rest)

    contacts: list[tuple[int, int, bool | None]] = []
    for rally_id, (start, end) in enumerate(spans):
        for contact_frame in detect_contacts(track, start, end):
            proximity_ok = contact_proximity_ok(track, positions, contact_frame)
            contacts.append((rally_id, contact_frame, proximity_ok))
    return spans, contacts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
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


def main() -> None:
    parser = argparse.ArgumentParser(description='Stage 8: rally spans and contacts from shuttle tracks.')
    parser.add_argument('--shuttle-dir', type=Path, required=True,
                        help='Directory of <video_id>.npy (t, 3) shuttle tracks')
    parser.add_argument('--pos-dir', type=Path, default=None,
                        help='Optional directory of <video_id>_pos.npy court positions')
    parser.add_argument('--rally-spans-csv', type=Path, default=RALLY_SPANS_CSV)
    parser.add_argument('--contact-frames-csv', type=Path, default=CONTACT_FRAMES_CSV)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    if not args.shuttle_dir.is_dir():
        raise FileNotFoundError(f'shuttle dir not found: {args.shuttle_dir}')

    args.rally_spans_csv.parent.mkdir(parents=True, exist_ok=True)
    args.contact_frames_csv.parent.mkdir(parents=True, exist_ok=True)

    span_rows: list[tuple[str, int, int, int]] = []
    contact_rows: list[tuple[str, int, int, str]] = []
    for track_path in sorted(args.shuttle_dir.glob('*.npy')):
        video_id = track_path.stem
        try:
            track = np.load(track_path)
            positions = _load_positions(args.pos_dir, video_id)
            spans, contacts = segment_video(track, positions)
        except Exception as exc:  # log-and-skip per video: one bad track must not sink the batch
            log.warning('skipping %s: %s', video_id, exc)
            continue
        for rally_id, (start, end) in enumerate(spans):
            span_rows.append((video_id, rally_id, start, end))
        for rally_id, contact_frame, proximity_ok in contacts:
            contact_rows.append((video_id, rally_id, contact_frame, _format_bool(proximity_ok)))
        log.info('%s: %d rallies, %d contacts', video_id, len(spans), len(contacts))

    with args.rally_spans_csv.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['video_id', 'rally_id', 'start_frame', 'end_frame'])
        writer.writerows(span_rows)
    with args.contact_frames_csv.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['video_id', 'rally_id', 'contact_frame', 'proximity_ok'])
        writer.writerows(contact_rows)
    log.info('wrote %d rally spans, %d contacts', len(span_rows), len(contact_rows))


if __name__ == '__main__':
    main()
