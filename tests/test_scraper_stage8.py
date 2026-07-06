"""Stage 8 rally-segmentation tests: synthetic tracks, CPU-only, fast.

Tracks are built to make the truth obvious: a rest span, a zig-zag rally with a
known number of clean velocity reversals, then a long rest. Reversal spacing is
kept above SMOOTH_WINDOW so the de-dup does not collapse distinct contacts, and
the step size keeps the smoothed reversal speed above MIN_CONTACT_SPEED.
"""
import numpy as np
import pytest

from src.scraper.config import (
    END_REST_FRAMES,
    MIN_CONTACT_SPEED,
    SMOOTH_WINDOW,
    START_SPEED,
)
from src.scraper.stage8_rally_segmentation import (
    compute_speed,
    contact_proximity_ok,
    detect_contacts,
    segment_video,
)

# A per-frame step that keeps raw speed above START_SPEED and, once smoothed,
# keeps the reversal-junction speed above MIN_CONTACT_SPEED.
RALLY_STEP = 0.14
REST_PRE = 45
REST_POST = 60
REST_Y = 0.01


def _bounce_positions() -> tuple[np.ndarray, list[int]]:
    """A three-reversal bounce path and its apex indices within the path."""
    up = np.round(np.arange(REST_Y, 0.99 + RALLY_STEP / 2, RALLY_STEP), 4)  # 0.01..0.99
    down = up[::-1]
    # lo -> hi (apex) -> lo (apex) -> hi (apex) -> lo, dropping shared endpoints.
    path = np.concatenate([up, down[1:], up[1:], down[1:]])
    apex_local = [len(up) - 1, len(up) - 1 + len(down[1:]), len(up) - 1 + len(down[1:]) + len(up[1:])]
    return path, apex_local


def _build_rally_track() -> tuple[np.ndarray, int, int, list[int]]:
    """Rest + three-reversal rally + long rest.

    :return: (track, rally_start_frame, rally_end_frame, contact_frames).
    """
    path, apex_local = _bounce_positions()
    rally_y = path[1:]                                    # drop leading REST_Y (seam with the rest)
    ys = np.concatenate([np.full(REST_PRE, REST_Y), rally_y, np.full(REST_POST, REST_Y)])
    xs = np.full_like(ys, 0.5)
    vis = np.ones_like(ys)
    track = np.column_stack([xs, ys, vis])

    rally_start = REST_PRE
    rally_end = REST_PRE + len(rally_y)
    contact_frames = [REST_PRE + (local - 1) for local in apex_local]  # -1: rally_y dropped path[0]
    return track, rally_start, rally_end, contact_frames


def test_compute_speed_and_visibility_nan():
    track = np.array([
        [0.0, 0.0, 1.0],
        [0.3, 0.4, 1.0],   # step (0.3, 0.4) -> speed 0.5
        [0.3, 0.4, 0.0],   # invisible: any step touching this frame is NaN
        [0.6, 0.8, 1.0],
    ])
    speed = compute_speed(track)
    assert np.isnan(speed[0])              # frame 0 has no predecessor
    assert speed[1] == pytest.approx(0.5)
    assert np.isnan(speed[2])              # step into the invisible frame
    assert np.isnan(speed[3])              # step out of the invisible frame


def test_single_rally_span_and_three_contacts():
    track, rally_start, rally_end, truth_contacts = _build_rally_track()
    spans, contacts = segment_video(track)

    assert len(spans) == 1
    start, end = spans[0]
    assert abs(start - rally_start) <= 4
    assert abs(end - rally_end) <= 4

    contact_frames = sorted(frame for _, frame, _ in contacts)
    assert len(contact_frames) == 3
    for detected, truth in zip(contact_frames, truth_contacts):
        assert abs(detected - truth) <= 3


def test_static_track_yields_no_rally():
    track = np.column_stack([
        np.full(150, 0.5), np.full(150, 0.5), np.ones(150),
    ])
    spans, contacts = segment_video(track)
    assert spans == []
    assert contacts == []


def test_invisible_moving_track_reads_as_rest():
    # Fast motion but never tracked: every step is NaN, so nothing reads as fast
    # and the window is mostly untracked -> rest. No rally is found.
    ys = np.tile([0.1, 0.9], 75)
    track = np.column_stack([np.full(150, 0.5), ys, np.zeros(150)])
    spans, _ = segment_video(track)
    assert spans == []


def _triangle_track(step: float) -> np.ndarray:
    """A single up-down triangle in y, all visible, for direct contact tests."""
    base = np.array([0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0])
    ys = REST_Y + step * base
    return np.column_stack([np.full(len(ys), 0.5), ys, np.ones(len(ys))])


def test_contact_detected_on_fast_reversal():
    track = _triangle_track(RALLY_STEP)          # step >> MIN_CONTACT_SPEED after smoothing
    contacts = detect_contacts(track, 0, len(track))
    assert len(contacts) == 1
    assert abs(contacts[0] - 7) <= 2             # apex sits at index 7


def test_contact_suppressed_below_min_contact_speed():
    slow_step = 0.01                             # < MIN_CONTACT_SPEED (0.02), so the gate rejects it
    assert slow_step < MIN_CONTACT_SPEED
    assert slow_step < START_SPEED
    track = _triangle_track(slow_step)
    assert detect_contacts(track, 0, len(track)) == []


def test_proximity_ok_true_false_blank():
    track, _, _, contacts = _build_rally_track()
    contact_frame = contacts[0]
    shuttle_xy = track[contact_frame, :2]

    # Unmeasured: no positions supplied -> blank (None), never True.
    assert contact_proximity_ok(track, None, contact_frame) is None

    # Near player (slot 0 on top of the shuttle) -> True.
    near = np.full((len(track), 2, 2), 5.0)
    near[contact_frame, 0] = shuttle_xy
    assert contact_proximity_ok(track, near, contact_frame) is True

    # Both players far away -> False (measured, unconfirmed).
    far = np.full((len(track), 2, 2), 5.0)
    assert contact_proximity_ok(track, far, contact_frame) is False


def test_segment_video_proximity_paths():
    track, _, _, _ = _build_rally_track()
    # No positions: every contact carries a blank guardrail.
    _, contacts_blank = segment_video(track, positions=None)
    assert contacts_blank
    assert all(proximity_ok is None for _, _, proximity_ok in contacts_blank)

    # Player sitting on the shuttle at every frame: every contact reads True.
    positions = np.repeat(track[:, None, :2], 2, axis=1)  # (t, 2, 2) both slots on the shuttle
    _, contacts_near = segment_video(track, positions=positions)
    assert all(proximity_ok is True for _, _, proximity_ok in contacts_near)


def test_end_rest_frames_constant_used():
    # A short rest between two bursts must not split into two rallies unless it
    # reaches END_REST_FRAMES. Sanity that the constant is the gate.
    assert END_REST_FRAMES > SMOOTH_WINDOW
