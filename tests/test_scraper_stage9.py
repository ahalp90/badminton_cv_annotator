"""Stage 9 replay-mask tests: each signal in isolation, the union, and missing inputs.

Synthetic inputs are shaped so each signal has one obvious firing region and one
obvious quiet region.
"""
import numpy as np

from src.scraper.config import (
    COURT_ABSENT_WINDOW,
    PERSPECTIVE_SHIFT_THRESHOLD,
    SLOWMO_SPEED_FRAC,
)
from src.scraper.stage9_replay_mask import (
    HOMOGRAPHY_CORNER_COLS,
    combine_mask,
    court_absence_signal,
    perspective_shift_signal,
    velocity_drop_signal,
)

# Reference court corners (interleaved [corner, xy]); bbox diagonal 500.
REFERENCE_CORNERS = [100.0, 100.0, 500.0, 100.0, 100.0, 400.0, 500.0, 400.0]


def _homography_row(video_id: str, start: int, end: int, corners: list[float]) -> dict:
    row = {'video_id': video_id, 'start_frame': str(start), 'end_frame': str(end)}
    row.update({col: str(value) for col, value in zip(HOMOGRAPHY_CORNER_COLS, corners)})
    return row


def test_court_absence_fires_on_sustained_gap_not_blips():
    n_frames = 60
    court_present = np.ones(n_frames, dtype=bool)
    long_gap = slice(10, 10 + COURT_ABSENT_WINDOW + 5)   # sustained absence, fires
    short_blip = slice(45, 48)                            # 3 frames, below the window
    court_present[long_gap] = False
    court_present[short_blip] = False

    mask = court_absence_signal(court_present, n_frames)
    assert mask[long_gap].all()
    assert not mask[short_blip].any()
    assert not mask[:10].any()


def test_perspective_fires_only_on_deviant_segment():
    shifted = [value + (200.0 if i % 2 == 0 else 0.0) for i, value in enumerate(REFERENCE_CORNERS)]
    rows = [
        _homography_row('v', 0, 100, REFERENCE_CORNERS),   # dominant view
        _homography_row('v', 100, 200, REFERENCE_CORNERS),  # dominant view
        _homography_row('v', 200, 220, shifted),            # +200 in x on every corner -> replay angle
    ]
    n_frames = 220
    mask = perspective_shift_signal(rows, n_frames)

    # 200 / 500 = 0.4 displacement, well over the threshold.
    assert (200.0 / 500.0) > PERSPECTIVE_SHIFT_THRESHOLD
    assert not mask[:200].any()
    assert mask[200:220].all()


def _speed_track(step: float, n_frames: int) -> np.ndarray:
    """A visible track whose per-frame speed is `step` (y toggles by `step`)."""
    ys = 0.4 + step * (np.arange(n_frames) % 2)
    return np.column_stack([np.full(n_frames, 0.5), ys, np.ones(n_frames)])


def test_velocity_drop_fires_on_slow_span_not_normal_play():
    n_frames = 90
    normal_step, slow_step = 0.1, 0.015
    track = _speed_track(normal_step, n_frames)
    track[40:80, 1] = 0.4 + slow_step * (np.arange(40, 80) % 2)  # slow replay span
    rally_spans = [(0, 31)]                                       # normal play defines the norm

    mask = velocity_drop_signal(track, rally_spans, n_frames)
    # slow_step (0.015) < SLOWMO_SPEED_FRAC * normal_step (0.03); normal play stays above.
    assert slow_step < SLOWMO_SPEED_FRAC * normal_step
    assert mask[45:78].all()
    assert not mask[3:28].any()


def test_velocity_drop_ignores_genuine_rest():
    """Rest (below REST_SPEED) is the between-rallies state, not slow motion.

    Post-rally commentary starts during rest; if rest fired this signal, stage 11
    would hold every post-rally chunk out of pairing.
    """
    n_frames = 90
    track = _speed_track(0.1, n_frames)
    track[40:80, 1] = 0.4                                 # shuttle at rest: zero speed, visible
    rally_spans = [(0, 31)]

    mask = velocity_drop_signal(track, rally_spans, n_frames)
    assert not mask[45:78].any()
    assert not mask[3:28].any()


def test_union_combines_signals():
    n_frames = 220
    court_present = np.ones(n_frames, dtype=bool)
    court_present[10:30] = False                         # court-absence region
    rows = [
        _homography_row('v', 0, 200, REFERENCE_CORNERS),
        _homography_row('v', 200, 220,
                        [value + (200.0 if i % 2 == 0 else 0.0)
                         for i, value in enumerate(REFERENCE_CORNERS)]),
    ]
    track = _speed_track(0.1, n_frames)
    rally_spans = [(30, 200)]

    mask = combine_mask(court_present, rows, track, rally_spans, n_frames)
    assert mask[15:25].all()                             # court absence
    assert mask[205:215].all()                           # perspective shift


def test_missing_inputs_contribute_all_false():
    n_frames = 50
    assert not court_absence_signal(None, n_frames).any()
    assert not perspective_shift_signal(None, n_frames).any()
    assert not velocity_drop_signal(None, [(0, 10)], n_frames).any()
    assert not velocity_drop_signal(_speed_track(0.1, n_frames), None, n_frames).any()
    assert not combine_mask(None, None, None, None, n_frames).any()
