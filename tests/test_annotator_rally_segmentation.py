"""Rally-segmentation tests using fast, CPU-only synthetic tracks.

Tracks are built to make the truth obvious: a rest span, a zig-zag rally with a
known number of clean velocity reversals, then a long rest. The contact tests
pin the measured impulse rule and its largest-impulse de-duplication.
"""
import numpy as np
import pytest

from annotator.calibration.fixtures import SSET_01
from annotator.config import (
    END_REST_FRAMES,
    SHIPPED_THRESHOLDS,
    SMOOTH_WINDOW,
)
from annotator.rally_segmentation import (
    CourtGeo,
    ServeSetupInputs,
    ServeStartClose,
    ServeStartMode,
    ServeStartOptions,
    SpanOpen,
    apply_replay_mask,
    compute_speed,
    contact_proximity_ok,
    court_scale_slots,
    detect_contact_flags,
    segment_video,
    span_impulses,
    suppress_contact_flags,
    wrist_contact_near,
)
from annotator.rally.serve import _serve_distance_ratio_passes
from annotator.rally.spans import (
    _find_rally_spans,
    _find_rally_spans_span_open,
    _last_rest_close,
    _serve_start_find_rally_spans,
)
from annotator.rally.trajectory import _nan_rolling_mean, _rolling_mean
from annotator.fps_constants import scale_for_fps
from annotator.types import SmoothingMode

SSET_01_COURT_GEO = CourtGeo(*SSET_01.court_geo)

# A per-frame step that keeps raw speed above START_SPEED.
RALLY_STEP = 0.14
REST_PRE = 45
REST_POST = 100  # past END_REST_FRAMES, so the trailing rest closes the rally region
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
    track, rally_start, rally_end, _truth_contacts = _build_rally_track()
    spans, contacts = segment_video(track)

    assert len(spans) == 1
    start, end = spans[0]
    assert abs(start - rally_start) <= 4
    assert abs(end - rally_end) <= 4

    contact_frames = sorted(frame for _, frame, *_ in contacts)
    # The zigzag's symmetric reversals produce exact impulse ties; the stable dedup
    # keeps the earlier frame (57, was 58 under the unstable sort).
    assert contact_frames == [46, 50, 57, 64, 71]


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


def test_ignore_invisible_smoothing_uses_even_fps_scaled_window() -> None:
    thresholds = SHIPPED_THRESHOLDS._replace(smooth_window=scale_for_fps(60.0).smooth_window)
    assert thresholds.smooth_window == 6
    track = np.column_stack([
        [1.0, 99.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0],
        np.zeros(8),
        [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ])

    impulses = span_impulses(
        track, 0, len(track), thresholds, smoothing_mode=SmoothingMode.IGNORE_INVISIBLE,
    )

    # Width-six smoothing gives [2, 3, 4, 5, 7, 8, 9, 10], so the impulse is
    # the absolute velocity change [0, 0, 1, 1, 0, 0].
    np.testing.assert_array_equal(impulses, [0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    assert track[1, 0] == 99.0


def test_ignore_invisible_smoothing_marks_all_invisible_span_unmeasurable() -> None:
    thresholds = SHIPPED_THRESHOLDS._replace(smooth_window=6)
    track = np.column_stack([np.arange(8, dtype=float), np.zeros(8), np.zeros(8)])

    impulses = span_impulses(
        track, 0, len(track), thresholds, smoothing_mode=SmoothingMode.IGNORE_INVISIBLE,
    )

    assert np.isnan(impulses).all()


def test_nan_rolling_mean_matches_stock_mean_on_finite_input() -> None:
    values = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0])

    assert np.array_equal(_nan_rolling_mean(values, 6), _rolling_mean(values, 6))


def _triangle_track(step: float) -> np.ndarray:
    """A single up-down triangle in y, all visible, for direct contact tests."""
    base = np.array([0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0])
    ys = REST_Y + step * base
    return np.column_stack([np.full(len(ys), 0.5), ys, np.ones(len(ys))])


def test_contact_detected_on_fast_reversal():
    track = _triangle_track(RALLY_STEP)
    contacts = [frame for frame, _impulse in detect_contact_flags(track, 0, len(track))]
    assert contacts == [1, 7, 13]


def test_contact_suppression_ranks_impulse_then_frame():
    # Radius pinned explicitly: this test is about the ranking rule, and the module default
    # scales with the fps table (8 at the 25 fps surface under base-30, where this fixture
    # would suppress nothing).
    flags = [(10, 1.0), (18, 2.0), (27, 2.0)]
    assert suppress_contact_flags(flags, radius=9) == [18, 27]
    assert detect_contact_flags(_triangle_track(RALLY_STEP), 0, 15)


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
    assert all(contact.proximity_ok is None for contact in contacts_blank)

    # Player sitting on the shuttle at every frame: every contact reads True.
    positions = np.repeat(track[:, None, :2], 2, axis=1)  # (t, 2, 2) both slots on the shuttle
    _, contacts_near = segment_video(track, positions=positions)
    assert all(contact.proximity_ok is True for contact in contacts_near)


# ---------------------------------------------------------------------------
# Contact wrist check: court_scale_slots, wrist_contact_near, segment_video
# ---------------------------------------------------------------------------
def test_court_scale_slots_keeps_identities_on_tied_scores():
    # Two detections share a score and only the second sits inside the court bounds.
    # The slot identity remains exact even when detections tie on score.
    court_geo = CourtGeo(x_range=(0.0, 100.0), y_range=(0.0, 100.0), net_band=(50.0, 50.0))
    frame_bboxes = np.array([
        [500.0, 500.0, 520.0, 540.0],  # slot 0: tied score, foot point far off court
        [40.0, 79.0, 60.0, 80.0],  # slot 1: in court, arbitrary box height
        [np.nan, np.nan, np.nan, np.nan],  # padding slot
    ])
    frame_scores = np.array([0.9, 0.9, np.nan])
    assert court_scale_slots(frame_bboxes, frame_scores, court_geo).tolist() == [1]


def test_wrist_contact_near_verdicts():
    # None distances (gate never ran): unmeasured -> None, never a pass.
    assert wrist_contact_near(None, 0) is None

    # Below / at / above the sticky body-unit gate, plus NaN.
    sticky_distances = np.array([1.39, 1.4, 1.41, np.nan])
    assert wrist_contact_near(sticky_distances, 0) is True
    assert wrist_contact_near(sticky_distances, 1) is True   # equality passes (<=)
    assert wrist_contact_near(sticky_distances, 2) is False
    assert wrist_contact_near(sticky_distances, 3) is False


def test_segment_video_wrist_near_paths():
    track, _, _, _ = _build_rally_track()

    # No gate inputs at all: the gate never ran, both verdicts are blank (None), and no
    # suppression happens, so every raw candidate stands (recall-first).
    _, contacts_blank = segment_video(track)
    assert contacts_blank
    assert all(contact.wrist_near is None and contact.suppressed is None for contact in contacts_blank)
    assert contacts_blank[0][:4] == (
        contacts_blank[0].rally_id, contacts_blank[0].contact_frame,
        contacts_blank[0].proximity_ok, contacts_blank[0].wrist_near,
    )

    # A wrist on the shuttle at every frame (distance 0): every candidate passes the gate, and
    # suppression records only its losers.
    near = np.zeros(len(track))
    _, contacts_near = segment_video(track, sticky_distances=near)
    assert [contact.wrist_near for contact in contacts_near] == [True] * len(contacts_near)
    assert [contact.suppressed for contact in contacts_near] == [True, False, True, False, True]
    filtered = [
        contact for contact in contacts_near
        if contact.wrist_near is not False and contact.suppressed is not True
    ]
    assert [contact.contact_frame for contact in filtered] == [
        contact.contact_frame for contact in contacts_near if not contact.suppressed
    ]

    # A wrist far from the shuttle everywhere: every contact drops (False).
    far = np.full(len(track), 2.0)
    _, contacts_far = segment_video(track, sticky_distances=far)
    assert all(contact.wrist_near is False and contact.suppressed is False for contact in contacts_far)


@pytest.mark.parametrize('failure_distance', [np.nan, np.inf])
def test_sticky_gate_failure_rows_fail_closed(failure_distance):
    """A no-evidence distance row (NaN inside spans, +inf outside) fails the gate closed.

    Uses the cached-distances injection surface that replaced the old internal
    monkeypatch: the caller supplies the full-length series directly.
    """
    track, _, _, _ = _build_rally_track()
    _, contacts = segment_video(
        track, sticky_distances=np.full(len(track), failure_distance),
    )
    assert contacts
    assert all(contact.wrist_near is False and contact.suppressed is False for contact in contacts)


def test_end_rest_frames_constant_used():
    # A short rest between two bursts must not split into two rallies unless it
    # reaches END_REST_FRAMES. Sanity that the constant is the gate.
    assert END_REST_FRAMES > SMOOTH_WINDOW


# ---------------------------------------------------------------------------
# Thresholds option: None reads globals; a preset changes behaviour
# ---------------------------------------------------------------------------
# END_REST_FRAMES patched into a preset for the synthetic serve-start / split tracks below,
# whose bursts sit in one active region (no rest run reaches this bound).
_SERVE_THRESHOLDS = SHIPPED_THRESHOLDS._replace(end_rest_frames=40)


def _burst_track() -> np.ndarray:
    """Rest, a visible burst at ~0.025/frame, then rest.

    The burst speed sits between the shipped start_speed (0.015) and a raised one (0.03), so a
    rally span forms under the shipped defaults but not under a stricter preset. Long rests on
    both sides isolate the burst; the trailing rest exceeds END_REST_FRAMES, so the span closes
    by rest, not by the track ending.
    """
    rest_pre, burst, rest_post = 40, 20, 100
    burst_step = 0.025
    xs = [0.5] * rest_pre
    position = 0.5
    for _ in range(burst):
        position += burst_step
        xs.append(position)
    xs += [position] * rest_post
    xs_arr = np.array(xs)
    ys = np.full_like(xs_arr, 0.5)
    vis = np.ones_like(xs_arr)
    return np.column_stack([xs_arr, ys, vis])


def test_thresholds_none_matches_explicit_shipped_preset():
    # thresholds=None reads the module globals; the shipped preset carries the same values, so
    # the two must agree bit-for-bit on the rally track (spans and contacts).
    track, _rally_start, _rally_end, _contacts = _build_rally_track()
    spans_globals, contacts_globals = segment_video(track)
    spans_shipped, contacts_shipped = segment_video(track, thresholds=SHIPPED_THRESHOLDS)
    assert spans_globals == spans_shipped
    assert contacts_globals == contacts_shipped


def test_thresholds_preset_changes_behaviour():
    # The 0.025 burst qualifies under the shipped START_SPEED 0.015 -> a span forms; a stricter
    # preset raising start_speed to 0.03 rejects the same burst -> no span, proving the preset
    # flows all the way through segmentation.
    track = _burst_track()
    assert len(segment_video(track, thresholds=SHIPPED_THRESHOLDS)[0]) >= 1
    stricter = SHIPPED_THRESHOLDS._replace(start_speed=0.03)
    assert segment_video(track, thresholds=stricter)[0] == []


# ---------------------------------------------------------------------------
# Replay mask: apply_replay_mask arithmetic + fail-loud + segment_video plumbing
# ---------------------------------------------------------------------------
def _distinct_track(n_frames: int) -> np.ndarray:
    """A (n, 3) track with a unique xy per frame and vis 0 everywhere.

    Distinct xy lets a test read exactly which frame a masked run froze to; vis 0 everywhere
    means a forced vis 1 is unambiguous evidence of masking.
    """
    xs = np.arange(n_frames, dtype=float) * 0.1
    ys = np.arange(n_frames, dtype=float) * 0.01 + 0.5
    vis = np.zeros(n_frames, dtype=float)
    return np.column_stack([xs, ys, vis])


def test_apply_replay_mask_mid_run_freezes_to_preceding_frame():
    track = _distinct_track(6)
    original = track.copy()
    mask = np.array([False, False, True, True, False, False])
    frozen = apply_replay_mask(track, mask)
    # Frames 2, 3 take frame 1's xy (the last live frame before the run); vis -> 1.
    assert np.array_equal(frozen[2, :2], track[1, :2])
    assert np.array_equal(frozen[3, :2], track[1, :2])
    assert frozen[2, 2] == 1.0 and frozen[3, 2] == 1.0
    assert np.array_equal(track, original)  # pure: the source track is untouched


def test_apply_replay_mask_run_at_frame_zero_freezes_to_first_post_run_frame():
    track = _distinct_track(6)
    mask = np.array([True, True, False, False, False, False])
    frozen = apply_replay_mask(track, mask)
    # No frame before the run, so frames 0, 1 take frame 2's xy (first live after).
    assert np.array_equal(frozen[0, :2], track[2, :2])
    assert np.array_equal(frozen[1, :2], track[2, :2])
    assert frozen[0, 2] == 1.0 and frozen[1, 2] == 1.0


def test_apply_replay_mask_two_runs_each_freeze_to_own_predecessor():
    track = _distinct_track(8)
    mask = np.array([False, True, False, False, True, True, False, False])
    frozen = apply_replay_mask(track, mask)
    assert np.array_equal(frozen[1, :2], track[0, :2])       # first run anchors to frame 0
    assert np.array_equal(frozen[4, :2], track[3, :2])       # second run anchors to frame 3
    assert np.array_equal(frozen[5, :2], track[3, :2])
    assert frozen[1, 2] == 1.0 and frozen[4, 2] == 1.0 and frozen[5, 2] == 1.0


def test_apply_replay_mask_all_false_returns_bit_identical():
    track = _distinct_track(5)
    assert np.array_equal(apply_replay_mask(track, np.zeros(5, dtype=bool)), track)


def test_apply_replay_mask_length_mismatch_raises():
    with pytest.raises(ValueError):
        apply_replay_mask(_distinct_track(5), np.zeros(4, dtype=bool))


def test_apply_replay_mask_all_true_raises():
    with pytest.raises(ValueError):
        apply_replay_mask(_distinct_track(5), np.ones(5, dtype=bool))


def test_apply_replay_mask_leaves_unmasked_frames_untouched():
    track = _distinct_track(6)
    mask = np.array([False, False, True, True, False, False])
    frozen = apply_replay_mask(track, mask)
    for frame in (0, 1, 4, 5):
        assert np.array_equal(frozen[frame], track[frame])
        assert frozen[frame, 2] == 0.0


def test_segment_video_exclusion_mask_freezes_masked_region_to_rest():
    # A whole sustained raw run freezes the complete rally region to rest.
    track, rally_start, rally_end, _contacts = _build_rally_track()
    assert len(segment_video(track)[0]) == 1
    mask = np.zeros(len(track), dtype=bool)
    mask[rally_start:rally_end] = True
    spans_masked, contacts_masked = segment_video(track, exclusion_mask=mask)
    assert spans_masked == []
    assert contacts_masked == []


# ---------------------------------------------------------------------------
# Span-open rule: region-start vs back-fill
# ---------------------------------------------------------------------------
def _span_open_speed_rest() -> tuple[np.ndarray, np.ndarray]:
    """Length-200 speed/at_rest: region [0,60) carries a qualifying fast burst, a 50-frame rest
    [60,110) splits (long at end_rest_frames <= 50), region [110,200) has no fast burst."""
    speed = np.zeros(200)
    speed[10:15] = 0.05  # a qualifying burst in region 1 (> START_SPEED, len 5 >= 3)
    at_rest = np.zeros(200, dtype=bool)
    at_rest[60:110] = True
    return speed, at_rest


def test_span_open_region_start_vs_back_fill_differ_on_no_burst_region():
    speed, at_rest = _span_open_speed_rest()
    thresholds = SHIPPED_THRESHOLDS._replace(end_rest_frames=40)
    region_start = _find_rally_spans_span_open(speed, at_rest, thresholds, SpanOpen.REGION_START)
    back_fill = _find_rally_spans_span_open(speed, at_rest, thresholds, SpanOpen.BACK_FILL)
    # The burst region opens at its start under BOTH rules; the no-burst region opens only under
    # REGION_START (the gate is dropped) and yields nothing under BACK_FILL (the gate holds).
    assert region_start == [(0, 60), (110, 200)]
    assert back_fill == [(0, 60)]


def _slow_drift_track() -> np.ndarray:
    """Rest, a visible slow drift (~0.01/frame, above REST_SPEED but below START_SPEED), rest.

    The drift is an active region (not rest) with no qualifying fast burst, so the default
    burst-open rule yields no span but REGION_START (gate dropped) does.
    """
    rest_pre, drift, rest_post = 40, 20, 40
    step = 0.01
    xs = [0.5] * rest_pre
    position = 0.5
    for _ in range(drift):
        position += step
        xs.append(position)
    xs += [position] * rest_post
    xs_arr = np.array(xs)
    ys = np.full_like(xs_arr, 0.5)
    vis = np.ones_like(xs_arr)
    return np.column_stack([xs_arr, ys, vis])


def test_segment_video_span_open_region_start_drops_the_burst_gate():
    # The slow-drift region carries no fast burst: the default rule finds no span, REGION_START
    # opens the region anyway.
    track = _slow_drift_track()
    assert segment_video(track)[0] == []
    assert len(segment_video(track, span_open=SpanOpen.REGION_START)[0]) >= 1


def test_segment_video_serve_start_with_region_start_raises():
    track, _rs, _re, _c = _build_rally_track()
    setup = ServeSetupInputs(
        count=np.ones(len(track)), wrist_dist=np.full((len(track), 2), np.nan),
        analysed=np.ones(len(track), dtype=bool),
        top_ankles=np.tile((0.2, 0.3), (len(track), 1)),
        bot_ankles=np.tile((0.7, 0.3), (len(track), 1)),
        top_height=np.full(len(track), 0.2), bot_height=np.full(len(track), 0.2),
    )
    options = ServeStartOptions(
        dist=None, threshold=0.10, mode=ServeStartMode.TRIM,
        setup=setup, lookback_frames=25,
    )
    with pytest.raises(ValueError):
        segment_video(track, serve_start=options, span_open=SpanOpen.REGION_START)


def test_segment_video_serve_start_split_with_back_fill_raises():
    # BACK_FILL emits one span per region; a split close has nothing to cut, so the combo raises
    # rather than silently swallowing the split (mirror of the REGION_START guard above).
    track, _rs, _re, _c = _build_rally_track()
    setup = ServeSetupInputs(
        count=np.ones(len(track)), wrist_dist=np.full((len(track), 2), np.nan),
        analysed=np.ones(len(track), dtype=bool),
        top_ankles=np.tile((0.2, 0.3), (len(track), 1)),
        bot_ankles=np.tile((0.7, 0.3), (len(track), 1)),
        top_height=np.full(len(track), 0.2), bot_height=np.full(len(track), 0.2),
    )
    options = ServeStartOptions(
        dist=None, threshold=0.10, mode=ServeStartMode.TRIM, close=ServeStartClose.BURST,
        setup=setup, lookback_frames=25,
    )
    with pytest.raises(ValueError):
        segment_video(track, serve_start=options, span_open=SpanOpen.BACK_FILL)


# ---------------------------------------------------------------------------
# Serve-start option path
# ---------------------------------------------------------------------------
def _serve_start_speed_rest_setup(
    qualifying_bursts: set[int],
) -> tuple[np.ndarray, np.ndarray, ServeSetupInputs]:
    """Length-120 speed with two bursts and synthetic sticky setup evidence."""
    speed = np.zeros(120)
    speed[10:15] = 0.05  # > START_SPEED, >= START_MIN_FRAMES 3
    speed[60:65] = 0.05
    at_rest = np.zeros(120, dtype=bool)
    wrist_dist = np.full((120, 2), np.nan)
    for burst in qualifying_bursts:
        wrist_dist[max(0, burst - 25):burst, 0] = 0.01  # 0.05 body heights, below 0.10
    setup = ServeSetupInputs(
        count=np.ones(120), wrist_dist=wrist_dist, analysed=np.ones(120, dtype=bool),
        top_ankles=np.tile((0.2, 0.3), (120, 1)), bot_ankles=np.tile((0.7, 0.3), (120, 1)),
        top_height=np.full(120, 0.2), bot_height=np.full(120, 0.2),
    )
    return speed, at_rest, setup


def test_serve_start_opens_at_first_qualifying_burst():
    # Burst 10's lookback is NaN (fails); burst 60's is small (passes). Both modes open at 60.
    speed, at_rest, setup = _serve_start_speed_rest_setup({60})
    assert _find_rally_spans(speed, at_rest, _SERVE_THRESHOLDS) == [(10, 120)]  # stock opens at 10
    for mode in (ServeStartMode.TRIM, ServeStartMode.REJECT):
        options = ServeStartOptions(dist=None, threshold=0.10, mode=mode, setup=setup, lookback_frames=25)
        assert _serve_start_find_rally_spans(speed, at_rest, _SERVE_THRESHOLDS, options, None) == [(60, 120)]


def test_serve_start_trim_falls_back_when_no_qualifying_burst():
    speed, at_rest, setup = _serve_start_speed_rest_setup(set())  # nothing qualifies
    diag: dict = {}
    options = ServeStartOptions(
        dist=None, threshold=0.10, mode=ServeStartMode.TRIM, diagnostics=diag,
        setup=setup, lookback_frames=25,
    )
    assert _serve_start_find_rally_spans(speed, at_rest, _SERVE_THRESHOLDS, options, None) == [(10, 120)]
    assert diag['n_no_qualify'] == 1 and diag['n_qualified'] == 0
    assert diag['no_qualify_regions'] == [(0, 120)]


def test_serve_start_reject_drops_region_when_no_qualifying_burst():
    speed, at_rest, setup = _serve_start_speed_rest_setup(set())
    diag: dict = {}
    options = ServeStartOptions(
        dist=None, threshold=0.10, mode=ServeStartMode.REJECT, diagnostics=diag,
        setup=setup, lookback_frames=25,
    )
    assert _serve_start_find_rally_spans(speed, at_rest, _SERVE_THRESHOLDS, options, None) == []
    assert diag['n_no_qualify'] == 1 and diag['no_qualify_regions'] == [(0, 120)]


def test_serve_start_back_fill_opens_qualifying_region_at_region_start():
    # serve_start + BACK_FILL: the serve gate decides qualification, the span opens at region_start.
    speed, at_rest, setup = _serve_start_speed_rest_setup({60})
    options = ServeStartOptions(
        dist=None, threshold=0.10, mode=ServeStartMode.REJECT, setup=setup, lookback_frames=25,
    )
    assert _serve_start_find_rally_spans(speed, at_rest, _SERVE_THRESHOLDS, options, SpanOpen.BACK_FILL) == [(0, 120)]


def test_segment_video_serve_start_none_is_exact_stock():
    track, _rs, _re, _c = _build_rally_track()
    assert segment_video(track, serve_start=None) == segment_video(track)


def test_segment_video_serve_start_reject_all_nan_drops_all_spans():
    # A track that forms one span by default; all-NaN setup evidence qualifies no burst, so REJECT drops
    # every region and segment_video routes through the serve-start finder to return no spans.
    track, _rs, _re, _c = _build_rally_track()
    assert len(segment_video(track)[0]) == 1
    setup = ServeSetupInputs(
        count=np.ones(len(track)), wrist_dist=np.full((len(track), 2), np.nan),
        analysed=np.ones(len(track), dtype=bool),
        top_ankles=np.tile((0.2, 0.3), (len(track), 1)),
        bot_ankles=np.tile((0.7, 0.3), (len(track), 1)),
        top_height=np.full(len(track), 0.2), bot_height=np.full(len(track), 0.2),
    )
    options = ServeStartOptions(
        dist=None, threshold=0.10, mode=ServeStartMode.REJECT,
        setup=setup, lookback_frames=25,
    )
    spans, contacts = segment_video(track, serve_start=options)
    assert spans == [] and contacts == []


# ---------------------------------------------------------------------------
# Serve-start split (close placement)
# ---------------------------------------------------------------------------
def _three_burst_speed_rest_setup(
    qualifying_bursts: set[int], rest_runs: tuple[tuple[int, int], ...] = (),
) -> tuple[np.ndarray, np.ndarray, ServeSetupInputs]:
    """Length-200 speed with three 5-frame bursts at 10, 80, 150, plus optional short rest runs.

    ``qualifying_bursts`` (a subset of {10, 80, 150}) get sticky setup evidence over their
    lookback; ``rest_runs`` stay shorter than end_rest_frames 40 so the track is one region.
    """
    speed = np.zeros(200)
    for burst in (10, 80, 150):
        speed[burst:burst + 5] = 0.05
    at_rest = np.zeros(200, dtype=bool)
    for start, end in rest_runs:
        at_rest[start:end] = True
    wrist_dist = np.full((200, 2), np.nan)
    for burst in qualifying_bursts:
        wrist_dist[max(0, burst - 25):burst, 0] = 0.01
    setup = ServeSetupInputs(
        count=np.ones(200), wrist_dist=wrist_dist, analysed=np.ones(200, dtype=bool),
        top_ankles=np.tile((0.2, 0.3), (200, 1)), bot_ankles=np.tile((0.7, 0.3), (200, 1)),
        top_height=np.full(200, 0.2), bot_height=np.full(200, 0.2),
    )
    return speed, at_rest, setup


def test_last_rest_close_picks_last_qualifying_run_else_burst():
    rest_runs = [(5, 8), (30, 40), (55, 60), (90, 100)]
    assert _last_rest_close(rest_runs, open_frame=10, next_burst=80) == 55   # later of (30,40),(55,60)
    assert _last_rest_close(rest_runs, open_frame=45, next_burst=80) == 55   # (30,40) starts before open
    assert _last_rest_close(rest_runs, open_frame=10, next_burst=25) == 25   # none between -> burst
    assert _last_rest_close(rest_runs, open_frame=60, next_burst=95) == 95   # (90,100) ends past burst


def test_serve_start_split_off_is_single_span():
    speed, at_rest, setup = _three_burst_speed_rest_setup({10, 80, 150})
    for mode in (ServeStartMode.TRIM, ServeStartMode.REJECT):
        diag: dict = {}
        options = ServeStartOptions(
            dist=None, threshold=0.10, mode=mode, diagnostics=diag,
            setup=setup, lookback_frames=25,
        )
        assert _serve_start_find_rally_spans(speed, at_rest, _SERVE_THRESHOLDS, options, None) == [(10, 200)]
        assert diag['qualifying_counts'] == [3]


def test_serve_start_split_burst_cuts_at_every_qualifying_burst():
    speed, at_rest, setup = _three_burst_speed_rest_setup({10, 80, 150})
    options = ServeStartOptions(
        dist=None, threshold=0.10, mode=ServeStartMode.REJECT, close=ServeStartClose.BURST,
        setup=setup, lookback_frames=25,
    )
    assert _serve_start_find_rally_spans(speed, at_rest, _SERVE_THRESHOLDS, options, None) == [
        (10, 80), (80, 150), (150, 200)]


def test_serve_start_split_burst_unions_to_the_single_span():
    speed, at_rest, setup = _three_burst_speed_rest_setup({10, 80, 150})
    single = _serve_start_find_rally_spans(
        speed, at_rest, _SERVE_THRESHOLDS,
        ServeStartOptions(
            dist=None, threshold=0.10, mode=ServeStartMode.REJECT,
            setup=setup, lookback_frames=25,
        ), None)
    split = _serve_start_find_rally_spans(
        speed, at_rest, _SERVE_THRESHOLDS,
        ServeStartOptions(
            dist=None, threshold=0.10, mode=ServeStartMode.REJECT, close=ServeStartClose.BURST,
            setup=setup, lookback_frames=25,
        ), None)
    assert single == [(10, 200)]
    assert split[0][0] == single[0][0] and split[-1][1] == single[0][1]
    assert all(earlier[1] == later[0] for earlier, later in zip(split, split[1:]))  # contiguous


def test_serve_start_split_last_rest_picks_run_else_falls_back_to_burst():
    speed, at_rest, setup = _three_burst_speed_rest_setup({10, 80, 150}, rest_runs=((100, 110),))
    options = ServeStartOptions(
        dist=None, threshold=0.10, mode=ServeStartMode.REJECT, close=ServeStartClose.LAST_REST,
        setup=setup, lookback_frames=25,
    )
    assert _serve_start_find_rally_spans(speed, at_rest, _SERVE_THRESHOLDS, options, None) == [
        (10, 80), (80, 100), (150, 200)]


def test_serve_start_split_last_rest_takes_the_last_of_several_runs():
    speed, at_rest, setup = _three_burst_speed_rest_setup({10, 80}, rest_runs=((30, 40), (55, 60)))
    options = ServeStartOptions(
        dist=None, threshold=0.10, mode=ServeStartMode.REJECT, close=ServeStartClose.LAST_REST,
        setup=setup, lookback_frames=25,
    )
    assert _serve_start_find_rally_spans(speed, at_rest, _SERVE_THRESHOLDS, options, None) == [(10, 55), (80, 200)]


def test_serve_start_split_no_qualify_region_honours_mode():
    speed, at_rest, setup = _three_burst_speed_rest_setup(set())
    for close in (ServeStartClose.BURST, ServeStartClose.LAST_REST):
        diag: dict = {}
        trim = ServeStartOptions(
            dist=None, threshold=0.10, mode=ServeStartMode.TRIM, close=close, diagnostics=diag,
            setup=setup, lookback_frames=25,
        )
        assert _serve_start_find_rally_spans(speed, at_rest, _SERVE_THRESHOLDS, trim, None) == [(10, 200)]
        assert diag['n_no_qualify'] == 1 and diag['qualifying_counts'] == [0]
        reject = ServeStartOptions(
            dist=None, threshold=0.10, mode=ServeStartMode.REJECT, close=close,
            setup=setup, lookback_frames=25,
        )
        assert _serve_start_find_rally_spans(speed, at_rest, _SERVE_THRESHOLDS, reject, None) == []


def test_serve_start_split_diagnostics_carry_counts_and_spacings():
    speed, at_rest, setup = _three_burst_speed_rest_setup({10, 80, 150})
    diag: dict = {}
    options = ServeStartOptions(
        dist=None, threshold=0.10, mode=ServeStartMode.REJECT,
        close=ServeStartClose.BURST, diagnostics=diag, setup=setup, lookback_frames=25,
    )
    _serve_start_find_rally_spans(speed, at_rest, _SERVE_THRESHOLDS, options, None)
    assert diag['qualifying_counts'] == [3]
    assert diag['qualifying_spacings'] == [70, 70]  # 80-10, 150-80


def test_serve_distance_ratio_helper_uses_distance_mask_and_boundary() -> None:
    window_dist = np.array([0.2, 0.4, np.nan])
    window_height = np.array([1.0, 1.0, 100.0])
    boundary = float(np.median(window_dist[:2]) / np.mean(window_height[:2]))
    assert _serve_distance_ratio_passes(window_dist, window_height, boundary)
    assert not _serve_distance_ratio_passes(window_dist, window_height, np.nextafter(boundary, 0.0))
    assert not _serve_distance_ratio_passes(np.full(3, np.nan), window_height, 1.0)
