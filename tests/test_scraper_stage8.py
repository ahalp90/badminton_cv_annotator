"""Stage 8 rally-segmentation tests: synthetic tracks, CPU-only, fast.

Tracks are built to make the truth obvious: a rest span, a zig-zag rally with a
known number of clean velocity reversals, then a long rest. Reversal spacing is
kept above SMOOTH_WINDOW so the de-dup does not collapse distinct contacts, and
the step size keeps the smoothed reversal speed above MIN_CONTACT_SPEED.
"""
import numpy as np
import pytest

from scripts.stage8_sweep import HOMOGRAPHY_COURT_BOX, PILOT_RESOLUTION, STANDIN_COURT_BOX
from src.scraper.config import (
    BEST_CONFIG_THRESHOLDS,
    END_REST_FRAMES,
    MIN_CONTACT_SPEED,
    SHIPPED_THRESHOLDS,
    SMOOTH_WINDOW,
    START_SPEED,
)
from src.scraper.stage8_rally_segmentation import (
    SERVE_START_LOOKBACK_FRAMES,
    ServeStartClose,
    ServeStartMode,
    ServeStartOptions,
    SpanOpen,
    _court_scale_boxes,
    _find_rally_spans,
    _find_rally_spans_span_open,
    _last_rest_close,
    _serve_setup_before,
    _serve_start_find_rally_spans,
    _wide_shot_before,
    apply_replay_mask,
    build_serve_start_wideshot_inputs,
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


# ---------------------------------------------------------------------------
# Thresholds option: None reads globals; a preset changes behaviour
# ---------------------------------------------------------------------------
# END_REST_FRAMES patched into a preset for the synthetic serve-start / split tracks below,
# whose bursts sit in one active region (no rest run reaches this bound).
_SERVE_THRESHOLDS = SHIPPED_THRESHOLDS._replace(end_rest_frames=40)


def _burst_track() -> np.ndarray:
    """Rest, a visible burst at ~0.025/frame, then rest.

    The burst speed sits between a lowered start_speed (0.02) and the shipped one (0.03), so a
    rally span forms only under the lower threshold. Long rests on both sides isolate the burst.
    """
    rest_pre, burst, rest_post = 40, 20, 40
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
    # The 0.025 burst never reads as fast under shipped START_SPEED 0.03 -> no span; a preset
    # lowering start_speed to 0.02 makes the same burst qualify -> a span forms.
    track = _burst_track()
    assert segment_video(track, thresholds=SHIPPED_THRESHOLDS)[0] == []
    lowered = SHIPPED_THRESHOLDS._replace(start_speed=0.02)
    assert len(segment_video(track, thresholds=lowered)[0]) >= 1


def test_thresholds_best_config_preset_shifts_spans():
    # The best-config preset (looser start/rest) forms a span on the burst track the shipped
    # preset rejects, proving the preset flows all the way through segmentation.
    track = _burst_track()
    assert segment_video(track, thresholds=SHIPPED_THRESHOLDS)[0] == []
    assert len(segment_video(track, thresholds=BEST_CONFIG_THRESHOLDS)[0]) >= 1


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


def test_segment_video_replay_mask_freezes_masked_region_to_rest():
    # A track that forms one span; masking the whole rally region freezes it to rest, so the
    # masked pass finds no span where the unmasked one did (the mask is applied inside
    # segment_video, before speed).
    track, rally_start, rally_end, _contacts = _build_rally_track()
    assert len(segment_video(track)[0]) == 1
    mask = np.zeros(len(track), dtype=bool)
    mask[rally_start:rally_end] = True
    spans_masked, contacts_masked = segment_video(track, replay_mask=mask)
    assert spans_masked == []
    assert contacts_masked == []


# ---------------------------------------------------------------------------
# Span-open rule: region-start vs back-fill
# ---------------------------------------------------------------------------
def _span_open_speed_rest() -> tuple[np.ndarray, np.ndarray]:
    """Length-200 speed/at_rest: region [0,60) carries a qualifying fast burst, a 50-frame rest
    [60,110) splits (long at end_rest_frames <= 50), region [110,200) has no fast burst."""
    speed = np.zeros(200)
    speed[10:15] = 0.05  # a qualifying burst in region 1 (> START_SPEED 0.03, len 5 >= 3)
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
    """Rest, a visible slow drift (~0.02/frame, above REST_SPEED but below START_SPEED), rest.

    The drift is an active region (not rest) with no qualifying fast burst, so the default
    burst-open rule yields no span but REGION_START (gate dropped) does.
    """
    rest_pre, drift, rest_post = 40, 20, 40
    step = 0.02
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
    options = ServeStartOptions(dist=np.full(len(track), np.nan), threshold=0.10, mode=ServeStartMode.TRIM)
    with pytest.raises(ValueError):
        segment_video(track, serve_start=options, span_open=SpanOpen.REGION_START)


def test_segment_video_serve_start_split_with_back_fill_raises():
    # BACK_FILL emits one span per region; a split close has nothing to cut, so the combo raises
    # rather than silently swallowing the split (mirror of the REGION_START guard above).
    track, _rs, _re, _c = _build_rally_track()
    options = ServeStartOptions(dist=np.full(len(track), np.nan), threshold=0.10,
                                mode=ServeStartMode.TRIM, close=ServeStartClose.BURST)
    with pytest.raises(ValueError):
        segment_video(track, serve_start=options, span_open=SpanOpen.BACK_FILL)


# ---------------------------------------------------------------------------
# Serve-start option path
# ---------------------------------------------------------------------------
def _serve_start_speed_rest_dist(qualifying_bursts: set[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Length-120 speed with two 5-frame bursts at 10 and 60, no rest, and a gate dist array.

    ``qualifying_bursts`` (a subset of {10, 60}) get a small (<= 0.10) distance over their
    SERVE_START_LOOKBACK lookback so the serve-setup gate passes; every other frame stays NaN.
    """
    speed = np.zeros(120)
    speed[10:15] = 0.05  # > START_SPEED 0.03, >= START_MIN_FRAMES 3
    speed[60:65] = 0.05
    at_rest = np.zeros(120, dtype=bool)
    dist = np.full(120, np.nan)
    for burst in qualifying_bursts:
        dist[max(0, burst - SERVE_START_LOOKBACK_FRAMES):burst] = 0.03  # median passes <= 0.10
    return speed, at_rest, dist


def test_serve_setup_before_gate_pass_fail_nan():
    dist = np.full(80, np.nan)
    dist[35:60] = 0.03
    assert _serve_setup_before(dist, 60, 0.10)          # median 0.03 <= 0.10
    assert not _serve_setup_before(dist, 60, 0.02)      # 0.03 > 0.02
    far = np.full(80, np.nan)
    far[35:60] = 0.5
    assert not _serve_setup_before(far, 60, 0.10)       # finite but far
    assert not _serve_setup_before(dist, 20, 0.10)      # all-NaN lookback


def test_serve_start_opens_at_first_qualifying_burst():
    # Burst 10's lookback is NaN (fails); burst 60's is small (passes). Both modes open at 60.
    speed, at_rest, dist = _serve_start_speed_rest_dist({60})
    assert _find_rally_spans(speed, at_rest, _SERVE_THRESHOLDS) == [(10, 120)]  # stock opens at 10
    for mode in (ServeStartMode.TRIM, ServeStartMode.REJECT):
        options = ServeStartOptions(dist=dist, threshold=0.10, mode=mode)
        assert _serve_start_find_rally_spans(speed, at_rest, _SERVE_THRESHOLDS, options, None) == [(60, 120)]


def test_serve_start_trim_falls_back_when_no_qualifying_burst():
    speed, at_rest, dist = _serve_start_speed_rest_dist(set())  # nothing qualifies
    diag: dict = {}
    options = ServeStartOptions(dist=dist, threshold=0.10, mode=ServeStartMode.TRIM, diagnostics=diag)
    assert _serve_start_find_rally_spans(speed, at_rest, _SERVE_THRESHOLDS, options, None) == [(10, 120)]
    assert diag['n_no_qualify'] == 1 and diag['n_qualified'] == 0
    assert diag['no_qualify_regions'] == [(0, 120)]


def test_serve_start_reject_drops_region_when_no_qualifying_burst():
    speed, at_rest, dist = _serve_start_speed_rest_dist(set())
    diag: dict = {}
    options = ServeStartOptions(dist=dist, threshold=0.10, mode=ServeStartMode.REJECT, diagnostics=diag)
    assert _serve_start_find_rally_spans(speed, at_rest, _SERVE_THRESHOLDS, options, None) == []
    assert diag['n_no_qualify'] == 1 and diag['no_qualify_regions'] == [(0, 120)]


def test_serve_start_back_fill_opens_qualifying_region_at_region_start():
    # serve_start + BACK_FILL: the serve gate decides qualification, the span opens at region_start.
    speed, at_rest, dist = _serve_start_speed_rest_dist({60})
    options = ServeStartOptions(dist=dist, threshold=0.10, mode=ServeStartMode.REJECT)
    assert _serve_start_find_rally_spans(speed, at_rest, _SERVE_THRESHOLDS, options, SpanOpen.BACK_FILL) == [(0, 120)]


def test_segment_video_serve_start_none_is_exact_stock():
    track, _rs, _re, _c = _build_rally_track()
    assert segment_video(track, serve_start=None) == segment_video(track)


def test_segment_video_serve_start_reject_all_nan_drops_all_spans():
    # A track that forms one span by default; an all-NaN dist qualifies no burst, so REJECT drops
    # every region and segment_video routes through the serve-start finder to return no spans.
    track, _rs, _re, _c = _build_rally_track()
    assert len(segment_video(track)[0]) == 1
    options = ServeStartOptions(dist=np.full(len(track), np.nan), threshold=0.10, mode=ServeStartMode.REJECT)
    spans, contacts = segment_video(track, serve_start=options)
    assert spans == [] and contacts == []


# ---------------------------------------------------------------------------
# Serve-start wide-shot refinement
# ---------------------------------------------------------------------------
# Synthetic court-scale boxes under the stand-in CourtBox (court x [635, 1316], foot y
# [254, 1030], height [84, 336], mid-line 642): one player per half, static.
TOP_BOX = (900.0, 500.0, 150.0)   # (foot_x, foot_y, height) px; foot y 500 < 642 -> top half
BOT_BOX = (1000.0, 900.0, 250.0)  # foot y 900 >= 642 -> bottom half


def _mk_wideshot_inputs(frame_boxes: list[list[tuple[float, float, float]]]):
    """WideshotInputs from per-frame (foot_x, foot_y, height) pixel boxes, stand-in CourtBox."""
    n_frames = len(frame_boxes)
    bboxes = np.full((n_frames, 16, 4), np.nan)
    scores = np.full((n_frames, 16), np.nan)
    for frame, boxes in enumerate(frame_boxes):
        for slot, (foot_x, foot_y, height) in enumerate(boxes):
            bboxes[frame, slot] = (foot_x - 30.0, foot_y - height, foot_x + 30.0, foot_y)
            scores[frame, slot] = 0.9 - 0.1 * slot
    return build_serve_start_wideshot_inputs(bboxes, scores, STANDIN_COURT_BOX, PILOT_RESOLUTION)


def test_wide_shot_gate_passes_on_static_two_player_lookback():
    inputs = _mk_wideshot_inputs([[TOP_BOX, BOT_BOX]] * 25)
    assert _wide_shot_before(inputs, burst_start=25)


def test_wide_shot_gate_count_fail():
    # Each half occupied for 13 of 25 frames but only frame 12 has both: count median 1 < 2.
    frames: list[list[tuple[float, float, float]]] = [[TOP_BOX] for _ in range(25)]
    for frame in range(12, 25):
        frames[frame] = [BOT_BOX]
    frames[12] = [TOP_BOX, BOT_BOX]
    assert not _wide_shot_before(_mk_wideshot_inputs(frames), burst_start=25)


def test_wide_shot_gate_slot_fail():
    # Two static players, both TOP: count_med 2 passes but the bottom half is never occupied.
    second_top = (800.0, 550.0, 160.0)
    assert not _wide_shot_before(_mk_wideshot_inputs([[TOP_BOX, second_top]] * 25), burst_start=25)


def test_wide_shot_gate_drift_fail():
    # Bottom player walks 10 px/frame: head/tail means 150 px apart (0.078 > 0.05).
    frames = [[TOP_BOX, (1000.0 + 10.0 * frame, 900.0, 250.0)] for frame in range(25)]
    assert not _wide_shot_before(_mk_wideshot_inputs(frames), burst_start=25)


def test_wide_shot_gate_short_series_drift_abstains():
    # Ten present feet fill one drift window (head/tail overlap -> 0.0 even as the player
    # sprints): the short series abstains to NaN and the gate fails closed.
    frames = [[TOP_BOX, (1000.0 + 15.0 * frame, 900.0, 250.0)] for frame in range(10)]
    assert not _wide_shot_before(_mk_wideshot_inputs(frames), burst_start=10)


def test_wide_shot_gate_empty_or_truncated_lookback_fails():
    inputs = _mk_wideshot_inputs([[] for _ in range(25)])
    assert not _wide_shot_before(inputs, burst_start=25)
    assert not _wide_shot_before(inputs, burst_start=0)


def test_serve_start_wideshot_requires_both_gates():
    # Bursts 10 and 60 both pass the distance gate; only burst 60's lookback holds the wide shot.
    speed, at_rest, dist = _serve_start_speed_rest_dist({10, 60})
    frames: list[list[tuple[float, float, float]]] = [[] for _ in range(120)]
    for frame in range(35, 60):
        frames[frame] = [TOP_BOX, BOT_BOX]
    inputs = _mk_wideshot_inputs(frames)
    on = ServeStartOptions(dist=dist, threshold=0.10, mode=ServeStartMode.TRIM, wideshot=inputs)
    assert _serve_start_find_rally_spans(speed, at_rest, _SERVE_THRESHOLDS, on, None) == [(60, 120)]
    off = ServeStartOptions(dist=dist, threshold=0.10, mode=ServeStartMode.TRIM)
    assert _serve_start_find_rally_spans(speed, at_rest, _SERVE_THRESHOLDS, off, None) == [(10, 120)]


def test_serve_start_wideshot_off_is_prior_behaviour():
    # wideshot=None reproduces the pre-refinement picks even when a failing wideshot vetoed 60.
    speed, at_rest, dist = _serve_start_speed_rest_dist({60})
    failing = _mk_wideshot_inputs([[] for _ in range(120)])
    vetoed = ServeStartOptions(dist=dist, threshold=0.10, mode=ServeStartMode.TRIM, wideshot=failing)
    assert _serve_start_find_rally_spans(speed, at_rest, _SERVE_THRESHOLDS, vetoed, None) == [(10, 120)]
    for mode in (ServeStartMode.TRIM, ServeStartMode.REJECT):
        off = ServeStartOptions(dist=dist, threshold=0.10, mode=mode)
        assert _serve_start_find_rally_spans(speed, at_rest, _SERVE_THRESHOLDS, off, None) == [(60, 120)]


# ---------------------------------------------------------------------------
# Serve-start split (close placement)
# ---------------------------------------------------------------------------
def _three_burst_speed_rest_dist(
    qualifying_bursts: set[int], rest_runs: tuple[tuple[int, int], ...] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Length-200 speed with three 5-frame bursts at 10, 80, 150, plus optional short rest runs.

    ``qualifying_bursts`` (a subset of {10, 80, 150}) get a small (<= 0.10) distance over their
    lookback; ``rest_runs`` stay shorter than end_rest_frames 40 so the track is one region.
    """
    speed = np.zeros(200)
    for burst in (10, 80, 150):
        speed[burst:burst + 5] = 0.05
    at_rest = np.zeros(200, dtype=bool)
    for start, end in rest_runs:
        at_rest[start:end] = True
    dist = np.full(200, np.nan)
    for burst in qualifying_bursts:
        dist[max(0, burst - SERVE_START_LOOKBACK_FRAMES):burst] = 0.03
    return speed, at_rest, dist


def test_last_rest_close_picks_last_qualifying_run_else_burst():
    rest_runs = [(5, 8), (30, 40), (55, 60), (90, 100)]
    assert _last_rest_close(rest_runs, open_frame=10, next_burst=80) == 55   # later of (30,40),(55,60)
    assert _last_rest_close(rest_runs, open_frame=45, next_burst=80) == 55   # (30,40) starts before open
    assert _last_rest_close(rest_runs, open_frame=10, next_burst=25) == 25   # none between -> burst
    assert _last_rest_close(rest_runs, open_frame=60, next_burst=95) == 95   # (90,100) ends past burst


def test_serve_start_split_off_is_single_span():
    speed, at_rest, dist = _three_burst_speed_rest_dist({10, 80, 150})
    for mode in (ServeStartMode.TRIM, ServeStartMode.REJECT):
        diag: dict = {}
        options = ServeStartOptions(dist=dist, threshold=0.10, mode=mode, diagnostics=diag)
        assert _serve_start_find_rally_spans(speed, at_rest, _SERVE_THRESHOLDS, options, None) == [(10, 200)]
        assert diag['qualifying_counts'] == [3]


def test_serve_start_split_burst_cuts_at_every_qualifying_burst():
    speed, at_rest, dist = _three_burst_speed_rest_dist({10, 80, 150})
    options = ServeStartOptions(dist=dist, threshold=0.10, mode=ServeStartMode.REJECT,
                                close=ServeStartClose.BURST)
    assert _serve_start_find_rally_spans(speed, at_rest, _SERVE_THRESHOLDS, options, None) == [
        (10, 80), (80, 150), (150, 200)]


def test_serve_start_split_burst_unions_to_the_single_span():
    speed, at_rest, dist = _three_burst_speed_rest_dist({10, 80, 150})
    single = _serve_start_find_rally_spans(
        speed, at_rest, _SERVE_THRESHOLDS,
        ServeStartOptions(dist=dist, threshold=0.10, mode=ServeStartMode.REJECT), None)
    split = _serve_start_find_rally_spans(
        speed, at_rest, _SERVE_THRESHOLDS,
        ServeStartOptions(dist=dist, threshold=0.10, mode=ServeStartMode.REJECT, close=ServeStartClose.BURST), None)
    assert single == [(10, 200)]
    assert split[0][0] == single[0][0] and split[-1][1] == single[0][1]
    assert all(earlier[1] == later[0] for earlier, later in zip(split, split[1:]))  # contiguous


def test_serve_start_split_last_rest_picks_run_else_falls_back_to_burst():
    speed, at_rest, dist = _three_burst_speed_rest_dist({10, 80, 150}, rest_runs=((100, 110),))
    options = ServeStartOptions(dist=dist, threshold=0.10, mode=ServeStartMode.REJECT,
                                close=ServeStartClose.LAST_REST)
    assert _serve_start_find_rally_spans(speed, at_rest, _SERVE_THRESHOLDS, options, None) == [
        (10, 80), (80, 100), (150, 200)]


def test_serve_start_split_last_rest_takes_the_last_of_several_runs():
    speed, at_rest, dist = _three_burst_speed_rest_dist({10, 80}, rest_runs=((30, 40), (55, 60)))
    options = ServeStartOptions(dist=dist, threshold=0.10, mode=ServeStartMode.REJECT,
                                close=ServeStartClose.LAST_REST)
    assert _serve_start_find_rally_spans(speed, at_rest, _SERVE_THRESHOLDS, options, None) == [(10, 55), (80, 200)]


def test_serve_start_split_no_qualify_region_honours_mode():
    speed, at_rest, dist = _three_burst_speed_rest_dist(set())
    for close in (ServeStartClose.BURST, ServeStartClose.LAST_REST):
        diag: dict = {}
        trim = ServeStartOptions(dist=dist, threshold=0.10, mode=ServeStartMode.TRIM,
                                 close=close, diagnostics=diag)
        assert _serve_start_find_rally_spans(speed, at_rest, _SERVE_THRESHOLDS, trim, None) == [(10, 200)]
        assert diag['n_no_qualify'] == 1 and diag['qualifying_counts'] == [0]
        reject = ServeStartOptions(dist=dist, threshold=0.10, mode=ServeStartMode.REJECT, close=close)
        assert _serve_start_find_rally_spans(speed, at_rest, _SERVE_THRESHOLDS, reject, None) == []


def test_serve_start_split_diagnostics_carry_counts_and_spacings():
    speed, at_rest, dist = _three_burst_speed_rest_dist({10, 80, 150})
    diag: dict = {}
    options = ServeStartOptions(dist=dist, threshold=0.10, mode=ServeStartMode.REJECT,
                                close=ServeStartClose.BURST, diagnostics=diag)
    _serve_start_find_rally_spans(speed, at_rest, _SERVE_THRESHOLDS, options, None)
    assert diag['qualifying_counts'] == [3]
    assert diag['qualifying_spacings'] == [70, 70]  # 80-10, 150-80


# ---------------------------------------------------------------------------
# Court box: the builders filter against the explicit CourtBox
# ---------------------------------------------------------------------------
def test_court_box_filter_honours_standin_vs_homography():
    # A foot at y=300 sits inside the stand-in court but above the homography quad's top edge
    # (461.1), so _court_scale_boxes keeps the box under stand-in and drops it under homography.
    bboxes = np.full((16, 4), np.nan)
    scores = np.full(16, np.nan)
    bboxes[0] = (970.0, 150.0, 1030.0, 300.0)  # foot (1000, 300), height 150
    scores[0] = 0.9
    assert len(_court_scale_boxes(bboxes, scores, STANDIN_COURT_BOX)[0]) == 1
    assert len(_court_scale_boxes(bboxes, scores, HOMOGRAPHY_COURT_BOX)[0]) == 0


def test_court_box_homography_mid_band_foot_claims_neither_half():
    # A foot inside the homography mid band (664.6, 703.7) claims NEITHER court half, while the
    # court-scale count still sees it. One foot clearly top, one in-band: top filled, bottom empty.
    bboxes = np.full((1, 16, 4), np.nan)
    scores = np.full((1, 16), np.nan)
    bboxes[0, 0] = (900.0, 380.0, 960.0, 500.0)    # foot (930, 500): top half under homography
    bboxes[0, 1] = (1000.0, 560.0, 1060.0, 684.0)  # foot (1030, 684): inside the band
    scores[0, :2] = 0.9
    inputs = build_serve_start_wideshot_inputs(bboxes, scores, HOMOGRAPHY_COURT_BOX, PILOT_RESOLUTION)
    assert inputs.count[0] == 2
    assert np.isfinite(inputs.top_foot[0]).all()
    assert np.isnan(inputs.bot_foot[0]).all()
