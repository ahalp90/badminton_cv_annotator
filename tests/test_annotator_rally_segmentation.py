"""Rally-segmentation tests using fast, CPU-only synthetic tracks.

Tracks are built to make the truth obvious: a rest span, a zig-zag rally with a
known number of clean velocity reversals, then a long rest. The contact tests
pin the measured impulse rule and its largest-impulse de-duplication.
"""

from dataclasses import replace

import numpy as np
import pytest

from annotator.array_calc_helpers import compute_speed, rolling_mean, rolling_nanmean
from annotator.config import BaseAnnotatorConfig, scale_config_to_fps
from annotator.contacts import heuristics as contact_detection
from annotator.contacts.heuristics import (
    impulse_contact_candidates,
    keep_strongest_spaced_contacts,
    span_impulses,
    wrist_contact_near,
)
from annotator.excluded_frames.replay import apply_replay_mask
from annotator.rallies.find_rallies import active_region_rally_spans, segment_video
from annotator.types import SmoothingMode
from tests.annotator_config import trajectory_config

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
    config = trajectory_config()
    config = replace(config, thresholds=replace(config.thresholds, end_rest_frames=40))
    spans, contacts = segment_video(track, video_config=config)

    assert len(spans) == 1
    start, end = spans[0]
    assert abs(start - rally_start) <= 4
    assert abs(end - rally_end) <= 4

    contact_frames = sorted(frame for _, frame, *_ in contacts)
    # The zigzag's symmetric reversals produce exact impulse ties; keep the earlier frame.
    assert contact_frames == [46, 50, 57, 64, 71]


def test_static_track_yields_no_rally():
    track = np.column_stack([
        np.full(150, 0.5), np.full(150, 0.5), np.ones(150),
    ])
    spans, contacts = segment_video(track, video_config=trajectory_config())
    assert spans == []
    assert contacts == []


def test_invisible_moving_track_reads_as_rest():
    # Fast motion but never tracked: every step is NaN, so nothing reads as fast
    # and the window is mostly untracked -> rest. No rally is found.
    ys = np.tile([0.1, 0.9], 75)
    track = np.column_stack([np.full(150, 0.5), ys, np.zeros(150)])
    spans, _ = segment_video(track, video_config=trajectory_config())
    assert spans == []


def test_ignore_invisible_smoothing_uses_even_fps_scaled_window() -> None:
    thresholds = replace(
        trajectory_config().thresholds, smooth_window=scale_config_to_fps(BaseAnnotatorConfig(), 60.0).thresholds.smooth_window
    )
    assert thresholds.smooth_window == 6
    track = np.column_stack([
        [1.0, 99.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0],
        np.zeros(8),
        [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ])

    measured = span_impulses(
        track, 0, len(track), smooth_window=thresholds.smooth_window,
        impulse_floor_half_window=thresholds.impulse_floor_half_window_frames,
        smoothing_mode=SmoothingMode.IGNORE_INVISIBLE,
    )
    assert measured is not None
    impulses, _ratios, _visible = measured

    # Width-six smoothing gives [2, 3, 4, 5, 7, 8, 9, 10], so the impulse is
    # the absolute velocity change [0, 0, 1, 1, 0, 0].
    np.testing.assert_array_equal(impulses, [0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    assert track[1, 0] == 99.0


def test_ignore_invisible_smoothing_marks_all_invisible_span_unmeasurable() -> None:
    thresholds = replace(trajectory_config().thresholds, smooth_window=6)
    track = np.column_stack([np.arange(8, dtype=float), np.zeros(8), np.zeros(8)])

    measured = span_impulses(
        track, 0, len(track), smooth_window=thresholds.smooth_window,
        impulse_floor_half_window=thresholds.impulse_floor_half_window_frames,
        smoothing_mode=SmoothingMode.IGNORE_INVISIBLE,
    )
    assert measured is not None
    impulses, ratios, visible = measured
    assert np.isnan(impulses).all()
    assert np.isnan(ratios).all()
    assert not visible.any()


def test_nan_rolling_mean_matches_stock_mean_on_finite_input() -> None:
    values = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0])

    assert np.array_equal(rolling_nanmean(values, 6), rolling_mean(values, 6))


def _triangle_track(step: float) -> np.ndarray:
    """A single up-down triangle in y, all visible, for direct contact tests."""
    base = np.array([0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0])
    ys = REST_Y + step * base
    return np.column_stack([np.full(len(ys), 0.5), ys, np.ones(len(ys))])


def test_contact_detected_on_fast_reversal():
    track = _triangle_track(RALLY_STEP)
    contacts = [
        frame
        for frame, _impulse in impulse_contact_candidates(
            track, 0, len(track), thresholds=trajectory_config().thresholds, smoothing_mode=SmoothingMode.INCLUDE_INVISIBLE
        )
    ]
    assert contacts == [1, 7, 13]


def test_contact_suppression_ranks_impulse_then_frame():
    # Radius pinned explicitly: this test is about the ranking rule, and the module default
    # scales with the fps table (8 at the 25 fps surface under base-30, where this fixture
    # would suppress nothing).
    flags = [(27, 2.0), (19, 2.0), (10, 1.0), (18, 2.0)]
    assert keep_strongest_spaced_contacts(flags, radius=9) == [(18, 2.0), (27, 2.0)]
    assert impulse_contact_candidates(
        _triangle_track(RALLY_STEP), 0, 15, thresholds=trajectory_config().thresholds, smoothing_mode=SmoothingMode.INCLUDE_INVISIBLE
    )


@pytest.mark.parametrize('with_wrist_distances', [False, True])
def test_contact_assembly_keeps_gate_verdicts_and_suppresses_across_rallies(monkeypatch, with_wrist_distances):
    track = np.zeros((40, 3))
    thresholds = replace(trajectory_config().thresholds, contact_suppression_radius_frames=9)
    candidates = {0: [(18, 2.0)], 20: [(22, 1.0), (26, 10.0)]}
    monkeypatch.setattr(
        contact_detection, 'impulse_contact_candidates',
        lambda _track, start, *_args, **_kwargs: candidates[start],
    )
    distances = None
    if with_wrist_distances:
        distances = np.zeros(len(track))
        distances[26] = np.inf
    contacts = contact_detection.assemble_contacts(
        track, [(0, 20), (20, 40)], thresholds, distances, smoothing_mode=SmoothingMode.INCLUDE_INVISIBLE,
    )
    assert [(contact.rally_id, contact.contact_frame) for contact in contacts] == [(0, 18), (1, 22), (1, 26)]
    if with_wrist_distances:
        assert [contact.wrist_near for contact in contacts] == [True, True, False]
        assert [contact.temporal_proximity_suppressed for contact in contacts] == [False, True, False]
    else:
        assert all(contact.wrist_near is None and contact.temporal_proximity_suppressed is None for contact in contacts)


# ---------------------------------------------------------------------------
# Contact wrist checks and segmentation
# ---------------------------------------------------------------------------


def test_wrist_contact_near_verdicts():
    assert wrist_contact_near(None, 0, threshold=1.4) is None
    nearest_wrist_distances = np.array([1.39, 1.4, 1.41, np.nan])
    assert wrist_contact_near(nearest_wrist_distances, 0, threshold=1.4) is True
    assert wrist_contact_near(nearest_wrist_distances, 1, threshold=1.4) is True
    assert wrist_contact_near(nearest_wrist_distances, 2, threshold=1.4) is False
    assert wrist_contact_near(nearest_wrist_distances, 3, threshold=1.4) is False


def test_segment_video_wrist_near_paths():
    track, _, _, _ = _build_rally_track()

    # No gate inputs at all: the gate never ran, both verdicts are blank (None), and no
    # suppression happens, so every raw candidate stands (recall-first).
    _, contacts_blank = segment_video(track, video_config=trajectory_config())
    assert contacts_blank
    assert all(contact.wrist_near is None and contact.temporal_proximity_suppressed is None for contact in contacts_blank)
    assert contacts_blank[0][:4] == (
        contacts_blank[0].rally_id, contacts_blank[0].contact_frame,
        contacts_blank[0].wrist_near, contacts_blank[0].temporal_proximity_suppressed,
    )

    # A wrist on the shuttle at every frame (distance 0): every candidate passes the gate, and
    # suppression records only its losers.
    near = np.zeros(len(track))
    _, contacts_near = segment_video(track, nearest_wrist_distances=near, video_config=trajectory_config())
    assert [contact.wrist_near for contact in contacts_near] == [True] * len(contacts_near)
    assert [contact.temporal_proximity_suppressed for contact in contacts_near] == [True, False, True, False, True]
    filtered = [
        contact for contact in contacts_near
        if contact.wrist_near is not False and contact.temporal_proximity_suppressed is not True
    ]
    assert [contact.contact_frame for contact in filtered] == [
        contact.contact_frame for contact in contacts_near if not contact.temporal_proximity_suppressed
    ]

    # A wrist far from the shuttle everywhere: every contact drops (False).
    far = np.full(len(track), 2.0)
    _, contacts_far = segment_video(track, nearest_wrist_distances=far, video_config=trajectory_config())
    assert all(contact.wrist_near is False and contact.temporal_proximity_suppressed is False for contact in contacts_far)


@pytest.mark.parametrize('failure_distance', [np.nan, np.inf])
def test_sticky_gate_failure_rows_fail_closed(failure_distance):
    """A no-evidence distance row (NaN inside spans, +inf outside) fails the gate closed.

    Uses the cached-distances injection surface that replaced the old internal
    monkeypatch: the caller supplies the full-length series directly.
    """
    track, _, _, _ = _build_rally_track()
    _, contacts = segment_video(track, nearest_wrist_distances=np.full(len(track), failure_distance), video_config=trajectory_config())
    assert contacts
    assert all(contact.wrist_near is False and contact.temporal_proximity_suppressed is False for contact in contacts)


# ---------------------------------------------------------------------------
# Resolved thresholds change which bursts qualify
# ---------------------------------------------------------------------------
# END_REST_FRAMES patched into a preset for the synthetic serve-start / split tracks below,
# whose bursts sit in one active region (no rest run reaches this bound).
_SERVE_THRESHOLDS = replace(trajectory_config().thresholds, end_rest_frames=40)


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


def test_thresholds_preset_changes_behaviour():
    # The 0.025 burst qualifies under the shipped START_SPEED 0.015 -> a span forms; a stricter
    # preset raising start_speed to 0.03 rejects the same burst -> no span, proving the preset
    # flows all the way through segmentation.
    track = _burst_track()
    assert len(segment_video(track, video_config=trajectory_config())[0]) >= 1
    stricter = replace(trajectory_config().thresholds, start_speed=0.03)
    assert segment_video(track, video_config=replace(trajectory_config(), thresholds=stricter))[0] == []


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
    assert len(segment_video(track, video_config=trajectory_config())[0]) == 1
    mask = np.zeros(len(track), dtype=bool)
    mask[rally_start:rally_end] = True
    spans_masked, contacts_masked = segment_video(track, exclusion_mask=mask, video_config=trajectory_config())
    assert spans_masked == []
    assert contacts_masked == []


# ---------------------------------------------------------------------------
# Span-open rule: region-start vs back-fill
# ---------------------------------------------------------------------------
def _two_active_regions() -> tuple[np.ndarray, np.ndarray]:
    """Length-200 speed/at_rest: region [0,60) carries a qualifying fast burst, a 50-frame rest
    [60,110) splits (long at end_rest_frames <= 50), region [110,200) has no fast burst."""
    speed = np.zeros(200)
    speed[10:15] = 0.05  # a qualifying burst in region 1 (> START_SPEED, len 5 >= 3)
    at_rest = np.zeros(200, dtype=bool)
    at_rest[60:110] = True
    return speed, at_rest


def test_requiring_fast_burst_discards_the_region_without_one():
    speed, at_rest = _two_active_regions()
    thresholds = replace(trajectory_config().thresholds, end_rest_frames=40)
    region_start = active_region_rally_spans(speed, at_rest, thresholds, require_fast_burst=False)
    back_fill = active_region_rally_spans(speed, at_rest, thresholds, require_fast_burst=True)

    assert region_start == [(0, 60), (110, 200)]
    assert back_fill == [(0, 60)]


def _slow_drift_track() -> np.ndarray:
    """Rest, a visible slow drift (~0.01/frame, above REST_SPEED but below START_SPEED), rest.

    The drift exceeds rest speed but never reaches the fast-burst threshold.
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


def test_segment_video_can_keep_an_active_region_without_fast_motion():

    track = _slow_drift_track()
    assert segment_video(track, video_config=trajectory_config())[0] == []
    assert len(segment_video(track, video_config=replace(trajectory_config(), require_fast_burst=False))[0]) >= 1
