"""Player-selection rules, running-position updates and BST-X output conventions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shared.player_tracking import (
    BOTTOM_ROW,
    TOP_ROW,
    StickyAnchorParams,
    compute_halfcourt_centres,
    track_frame_players,
)
from src.bst_x.preparing_data.heuristics.base import (
    ClipContext,
    RawClip,
)
from src.bst_x.preparing_data.heuristics.sticky_anchor import _run_clip

J = 17  # COCO 17-joint skeleton


def _identity_court_ctx(vid: int = 1):
    """Build a synthetic ClipContext whose pixel->normalised mapping is identity over [0, 1280]x[0, 720]."""
    H = np.eye(3, dtype=np.float64)
    court_info = {
        "H": H,
        "border_L": 0.0,
        "border_R": 1280.0,
        "border_U": 0.0,
        "border_D": 720.0,
    }
    all_court_info = {vid: court_info}
    res_df = pd.DataFrame({"width": [1280], "height": [720]}, index=[vid])
    return ClipContext(vid=vid, all_court_info=all_court_info, res_df=res_df)


def _bbox_for(norm_x: float, norm_y: float, half_w: float = 30.0, half_h: float = 60.0) -> np.ndarray:
    """Return a bbox whose bottom-centre projects to ``(norm_x, norm_y)`` in normalised court coords."""
    cx = norm_x * 1280.0
    by = norm_y * 720.0
    x1, x2 = cx - half_w, cx + half_w
    y1, y2 = by - 2 * half_h, by
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _standing_kps_for_bbox(bbox: np.ndarray) -> np.ndarray:
    """Synthesise a plausible standing pose with shoulders above hips, knees below hips."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    h = y2 - y1
    kps = np.zeros((J, 2), dtype=np.float32)
    # Shoulders: 5 left, 6 right, near top of bbox.
    kps[5] = (cx - 15, y1 + 0.25 * h)
    kps[6] = (cx + 15, y1 + 0.25 * h)
    # Hips: 11 left, 12 right, mid-bbox.
    kps[11] = (cx - 12, y1 + 0.55 * h)
    kps[12] = (cx + 12, y1 + 0.55 * h)
    # Knees: 13 left, 14 right, below hips.
    kps[13] = (cx - 14, y1 + 0.80 * h)
    kps[14] = (cx + 14, y1 + 0.80 * h)
    return kps


def _sitting_kps_for_bbox(bbox: np.ndarray) -> np.ndarray:
    """Synthesise a sitting pose: knees roughly at hip level horizontally (perpendicular to body axis)."""
    kps = _standing_kps_for_bbox(bbox)
    # Override knees to sit roughly at hip y.
    knee_y = (kps[11, 1] + kps[12, 1]) / 2
    kps[13] = (kps[11, 0] - 30, knee_y)
    kps[14] = (kps[12, 0] + 30, knee_y)
    return kps


def _build_raw_clip(per_frame_candidates: list[list[tuple[np.ndarray, np.ndarray, float]]]) -> RawClip:
    """Build a RawClip from ``[[(bbox, kps, score), ...], ...]``.

    Each inner list is one frame; pads the detection axis to the per-clip max
    with NaNs so the writer's NaN-padding contract holds.
    """
    f_count = len(per_frame_candidates)
    n_max = max(len(fc) for fc in per_frame_candidates) if f_count else 0
    if n_max == 0:
        n_max = 1  # avoid zero-shape arrays even on an all-empty clip

    kps = np.full((f_count, n_max, J, 2), np.nan, dtype=np.float32)
    bboxes = np.full((f_count, n_max, 4), np.nan, dtype=np.float32)
    scores = np.full((f_count, n_max), np.nan, dtype=np.float32)
    kp_scores = np.full((f_count, n_max, J), np.nan, dtype=np.float32)
    ndet = np.zeros(f_count, dtype=np.int64)

    for f, frame_candidates in enumerate(per_frame_candidates):
        ndet[f] = len(frame_candidates)
        for c, (bbox, kp, score) in enumerate(frame_candidates):
            bboxes[f, c] = bbox
            kps[f, c] = kp
            scores[f, c] = score
            kp_scores[f, c] = 0.9  # uniform; the heuristic doesn't read this directly

    return RawClip(kps=kps, bboxes=bboxes, scores=scores, kp_scores=kp_scores, ndet=ndet)


def _identity_normalize_joints(arr, bbox, v_height, center_align):
    """No-op stand-in for ``prepare_train_on_shuttleset.normalize_joints``.

    The picking logic does not depend on joint normalisation; tests 5-7 only
    inspect EMA state, not the output joints.
    """
    return np.asarray(arr, dtype=np.float64)


def _params(**overrides) -> StickyAnchorParams:
    """Build a StickyAnchorParams with optional field overrides."""
    return StickyAnchorParams(**overrides)


def _track_frame(raw, frame, running_positions, centres, court, resolution, params):
    count = int(raw.ndet[frame])
    return track_frame_players(
        raw.bboxes[frame, :count], raw.scores[frame, :count], raw.kps[frame, :count],
        running_positions, centres, court, resolution, params,
    )


def _track_history(raw, ctx, params):
    court = ctx.all_court_info[ctx.vid]
    centres = compute_halfcourt_centres(court)
    running = centres.copy()
    history = []
    for frame in range(len(raw.ndet)):
        _track_frame(raw, frame, running, centres, court, (1280., 720.), params)
        history.append(running.copy())
    return np.asarray(history)


# -- Test 1: Voronoi partition picks the right side --------------------------


def test_voronoi_partition_picks_correct_side():
    """Two candidates straddling the halfcourt line: the upper one fills Top, the lower one fills Bottom."""
    ctx = _identity_court_ctx()
    halfcourt_centre = np.array([[0.5, 0.25], [0.5, 0.75]])
    ema = halfcourt_centre.copy()

    # Top-side candidate at norm y = 0.20; Bottom-side at norm y = 0.80.
    top_bbox = _bbox_for(0.50, 0.20)
    bot_bbox = _bbox_for(0.50, 0.80)
    top_kps = _standing_kps_for_bbox(top_bbox)
    bot_kps = _standing_kps_for_bbox(bot_bbox)

    raw = _build_raw_clip([[(top_bbox, top_kps, 0.9), (bot_bbox, bot_kps, 0.9)]])
    res = _track_frame(raw, 0, ema, halfcourt_centre, ctx.all_court_info[ctx.vid], (1280.0, 720.0), _params())
    picks = res.detection_indices
    assert picks[TOP_ROW] == 0  # candidate 0 is the upper one
    assert picks[BOTTOM_ROW] == 1  # candidate 1 is the lower one

    # Swap input order: the assignment must not depend on candidate order.
    raw_swapped = _build_raw_clip([[(bot_bbox, bot_kps, 0.9), (top_bbox, top_kps, 0.9)]])
    res = _track_frame(raw_swapped, 0, ema, halfcourt_centre, ctx.all_court_info[ctx.vid], (1280.0, 720.0), _params())
    picks = res.detection_indices
    assert picks[TOP_ROW] == 1
    assert picks[BOTTOM_ROW] == 0


# -- Test 2: Bottom-first slot ordering with cross-slot exclusion ------------


def test_bottom_first_with_cross_slot_exclusion():
    """When both candidates fall on the Bottom side, Bottom fills with the closer one and Top stays empty."""
    ctx = _identity_court_ctx()
    halfcourt_centre = np.array([[0.5, 0.25], [0.5, 0.75]])
    ema = halfcourt_centre.copy()

    # Both candidates well into the Bottom half.
    bbox_a = _bbox_for(0.45, 0.78)
    bbox_b = _bbox_for(0.55, 0.72)
    raw = _build_raw_clip([[
        (bbox_a, _standing_kps_for_bbox(bbox_a), 0.9),
        (bbox_b, _standing_kps_for_bbox(bbox_b), 0.9),
    ]])
    res = _track_frame(raw, 0, ema, halfcourt_centre, ctx.all_court_info[ctx.vid], (1280.0, 720.0), _params())
    picks = res.detection_indices

    # Bottom slot: whichever candidate is closer to (0.5, 0.75) wins.
    # bbox_b at (0.55, 0.72) is closer than bbox_a at (0.45, 0.78).
    assert picks[BOTTOM_ROW] == 1
    # Top slot: both candidates are on the Bottom side of the Voronoi partition,
    # and the Bottom-picked one is excluded from Top consideration. Either Top
    # ends up empty or it fills with the OTHER bottom-side candidate; the test
    # pins the "no Top pick" outcome that the implementation produces.
    assert picks[TOP_ROW] == -1


# -- Test 3: Sitting tiebreaker with fallback --------------------------------


def test_sitting_tiebreaker_prefers_standing_then_falls_back():
    """Two near-tied candidates: non-sitting wins. If both sit, fall back to plain argmin."""
    ctx = _identity_court_ctx()
    halfcourt_centre = np.array([[0.5, 0.25], [0.5, 0.75]])
    ema = halfcourt_centre.copy()

    # Both Top-side candidates at near-identical anchor distance, large bboxes.
    bbox_sitting = _bbox_for(0.50, 0.25, half_w=40.0, half_h=70.0)
    bbox_standing = _bbox_for(0.495, 0.255, half_w=40.0, half_h=70.0)

    sitting_kps = _sitting_kps_for_bbox(bbox_sitting)
    standing_kps = _standing_kps_for_bbox(bbox_standing)

    # Single Top-side frame: sitting cand at index 0, standing at index 1.
    # Add a Bottom-side cand so the heuristic doesn't reject the frame.
    bot_bbox = _bbox_for(0.5, 0.75)
    bot_kps = _standing_kps_for_bbox(bot_bbox)
    raw = _build_raw_clip([[
        (bbox_sitting, sitting_kps, 0.9),
        (bbox_standing, standing_kps, 0.9),
        (bot_bbox, bot_kps, 0.9),
    ]])
    # sanity_ceiling generous so the standing cand at norm y=0.255 stays eligible.
    kw = _params(tiebreaker_tol=0.05)
    res = _track_frame(raw, 0, ema, halfcourt_centre, ctx.all_court_info[ctx.vid], (1280.0, 720.0), kw)
    picks = res.detection_indices
    assert picks[TOP_ROW] == 1, "non-sitting candidate should win the tiebreaker"

    # Both candidates sitting: fallback rule reverts to plain argmin on the
    # eligible set. The bbox_sitting candidate is exactly at the Top anchor
    # (distance 0.0) and bbox_standing is ~0.007 away, so the closer one
    # wins regardless of pose.
    raw_both_sit = _build_raw_clip([[
        (bbox_sitting, _sitting_kps_for_bbox(bbox_sitting), 0.9),
        (bbox_standing, _sitting_kps_for_bbox(bbox_standing), 0.9),
        (bot_bbox, bot_kps, 0.9),
    ]])
    res = _track_frame(raw_both_sit, 0, ema, halfcourt_centre, ctx.all_court_info[ctx.vid], (1280.0, 720.0), kw)
    picks = res.detection_indices
    assert picks[TOP_ROW] == 0, "fallback should revert to argmin when all candidates sit"


# -- Test 4: Rally-presence rejection ----------------------------------------


def test_rally_presence_rejects_when_both_picks_far_off_court():
    """When both picks land far outside the generous court margin, the frame is rejected."""
    ctx = _identity_court_ctx()
    halfcourt_centre = np.array([[0.5, 0.25], [0.5, 0.75]])
    ema = halfcourt_centre.copy()

    # Allow distant picks so the two-off-court rejection is tested separately.
    far_top_bbox = _bbox_for(-0.5, -0.5)
    far_bot_bbox = _bbox_for(1.5, 1.5)
    raw = _build_raw_clip([[
        (far_top_bbox, _standing_kps_for_bbox(far_top_bbox), 0.9),
        (far_bot_bbox, _standing_kps_for_bbox(far_bot_bbox), 0.9),
    ]])
    res = _track_frame(
        raw, 0, ema, halfcourt_centre, ctx.all_court_info[ctx.vid], (1280.0, 720.0),
        _params(sanity_ceiling=5.0, generous_margin=0.15),
    )
    np.testing.assert_array_equal(res.detection_indices, [-1, -1])
    np.testing.assert_array_equal(ema, halfcourt_centre)


# -- Test 5: Full-frame failure resets EMA to halfcourt centres --------------


def test_full_frame_failure_resets_ema():
    """After a frame with zero detections, EMA for both slots is at halfcourt centres."""
    ctx = _identity_court_ctx()

    # Frame 0: a valid pick so the EMA advances to a non-centre value.
    bot_bbox = _bbox_for(0.45, 0.80)
    top_bbox = _bbox_for(0.55, 0.20)
    frame_0 = [
        (top_bbox, _standing_kps_for_bbox(top_bbox), 0.9),
        (bot_bbox, _standing_kps_for_bbox(bot_bbox), 0.9),
    ]
    # Frame 1: zero detections (full-frame failure).
    frame_1: list = []

    raw = _build_raw_clip([frame_0, frame_1])
    output = _run_clip(raw, ctx, _identity_normalize_joints, _params())
    ema_history = _track_history(raw, ctx, _params())

    assert not output.failed[0]  # frame 0 picks both slots
    assert output.failed[1]  # frame 1 zero detections

    # Frame 0 EMA must have moved off the centres (alpha=0.1, target slightly off-centre).
    halfcourt = np.array([[0.5, 0.25], [0.5, 0.75]])
    assert not np.allclose(ema_history[0], halfcourt), "EMA should advance after a successful frame"

    # Frame 1 EMA must be back at the halfcourt centres.
    np.testing.assert_allclose(ema_history[1], halfcourt, atol=1e-12)


# -- Test 6: Mixed-pick frame resets only the unpicked slot's EMA ------------


def test_mixed_pick_resets_only_unpicked_slot():
    """A frame that picks Bottom but not Top resets only Top's EMA; Bottom's EMA advances."""
    ctx = _identity_court_ctx()
    halfcourt = np.array([[0.5, 0.25], [0.5, 0.75]])

    # Frame 0: both picks land, EMA advances on both slots.
    top_bbox = _bbox_for(0.55, 0.20)
    bot_bbox = _bbox_for(0.45, 0.80)
    frame_0 = [
        (top_bbox, _standing_kps_for_bbox(top_bbox), 0.9),
        (bot_bbox, _standing_kps_for_bbox(bot_bbox), 0.9),
    ]
    # Frame 1: only a Bottom-side candidate (no Top candidate exists).
    bot_only_bbox = _bbox_for(0.50, 0.78)
    frame_1 = [
        (bot_only_bbox, _standing_kps_for_bbox(bot_only_bbox), 0.9),
    ]
    raw = _build_raw_clip([frame_0, frame_1])
    output = _run_clip(raw, ctx, _identity_normalize_joints, _params())
    ema_history = _track_history(raw, ctx, _params())

    # Frame 1: Top EMA reset to (0.5, 0.25); Bottom EMA continued to advance.
    np.testing.assert_allclose(ema_history[1, TOP_ROW], halfcourt[TOP_ROW], atol=1e-12)
    assert not np.allclose(ema_history[1, BOTTOM_ROW], halfcourt[BOTTOM_ROW]), \
        "Bottom EMA should keep advancing when Bottom is picked"
    # Frame 1 marked as a partial failure.
    assert output.failed[1]


# -- Test 7: update_gate_eps blocks EMA updates from off-court picks ---------


def test_update_gate_eps_blocks_off_court_update():
    """A pick whose normalised position is outside the gate margin does not move the EMA."""
    ctx = _identity_court_ctx()
    halfcourt = np.array([[0.5, 0.25], [0.5, 0.75]])

    # Frame 0: an off-court Bottom pick (norm y = 1.5) plus an on-court Top
    # pick. update_gate_eps is small so the off-court pick is gated out.
    # sanity_ceiling needs to be wide enough for the off-court bottom to
    # still be considered eligible.
    top_bbox = _bbox_for(0.50, 0.20)
    bot_offcourt_bbox = _bbox_for(0.50, 1.50)
    frame_0 = [
        (top_bbox, _standing_kps_for_bbox(top_bbox), 0.9),
        (bot_offcourt_bbox, _standing_kps_for_bbox(bot_offcourt_bbox), 0.9),
    ]
    raw = _build_raw_clip([frame_0])
    ema_history = _track_history(
        raw, ctx,
        _params(
            sanity_ceiling=2.0,
            generous_margin=2.0,  # disable rally-presence rejection for this test
            update_gate_eps=0.01,
        ),
    )

    # Both picks land (rally-presence is widened above), but only the
    # in-court Top pick is allowed to move its EMA. The off-court Bottom
    # pick is gated out, so Bottom EMA stays at the halfcourt centre.
    np.testing.assert_allclose(ema_history[0, BOTTOM_ROW], halfcourt[BOTTOM_ROW], atol=1e-12)
    # Top EMA must have moved (alpha=0.1, target around (0.55, 0.20)).
    assert not np.allclose(ema_history[0, TOP_ROW], halfcourt[TOP_ROW])


def test_tracking_returns_original_detection_indices_after_filtering():
    ctx = _identity_court_ctx()
    court = ctx.all_court_info[ctx.vid]
    centres = compute_halfcourt_centres(court)
    boxes = [_bbox_for(.2, .2), _bbox_for(.5, .2), _bbox_for(.5, .8)]
    raw = _build_raw_clip([[(box, _standing_kps_for_bbox(box), score)
                           for box, score in zip(boxes, (.1, .9, .9), strict=True)]])
    players = _track_frame(raw, 0, centres.copy(), centres, court, (1280., 720.), _params())
    np.testing.assert_array_equal(players.detection_indices, [1, 2])
    np.testing.assert_allclose(players.court_positions, [[.5, .2], [.5, .8]])


@pytest.mark.parametrize('case', ['empty', 'low_score', 'score_equal', 'nan_projection', 'infinite_projection'])
def test_unusable_detections_reset_positions(case):
    ctx = _identity_court_ctx()
    court = ctx.all_court_info[ctx.vid]
    centres = compute_halfcourt_centres(court)
    running = centres + .1
    box = _bbox_for(.5, .75)
    pose = _standing_kps_for_bbox(box)
    score = .1 if case == 'low_score' else .2 if case == 'score_equal' else .9
    if case == 'nan_projection':
        box[:] = np.nan
    if case == 'infinite_projection':
        court['H'][2] = 0
    raw = _build_raw_clip([[] if case == 'empty' else [(box, pose, score)]])
    with np.errstate(divide='ignore', invalid='ignore'):
        players = _track_frame(raw, 0, running, centres, court, (1280., 720.), _params())
    np.testing.assert_array_equal(players.detection_indices, [-1, -1])
    assert np.isnan(players.court_positions).all()
    assert players.standing_count == 0
    np.testing.assert_array_equal(running, centres)


def test_standing_count_survives_selection_failure():
    ctx = _identity_court_ctx()
    court = ctx.all_court_info[ctx.vid]
    centres = compute_halfcourt_centres(court)
    boxes = [_bbox_for(.5, y) for y in (.25, .5, .75)]
    raw = _build_raw_clip([[(box, _standing_kps_for_bbox(box), .9) for box in boxes]])
    params = _params(sanity_ceiling=-1.)
    players = _track_frame(raw, 0, centres.copy(), centres, court, (1280., 720.), params)
    np.testing.assert_array_equal(players.detection_indices, [-1, -1])
    assert players.standing_count == 3
    output = _run_clip(raw, ctx, _identity_normalize_joints, params)
    assert output.failed[0]
    assert not output.overcount[0]
    assert not output.pos.any()
    assert not output.joints.any()


@pytest.mark.parametrize('resolution', [(640., 360.), (1280., 720.), (1920., 1080.)])
def test_clip_and_frame_tracking_use_the_pose_resolution(resolution):
    ctx = _identity_court_ctx()
    ctx.res_df.loc[ctx.vid, ['width', 'height']] = resolution
    scale = np.asarray(resolution) / (1280., 720.)
    boxes = [_bbox_for(.5, side) * np.tile(scale, 2) for side in (.25, .75)]
    raw = _build_raw_clip([[(box, _standing_kps_for_bbox(box), .9) for box in boxes]])
    court = ctx.all_court_info[ctx.vid]
    centres = compute_halfcourt_centres(court)
    players = _track_frame(raw, 0, centres.copy(), centres, court, resolution, _params())
    np.testing.assert_array_equal(players.detection_indices, [0, 1])
    np.testing.assert_allclose(players.court_positions, [[.5, .25], [.5, .75]])
    output = _run_clip(raw, ctx, _identity_normalize_joints, _params())
    np.testing.assert_allclose(output.pos[0], [[.5, .25], [.5, .75]])
    assert not output.failed[0]
