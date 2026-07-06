"""Tests for the additive doubles over-count signal in the pose lane (spec s8).

Two producers feed the windowed ``scraper.doubles_flag`` rule:

- ``_order_two_on_court`` (the live rtmlib path) now returns the in-court count
  beside its ordered-pair result, so a doubles frame (>2 in court) is
  distinguishable from an ordinary miss.
- ``sticky_anchor`` counts the in-court candidates per frame and lands the
  over-count in ``HeuristicOutput`` while still picking only two players.

The court projection is stubbed for the first group (the count logic is what is
under test, not the homography) and identity-mapped for the second, following the
house patterns in test_extract_failure_guard.py and test_sticky_anchor.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from preparing_data import prepare_train_on_shuttleset as prep
from preparing_data.heuristics.base import ClipContext, RawClip
from preparing_data.heuristics.sticky_anchor import (
    StickyAnchorParams,
    _pick_one_frame,
    _run_clip,
)
from preparing_data.prepare_train_on_shuttleset import _order_two_on_court

J = 17  # COCO 17-joint skeleton


# ---------------------------------------------------------------------------
# _order_two_on_court: the in-court count rides out beside the result
# ---------------------------------------------------------------------------


def _fake_keypoints(m: int) -> np.ndarray:
    """(m, J, 2) keypoints; content is irrelevant since check_pos_in_court is stubbed."""
    return np.zeros((m, J, 2), dtype=np.float64)


def _stub_check_pos_in_court(in_court: np.ndarray, pos: np.ndarray):
    """Return a check_pos_in_court replacement yielding a fixed (in_court, pos)."""
    def _fn(keypoints, vid, all_court_info, res_df):
        return in_court, pos
    return _fn


def test_order_under_two_detections_short_circuits_with_count(monkeypatch):
    """<2 detections: the count is the detection count and the court check never runs."""
    def _must_not_run(*args, **kwargs):
        raise AssertionError("check_pos_in_court must not run on <2 detections")

    monkeypatch.setattr(prep, "check_pos_in_court", _must_not_run)
    result, n_in_court = _order_two_on_court(
        _fake_keypoints(1), vid=11, all_court_info={}, res_df=pd.DataFrame()
    )
    assert result is None
    assert n_in_court == 1  # detection count, safely bounding the (unmeasured) in-court count


def test_order_exactly_two_in_court(monkeypatch):
    """Exactly two in court: success, count 2, ordered Top-before-Bottom."""
    in_court = np.array([True, True, False])
    # player 0 sits Bottom (y 0.8), player 1 Top (y 0.2); the flip must reorder to [1, 0].
    pos = np.array([[0.5, 0.8], [0.5, 0.2], [0.9, 0.9]])
    monkeypatch.setattr(prep, "check_pos_in_court", _stub_check_pos_in_court(in_court, pos))

    result, n_in_court = _order_two_on_court(
        _fake_keypoints(3), vid=11, all_court_info={}, res_df=pd.DataFrame()
    )
    assert n_in_court == 2
    assert result is not None
    in_court_pid, pos_out = result
    assert list(in_court_pid) == [1, 0]  # Top (player 1) before Bottom (player 0)
    assert pos_out is pos  # full array returned, caller does its own indexing


def test_order_overcount_fails_but_reports_count(monkeypatch):
    """Four in court: not exactly two so result is None, but count 4 is doubles evidence."""
    in_court = np.array([True, True, True, True])
    pos = np.array([[0.5, 0.2], [0.5, 0.3], [0.5, 0.7], [0.5, 0.8]])
    monkeypatch.setattr(prep, "check_pos_in_court", _stub_check_pos_in_court(in_court, pos))

    result, n_in_court = _order_two_on_court(
        _fake_keypoints(4), vid=11, all_court_info={}, res_df=pd.DataFrame()
    )
    assert result is None
    assert n_in_court == 4


# ---------------------------------------------------------------------------
# sticky_anchor: over-count lands in HeuristicOutput while picks stay two
# ---------------------------------------------------------------------------
# Fixtures mirror test_sticky_anchor.py: an identity homography over [0, 1280]x
# [0, 720] so normalise collapses to (px / 1280, py / 720).


def _identity_court_ctx(vid: int = 1) -> ClipContext:
    court_info = {
        "H": np.eye(3, dtype=np.float64),
        "border_L": 0.0,
        "border_R": 1280.0,
        "border_U": 0.0,
        "border_D": 720.0,
    }
    res_df = pd.DataFrame({"width": [1280], "height": [720]}, index=[vid])
    return ClipContext(vid=vid, all_court_info={vid: court_info}, res_df=res_df)


def _bbox_for(norm_x: float, norm_y: float, half_w: float = 30.0, half_h: float = 60.0) -> np.ndarray:
    """Bbox whose bottom-centre projects to ``(norm_x, norm_y)`` in normalised court coords."""
    cx = norm_x * 1280.0
    by = norm_y * 720.0
    return np.array([cx - half_w, by - 2 * half_h, cx + half_w, by], dtype=np.float32)


def _standing_kps_for_bbox(bbox: np.ndarray) -> np.ndarray:
    """Plausible standing pose: shoulders above hips, knees below hips."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    h = y2 - y1
    kps = np.zeros((J, 2), dtype=np.float32)
    kps[5] = (cx - 15, y1 + 0.25 * h)   # shoulders
    kps[6] = (cx + 15, y1 + 0.25 * h)
    kps[11] = (cx - 12, y1 + 0.55 * h)  # hips
    kps[12] = (cx + 12, y1 + 0.55 * h)
    kps[13] = (cx - 14, y1 + 0.80 * h)  # knees
    kps[14] = (cx + 14, y1 + 0.80 * h)
    return kps


def _build_single_frame_raw(candidates: list[tuple[np.ndarray, np.ndarray, float]]) -> RawClip:
    """One-frame RawClip from ``[(bbox, kps, score), ...]``; detection axis padded with NaN."""
    n_max = len(candidates)
    kps = np.full((1, n_max, J, 2), np.nan, dtype=np.float32)
    bboxes = np.full((1, n_max, 4), np.nan, dtype=np.float32)
    scores = np.full((1, n_max), np.nan, dtype=np.float32)
    kp_scores = np.full((1, n_max, J), np.nan, dtype=np.float32)
    ndet = np.array([n_max], dtype=np.int64)
    for c, (bbox, kp, score) in enumerate(candidates):
        bboxes[0, c] = bbox
        kps[0, c] = kp
        scores[0, c] = score
        kp_scores[0, c] = 0.9
    return RawClip(kps=kps, bboxes=bboxes, scores=scores, kp_scores=kp_scores, ndet=ndet)


def _identity_normalize_joints(arr, bbox, v_height, center_align):
    """No-op stand-in for prepare_train_on_shuttleset.normalize_joints (joints unused here)."""
    return np.asarray(arr, dtype=np.float64)


def test_sticky_anchor_overcount_flags_doubles_while_picking_two():
    """Four in-court candidates: picks stay two, but the over-count lands True."""
    ctx = _identity_court_ctx()
    halfcourt_centre = np.array([[0.5, 0.25], [0.5, 0.75]])
    ema = halfcourt_centre.copy()
    params = StickyAnchorParams()

    # Two candidates near the Top anchor, two near the Bottom: all four in court.
    top1, top2 = _bbox_for(0.50, 0.25), _bbox_for(0.50, 0.30)
    bot1, bot2 = _bbox_for(0.50, 0.75), _bbox_for(0.50, 0.70)
    raw = _build_single_frame_raw([
        (top1, _standing_kps_for_bbox(top1), 0.9),
        (top2, _standing_kps_for_bbox(top2), 0.9),
        (bot1, _standing_kps_for_bbox(bot1), 0.9),
        (bot2, _standing_kps_for_bbox(bot2), 0.9),
    ])

    # _pick_one_frame reports the in-court candidate count as its 5th element.
    res = _pick_one_frame(raw, 0, ema, halfcourt_centre, ctx, params)
    assert res is not None
    picks, _court_base_pos, _kps_f, _bboxes_f, n_in_court = res
    assert n_in_court == 4          # all four project inside the generous court
    assert picks.count(-1) == 0     # both slots still fill: exactly two picks

    # Via _run_clip the over-count lands in HeuristicOutput; the frame is not a failure.
    output, _ema_history = _run_clip(raw, ctx, _identity_normalize_joints, params)
    assert bool(output.overcount[0])
    assert not output.failed[0]


def test_sticky_anchor_two_players_no_overcount():
    """A clean two-player frame: over-count stays False."""
    ctx = _identity_court_ctx()
    params = StickyAnchorParams()

    top, bot = _bbox_for(0.50, 0.25), _bbox_for(0.50, 0.75)
    raw = _build_single_frame_raw([
        (top, _standing_kps_for_bbox(top), 0.9),
        (bot, _standing_kps_for_bbox(bot), 0.9),
    ])

    output, _ema_history = _run_clip(raw, ctx, _identity_normalize_joints, params)
    assert not output.overcount[0]
    assert not output.failed[0]
