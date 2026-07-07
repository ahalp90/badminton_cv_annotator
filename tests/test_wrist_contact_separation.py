"""Unit tests for the A0 wrist-contact separation measurement helpers.

Synthetic only, no real data. Covers the four pieces that carry the logic:
window/local mapping (incl. multiple true contacts), source-space candidate
labelling (incl. a GT contact just outside the window), wrist-distance geometry
(NaN padding, ndet 0, invisible shuttle, out-of-range), and the summary's
precision/recall arithmetic.
"""
import numpy as np

from scripts.wrist_contact_separation import (
    contacts_in_window,
    format_summary,
    frame_measurement,
    matched_within,
    min_wrist_distance,
    safe_ratio,
)


# ---------------------------------------------------------------------------
# 1. Window / local mapping, including the multi-true-contact case
# ---------------------------------------------------------------------------
def test_contacts_in_window_multi_and_bounds():
    # Window is start-inclusive, end-exclusive. 99 is before the start, 150 is
    # the exclusive end, 152 is past it; 105 and 148 are the two in-window
    # strokes (neighbours sharing the overlapping window).
    vid_frames = np.array([99, 105, 148, 150, 152])
    in_window = contacts_in_window(vid_frames, start_f=100, end_f=150)
    assert in_window.tolist() == [105, 148]

    # Clip-local index is source - start_f.
    locals_ = (in_window - 100).tolist()
    assert locals_ == [5, 48]


def test_contacts_in_window_inclusive_start_exclusive_end():
    vid_frames = np.array([100, 149, 150])
    in_window = contacts_in_window(vid_frames, start_f=100, end_f=150)
    # 100 sits on the inclusive start, 149 is inside, 150 is the exclusive end.
    assert in_window.tolist() == [100, 149]


def test_contacts_in_window_empty():
    vid_frames = np.array([10, 20, 30])
    assert contacts_in_window(vid_frames, start_f=100, end_f=150).tolist() == []


# ---------------------------------------------------------------------------
# 2. Source-space candidate labelling truth table
# ---------------------------------------------------------------------------
def test_matched_within_truth_table_incl_gt_outside_window():
    # Window would be [100, 150); GT 152 sits just past the exclusive end, so it
    # is NOT an in-window contact...
    vid_gt = np.array([105, 148, 152])
    assert contacts_in_window(vid_gt, 100, 150).tolist() == [105, 148]

    # ...but a candidate at source 151 still matches it, because labelling is in
    # source space against ALL of the video's GT, not just the in-window ones
    # (the post-hit tail can clip the next stroke).
    cand_sources = np.array([106, 151, 135])
    matched = matched_within(cand_sources, vid_gt, tolerance=2)
    assert matched.tolist() == [True, True, False]


def test_matched_within_tolerance_boundary():
    reference = np.array([105])
    # Gap exactly equal to the tolerance matches (<=); one past it does not.
    on_edge = matched_within(np.array([107]), reference, tolerance=2)
    just_over = matched_within(np.array([108]), reference, tolerance=2)
    assert on_edge.tolist() == [True]
    assert just_over.tolist() == [False]


def test_matched_within_empty_inputs():
    assert matched_within(np.array([]), np.array([1, 2]), 2).tolist() == []
    # No references means nothing can match.
    assert matched_within(np.array([1, 2]), np.array([]), 2).tolist() == [False, False]


# ---------------------------------------------------------------------------
# 3. Wrist-distance geometry
# ---------------------------------------------------------------------------
def test_min_wrist_distance_picks_nearest_over_dets_and_wrists():
    # width 1000, height 500 -> shuttle norm (0.5, 0.5) is pixel (500, 250).
    shuttle = np.array([0.5, 0.5])
    # det0: both wrists 0.1 away. det1: left wrist NaN, right wrist on the shuttle.
    wrists = np.array([
        [[600.0, 250.0], [500.0, 300.0]],   # -> dists 0.1, 0.1
        [[np.nan, np.nan], [500.0, 250.0]],  # -> NaN (skipped), 0.0
    ])
    dist = min_wrist_distance(wrists, shuttle, width=1000, height=500)
    assert dist == 0.0


def test_min_wrist_distance_single_detection_value():
    shuttle = np.array([0.5, 0.5])
    wrists = np.array([[[600.0, 250.0], [500.0, 300.0]]])  # both 0.1 away
    dist = min_wrist_distance(wrists, shuttle, width=1000, height=500)
    assert dist is not None
    assert abs(dist - 0.1) < 1e-9


def test_min_wrist_distance_ndet_zero_and_all_nan():
    shuttle = np.array([0.5, 0.5])
    assert min_wrist_distance(np.empty((0, 2, 2)), shuttle, 1000, 500) is None
    all_nan = np.full((1, 2, 2), np.nan)
    assert min_wrist_distance(all_nan, shuttle, 1000, 500) is None


def _tiny_clip_arrays():
    """A 4-frame clip: frame layout drives the frame_measurement cases below."""
    width, height = 1000, 500
    # track: [x_norm, y_norm, visibility]; frame 2 is invisible (visibility 0).
    track = np.array([
        [0.5, 0.5, 1.0],
        [0.5, 0.5, 1.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 1.0],
    ])
    n_max = 2
    raw_kps = np.full((4, n_max, 17, 2), np.nan, dtype=float)
    # Frame 0: one detection with left wrist on the shuttle (500, 250).
    raw_kps[0, 0, 9] = [500.0, 250.0]
    raw_kps[0, 0, 10] = [600.0, 250.0]
    # Frame 1: one detection, both wrists 0.1 away.
    raw_kps[1, 0, 9] = [600.0, 250.0]
    raw_kps[1, 0, 10] = [500.0, 300.0]
    # Frame 3: a detection exists but its wrists are NaN.
    raw_kps[3, 0, 9] = [np.nan, np.nan]
    raw_kps[3, 0, 10] = [np.nan, np.nan]
    ndet = np.array([1, 1, 0, 1], dtype=np.int8)
    return track, raw_kps, ndet, width, height


def test_frame_measurement_visible_with_wrist():
    track, raw_kps, ndet, width, height = _tiny_clip_arrays()
    visible, n_det, dist = frame_measurement(0, track, raw_kps, ndet, width, height)
    assert visible is True
    assert n_det == 1
    assert dist == 0.0


def test_frame_measurement_invisible_shuttle():
    track, raw_kps, ndet, width, height = _tiny_clip_arrays()
    # Frame 2 has visibility 0: shuttle unmeasured, distance blank, ndet kept.
    visible, n_det, dist = frame_measurement(2, track, raw_kps, ndet, width, height)
    assert visible is False
    assert n_det == 0
    assert dist is None


def test_frame_measurement_ndet_zero_still_reports_count():
    track, raw_kps, ndet, width, height = _tiny_clip_arrays()
    # Frame 1 is visible with one detection at 0.1; sanity on the value.
    visible, n_det, dist = frame_measurement(1, track, raw_kps, ndet, width, height)
    assert visible is True and n_det == 1
    assert abs(dist - 0.1) < 1e-9


def test_frame_measurement_detection_with_nan_wrists():
    track, raw_kps, ndet, width, height = _tiny_clip_arrays()
    # Frame 3: visible shuttle, one detection, but its wrists are NaN -> blank.
    visible, n_det, dist = frame_measurement(3, track, raw_kps, ndet, width, height)
    assert visible is True and n_det == 1
    assert dist is None


def test_frame_measurement_out_of_range():
    track, raw_kps, ndet, width, height = _tiny_clip_arrays()
    # A GT local frame past a short track is unmeasurable: all blank.
    visible, n_det, dist = frame_measurement(99, track, raw_kps, ndet, width, height)
    assert visible is False
    assert n_det is None
    assert dist is None


# ---------------------------------------------------------------------------
# 4. Summary precision / recall arithmetic
# ---------------------------------------------------------------------------
def test_safe_ratio():
    assert safe_ratio(3, 4) == 0.75
    assert safe_ratio(0, 0) is None
    assert safe_ratio(2, 0) is None


def test_precision_recall_from_matched_flags():
    # Candidate precision: 2 of 3 candidates land within tolerance of a GT.
    cand_sources = np.array([10, 20, 30])
    gt_sources = np.array([10, 21])
    cand_matched = matched_within(cand_sources, gt_sources, tolerance=2)
    precision = safe_ratio(int(cand_matched.sum()), len(cand_sources))
    assert cand_matched.tolist() == [True, True, False]
    assert abs(precision - 2 / 3) < 1e-9

    # GT recall: both GT contacts have a candidate within tolerance.
    gt_matched = matched_within(gt_sources, cand_sources, tolerance=2)
    recall = safe_ratio(int(gt_matched.sum()), len(gt_sources))
    assert gt_matched.tolist() == [True, True]
    assert recall == 1.0


def test_format_summary_reports_expected_ratios():
    totals = {
        'n_gt': 2, 'n_cand': 3,
        'cand_matched_tol': 2, 'cand_matched_extra': 3,
        'gt_matched_tol': 2, 'gt_matched_extra': 2,
        'n_gt_invisible': 1,
    }
    text = format_summary(
        kind_counts={'gt': 2, 'cand': 3},
        skip_counts={'missing_file': 4},
        window_mismatch_count=1,
        dist_groups={'gt': [0.1], 'matched_cand': [], 'unmatched_cand': [0.4, 0.5]},
        totals=totals,
        tolerance=2,
    )
    assert '0.667' in text   # candidate precision @+/-2 = 2/3
    assert '1.000' in text   # recall @+/-2 = 2/2 and precision @+/-5 = 3/3
    assert '0.500' in text   # invisible fraction = 1/2
    assert 'no rows' in text  # matched_cand group is empty
    assert 'missing_file=4' in text


def test_format_summary_na_when_no_candidates():
    totals = {
        'n_gt': 0, 'n_cand': 0,
        'cand_matched_tol': 0, 'cand_matched_extra': 0,
        'gt_matched_tol': 0, 'gt_matched_extra': 0,
        'n_gt_invisible': 0,
    }
    text = format_summary(
        kind_counts={'gt': 0, 'cand': 0},
        skip_counts={},
        window_mismatch_count=0,
        dist_groups={'gt': [], 'matched_cand': [], 'unmatched_cand': []},
        totals=totals,
        tolerance=2,
    )
    assert 'n/a' in text
