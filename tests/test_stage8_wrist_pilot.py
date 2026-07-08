"""Unit tests for the stage-C wrist-pilot pure transforms.

Synthetic only. Covers the two pieces this script adds on top of the reused
segmentation/scoring code: the three-space per-frame distance (nearest-wrist
selection per space, the striker-bbox-height division, NaN handling) and the
TRUE/JUNK split's "matched in any rally's pool" semantics.
"""
import numpy as np

from scripts.stage8_score import GtRally
from scripts.stage8_wrist_pilot import (
    FrameDistances,
    frame_distances,
    measure_frame,
    split_true_junk,
)


# ---------------------------------------------------------------------------
# 1. Three-space per-frame distance
# ---------------------------------------------------------------------------
def test_frame_distances_three_spaces_and_owner_bbox():
    # width 1000, height 500 -> shuttle norm (0.5, 0.5) is pixel (500, 250).
    shuttle = np.array([0.5, 0.5])
    # det0: far wrists. det1: left wrist 0.06 away in norm x, owns the nearest wrist.
    wrists = np.array([
        [[900.0, 450.0], [800.0, 400.0]],  # det0: norms ~0.566, ~0.424
        [[560.0, 250.0], [500.0, 350.0]],  # det1: norms 0.06 (nearest), 0.2
    ])
    # bboxes [x1, y1, x2, y2]; det1 height 400 px -> 0.8 normalised.
    bboxes = np.array([
        [0.0, 0.0, 100.0, 100.0],
        [400.0, 50.0, 600.0, 450.0],
    ])
    dist = frame_distances(wrists, bboxes, shuttle, width=1000, height=500)
    assert abs(dist.image_norm - 0.06) < 1e-9        # A0 normalised gap to det1 L
    assert abs(dist.pixels - 60.0) < 1e-9            # 60 px gap to det1 L
    assert abs(dist.bbox_height_norm - 0.06 / 0.8) < 1e-9  # scaled by det1's 0.8 height


def test_frame_distances_owner_and_pixel_use_their_own_space():
    # A wide frame (width 1000, height 100) makes normalised-nearest and
    # pixel-nearest fall on different detections, pinning that (a)/(c) use the
    # normalised nearest while (b) uses the pixel nearest.
    shuttle = np.array([0.5, 0.5])  # pixel (500, 50)
    wrists = np.array([
        [[500.0, 60.0], [500.0, 90.0]],  # detA: L norm (0, 0.1)=0.1, pixel 10
        [[560.0, 50.0], [560.0, 90.0]],  # detB: L norm (0.06, 0)=0.06, pixel 60
    ])
    bboxes = np.array([
        [400.0, 25.0, 600.0, 75.0],  # detA height 50 -> 0.5
        [500.0, 40.0, 520.0, 60.0],  # detB height 20 -> 0.2
    ])
    dist = frame_distances(wrists, bboxes, shuttle, width=1000, height=100)
    assert abs(dist.image_norm - 0.06) < 1e-9         # detB owns the normalised nearest
    assert abs(dist.pixels - 10.0) < 1e-9             # detA owns the pixel nearest
    assert abs(dist.bbox_height_norm - 0.06 / 0.2) < 1e-9  # owner is detB, not the pixel-nearest detA


def test_frame_distances_no_wrist_is_all_nan():
    shuttle = np.array([0.5, 0.5])
    empty = frame_distances(np.empty((0, 2, 2)), np.empty((0, 4)), shuttle, 1000, 500)
    assert all(np.isnan(value) for value in empty)
    all_nan = frame_distances(
        np.full((1, 2, 2), np.nan), np.array([[0.0, 0.0, 100.0, 100.0]]), shuttle, 1000, 500
    )
    assert all(np.isnan(value) for value in all_nan)


def test_measure_frame_invisible_shuttle_is_all_nan():
    track = np.array([[0.5, 0.5, 1.0], [0.5, 0.5, 0.0]])  # frame 1 invisible
    kps = np.full((2, 1, 17, 2), np.nan)
    kps[0, 0, 9] = [500.0, 250.0]  # left wrist on the shuttle at frame 0
    bboxes = np.array([[[400.0, 100.0, 600.0, 400.0]], [[400.0, 100.0, 600.0, 400.0]]])
    ndet = np.array([1, 1], dtype=np.int8)

    visible = measure_frame(0, track, kps, bboxes, ndet, 1000, 500)
    assert abs(visible.image_norm - 0.0) < 1e-9
    hidden = measure_frame(1, track, kps, bboxes, ndet, 1000, 500)
    assert isinstance(hidden, FrameDistances)
    assert all(np.isnan(value) for value in hidden)


# ---------------------------------------------------------------------------
# 2. TRUE / JUNK split ("matched in any rally's pool")
# ---------------------------------------------------------------------------
def test_split_true_junk_matches_per_rally_pool():
    spans = [(0, 10), (10, 20)]  # frame-disjoint
    gt_rallies = [
        GtRally(set_id='set1', rally=1, stroke_frames=(2, 3)),
        GtRally(set_id='set1', rally=2, stroke_frames=(12, 13)),
    ]
    # span 0 fires at 2 (real) and 8 (junk); span 1 at 12 (real) and 18 (junk).
    contacts = [(0, 2, None), (0, 8, None), (1, 12, None), (1, 18, None)]

    true_frames, junk_frames, total_matches = split_true_junk(spans, contacts, gt_rallies, tolerance=2)
    # At +/-2 only the on-stroke candidates match; 8 and 18 fall out as junk.
    assert true_frames == [2, 12]
    assert junk_frames == [8, 18]
    assert total_matches == 2


def test_split_true_junk_loose_tolerance_pulls_in_neighbour():
    spans = [(0, 10)]
    gt_rallies = [GtRally(set_id='set1', rally=1, stroke_frames=(2, 3))]
    contacts = [(0, 2, None), (0, 7, None)]
    # At +/-5 the second stroke (3) claims candidate 7 (gap 4), so both are TRUE.
    true_frames, junk_frames, total_matches = split_true_junk(spans, contacts, gt_rallies, tolerance=5)
    assert true_frames == [2, 7]
    assert junk_frames == []
    assert total_matches == 2
