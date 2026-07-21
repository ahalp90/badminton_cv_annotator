"""Unit tests for scene-gated sticky tracker intervals."""

import numpy as np
import pytest

from annotator.rally_segmentation import tracker_segments, wrist_contact_near
from annotator.run_video import scoring_filter
from annotator.types import ContactCandidate


def _rows(*bounds):
    return [
        {'start_frame': str(start), 'end_frame': str(end)}
        for start, end in bounds
    ]


def test_tracker_segments_reset_at_scene_row_boundary():
    assert tracker_segments(_rows((0, 3), (3, 6)), np.ones(6, dtype=bool), 6) == [
        (0, 3), (3, 6),
    ]


def test_tracker_segments_do_not_bridge_court_dropout():
    present = np.array([True, False, True])
    assert tracker_segments(_rows((0, 3)), present, 3) == [(0, 1), (2, 3)]


def test_tracker_segments_leave_holes_between_rows_unanalysed():
    assert tracker_segments(_rows((0, 2), (4, 6)), np.ones(6, dtype=bool), 6) == [
        (0, 2), (4, 6),
    ]


def test_tracker_segments_clip_rows_to_video():
    assert tracker_segments(_rows((-2, 8)), np.ones(5, dtype=bool), 5) == [(0, 5)]


def test_tracker_segments_empty_inputs_and_all_false_mask():
    assert tracker_segments([], np.ones(5, dtype=bool), 5) == []
    assert tracker_segments(_rows((0, 5)), np.zeros(5, dtype=bool), 5) == []


@pytest.mark.parametrize(
    ('rows', 'message'),
    [
        (_rows((4, 2)), 'reversed'),
        (_rows((0, 4), (3, 5)), 'overlap'),
        ([{'start_frame': 'start', 'end_frame': '5'}], 'bounds'),
    ],
)
def test_tracker_segments_reject_bad_rows(rows, message):
    with pytest.raises(ValueError, match=message):
        tracker_segments(rows, np.ones(6, dtype=bool), 6)


@pytest.mark.parametrize(
    'present',
    [np.ones((4, 1), dtype=bool), np.ones(4, dtype=np.int8), [True] * 4],
)
def test_tracker_segments_reject_bad_masks(present):
    with pytest.raises(ValueError, match='court_present'):
        tracker_segments(_rows((0, 4)), present, 4)


def test_contact_outside_tracker_segment_fails_wrist_gate_and_scoring():
    distances = np.full(4, np.inf)
    contact = ContactCandidate(0, 2, None, None, False)
    wrist_near = wrist_contact_near(distances, contact.contact_frame)
    contact = contact._replace(wrist_near=wrist_near)
    assert not np.isfinite(distances[contact.contact_frame])
    assert contact.wrist_near is False
    assert scoring_filter([contact]) == []
