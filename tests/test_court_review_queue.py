"""Review decisions survive restarting independently of corner coordinates."""
import csv
import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from court_annotation.review_queue import ReviewItem, ReviewQueue, load_candidates


def test_change_and_court_are_distinct_and_resume(tmp_path):
    path = tmp_path / 'review.json.gz'
    queue = ReviewQueue(path, 'match.mp4', [ReviewItem(0, 'Initial'), ReviewItem(100, 'Movement')])
    queue.annotated(20)
    assert queue.move(1, pending_only=True) == 100
    queue.mark_change(105)
    queue.annotated(140)
    resumed = ReviewQueue(path, 'match.mp4', [ReviewItem(0, 'Initial'), ReviewItem(100, 'Movement')])
    assert resumed.markers == [105]
    assert resumed.items[1].change_frame == 105
    assert resumed.items[1].annotation_frame == 140
    assert resumed.items[0].status == 'annotated'
    assert resumed.items[0].change_frame is None
    assert resumed.remove_change(105)
    assert resumed.items[1].status == 'pending'


def test_uncertain_revisited_and_optional_decisions_preserved(tmp_path):
    path = tmp_path / 'review.json.gz'
    queue = ReviewQueue(path, 'match.mp4', [ReviewItem(0, 'Initial'), ReviewItem(100, 'Uncertain')])
    queue.decide('unchanged')
    queue.move(1)
    queue.decide('uncertain')
    filtered = ReviewQueue(path, 'match.mp4', [ReviewItem(0, 'Initial')])
    filtered.save()
    resumed = ReviewQueue(path, 'match.mp4', [ReviewItem(0, 'Initial'), ReviewItem(100, 'Uncertain')])
    assert resumed.index == 1
    assert resumed.current.status == 'uncertain'


def test_candidate_merge_and_frame_conversion(tmp_path):
    path = tmp_path / 'candidates.csv.gz'
    with gzip.open(path, 'wt') as output:
        writer = csv.writer(output)
        writer.writerow(['video', 'end_seconds', 'flag'])
        writer.writerows([['match.mp4', 10, 'camera_movement_candidate'],
                         ['match.mp4', 12, 'drift_candidate'],
                         ['match.mp4', 25, 'uncertain_transition']])
    items = load_candidates(path, 'match.mp4', 30, 1000)
    assert [item.frame for item in items] == [0, 285]
    assert len(load_candidates(path, 'match.mp4', 30, 1000, True)) == 3


def test_redecision_removes_confirmed_marker(tmp_path):
    queue = ReviewQueue(tmp_path / 'r.json.gz', 'm.mp4', [ReviewItem(0, 'Initial')])
    queue.mark_change(50)
    queue.decide('unchanged')
    assert queue.markers == []
    assert queue.current.change_frame is None


def test_manual_marker_does_not_decide_selected_candidate(tmp_path):
    queue = ReviewQueue(tmp_path / 'r.json.gz', 'm.mp4', [ReviewItem(0, 'Initial')])
    queue.mark_change(500, associate=False)
    assert queue.current.status == 'pending'
    assert queue.current.change_frame is None
    assert queue.markers == [500]


def test_previous_overlay_never_uses_closer_future_court():
    from court_annotation.annotate_court_corners_offframe import previous_court

    points = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    assert previous_court({20: points, 101: points}, 100) == (20, points)
    assert previous_court({101: points}, 100) is None


def test_fresh_annotation_output_has_empty_saved_court_list(tmp_path):
    from court_annotation.annotate_court_corners_offframe import read_saved_courts

    assert read_saved_courts(tmp_path / 'new' / 'corners.csv', 'match.mp4') == {}


def test_reflag_keeps_later_court_replaces_marker_and_resumes(tmp_path):
    path = tmp_path / 'review.json.gz'
    queue = ReviewQueue(path, 'm.mp4', [ReviewItem(0, 'Initial')])
    queue.mark_change(50)
    queue.annotated(100)
    queue.mark_change(60)
    assert queue.markers == [60]
    assert queue.current.annotation_frame == 100
    queue.mark_change(100)
    assert queue.current.annotation_frame == 100
    resumed = ReviewQueue(path, 'm.mp4', [ReviewItem(0, 'Initial')])
    assert resumed.markers == [100]
    assert resumed.current.annotation_frame == 100
    assert not resumed.needs_review(resumed.current)
    resumed.mark_change(101)
    assert resumed.current.annotation_frame is None


def test_reflag_preserves_standalone_and_shared_hidden_markers(tmp_path):
    path = tmp_path / 'review.json.gz'
    queue = ReviewQueue(path, 'm.mp4', [ReviewItem(0, 'Initial'), ReviewItem(50, 'Movement')])
    queue.mark_change(20, associate=False)
    queue.mark_change(20)
    queue.mark_change(30)
    assert sorted(queue.markers) == [20, 30]
    queue.move(1)
    queue.mark_change(30)
    filtered = ReviewQueue(path, 'm.mp4', [ReviewItem(0, 'Initial')])
    filtered.mark_change(40)
    assert sorted(filtered.markers) == [20, 30, 40]
    assert filtered.next_change(20) == 30
    assert filtered.next_change(40) == 20
    resumed = ReviewQueue(path, 'm.mp4', [ReviewItem(0, 'Initial'), ReviewItem(50, 'Movement')])
    assert resumed.standalone_markers == [20]
    assert resumed.items[1].change_frame == 30


def test_periods_notes_and_shakes_resume_from_a_file_without_them(tmp_path):
    import json

    import pytest

    path = tmp_path / 'review.json.gz'
    with gzip.open(path, 'wt') as output:
        json.dump({'version': 1, 'video': 'm.mp4', 'items': [], 'camera_changes': []}, output)
    queue = ReviewQueue(path, 'm.mp4', [ReviewItem(0, 'Initial')])
    assert (queue.periods, queue.notes) == ([], '')
    queue.start_period('play', 100)
    with pytest.raises(ValueError, match='before its start'):
        queue.end_period('play', 99)
    queue.end_period('play', 400)
    with pytest.raises(ValueError, match='already inside'):
        queue.start_period('play', 400)
    queue.start_period('play', 50)
    with pytest.raises(ValueError, match='swallow'):
        queue.end_period('play', 100)
    queue.remove_period(queue.open_period('play'))
    queue.add_shake(120)
    queue.start_period('jitter', 150)
    queue.decide('unchanged')
    queue.set_notes('Doubles match: exclude from benchmarking')
    resumed = ReviewQueue(path, 'm.mp4', [ReviewItem(0, 'Initial')])
    assert resumed.periods == [{'kind': 'play', 'start': 100, 'end': 400}, {'kind': 'jitter', 'start': 120, 'end': 120},
                               {'kind': 'jitter', 'start': 150, 'end': None}]
    assert resumed.open_period('jitter') is not None and resumed.open_period('play') is None
    assert resumed.notes == 'Doubles match: exclude from benchmarking'
    assert not resumed.needs_review(resumed.current)
