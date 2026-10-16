"""Check the shared scoring and the adapters that differ from released MonoTrack."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import monotrack_runner
import score
from common import read_json, write_json
from prepare_inputs import image_order, vote_free_courts
from score import score_item


def item() -> dict:
    return {'item_id': 'fixture', 'video_id': 8, 'fps': 30, 'frame_count': 240, 'resolution': [1920, 1080],
            'spans': [{'span_id': 0, 'rally_id': 'r1', 'start_frame': 70, 'end_frame': 151,
                       'label_frames': [100, 120], 'label_sides': ['Bot', 'Top']}]}


def test_discovery_counts_contacts_outside_supplied_windows() -> None:
    events = [[100, 1.0, 'Bot'], [120, 1.0, 'Top'], [200, 1.0, 'Bot']]
    result = {'boundaries': 'discovered', 'stream': {'events': events}}
    scores = score_item(item(), result)['10']
    assert scores['matched_contacts'] == 2 and scores['predicted_contacts'] == 3
    assert scores['fully_correct_rallies'] == 1
    assert scores['predicted_contacts_outside_windows'] == 1
    result['boundaries'] = 'provided'
    assert score_item(item(), result)['10']['predicted_contacts'] == 2
    assert score_item(item(), result)['10']['predicted_contacts_outside_windows'] == 0


def test_failure_summary_separates_courts_from_pose_filling(tmp_path: Path, monkeypatch) -> None:
    video = item()
    video['spans'].append({'span_id': 1, 'rally_id': 'r2', 'start_frame': 170, 'end_frame': 231,
                           'label_frames': [180, 200], 'label_sides': ['Bot', 'Top']})
    monkeypatch.setattr(score, 'load_manifest', lambda _: {'items': [video]})
    write_json(tmp_path / 'fixture.json.gz', {
        'item_id': 'fixture', 'status': 'completed', 'boundaries': 'provided', 'stream': {'events': []},
        'failed_rallies': [{'span_id': 0, 'reason': 'court_detection'},
                           {'span_id': 1, 'reason': 'ValueError', 'stage': 'pose_filling'}],
    })
    result = score.score_run(tmp_path, tmp_path)
    assert result['failed_rallies'] == 2
    assert result['failure_stages'] == {'court_detection': 1, 'pose_filling': 1}
    assert result['by_tolerance']['10']['labelled_contacts'] == 4
    assert result['by_tolerance']['10']['matched_contacts'] == 0


def test_empty_inference_keeps_all_labels() -> None:
    scores = score_item(item(), {'boundaries': 'provided', 'stream': {'events': []}})['10']
    assert scores['labelled_contacts'] == 2 and scores['labelled_rallies'] == 1
    assert scores['matched_contacts'] == 0 and scores['fully_correct_rallies'] == 0


def test_wrong_hitter_reduces_attributed_scores_and_rally_accuracy() -> None:
    events = [[100, 1.0, 'Bot'], [120, 1.0, 'Bot']]
    scores = score_item(item(), {'boundaries': 'provided', 'stream': {'events': events}})['10']
    assert (scores['precision'], scores['recall'], scores['f1']) == (1.0, 1.0, 1.0)
    assert (scores['attributed_precision'], scores['attributed_recall'], scores['attributed_f1']) == (0.5, 0.5, 0.5)
    assert scores['fully_correct_rallies_percent'] == 0.0
    events[1][2] = 'Top'
    corrected = score_item(item(), {'boundaries': 'provided', 'stream': {'events': events}})['10']
    assert corrected['attributed_f1'] == 1.0
    assert corrected['fully_correct_rallies_percent'] == 100.0


def test_corner_adapter_only_changes_hitnet_corner_slots(monkeypatch) -> None:
    corners = np.array([[100, 100], [200, 100], [250, 300], [50, 300]])
    selected = []
    detector = SimpleNamespace()
    detector.detect_hits = lambda: selected.append(detector.court.corners[[1, 2, 0, 3]]) or ([], [])
    monkeypatch.setattr(monotrack_runner, 'trajectory', lambda shuttle, interpolate: (shuttle, interpolate))
    monotrack_runner.detect_rally(detector, corners, np.zeros((20, 2, 17, 2)), np.zeros((20, 2)), 30, True)
    np.testing.assert_array_equal(selected[0], corners[[3, 2, 0, 1]])
    assert detector.trajectory[1] is True


def test_shuttle_interpolation_switch_uses_released_trajectory() -> None:
    shuttle = np.array([[0., 0.], [10, 20], [0, 0], [30, 60], [0, 0]])
    raw = monotrack_runner.trajectory(shuttle, False)
    filled = monotrack_runner.trajectory(shuttle, True)
    np.testing.assert_array_equal(raw.X, shuttle[:, 0])
    np.testing.assert_array_equal(filled.X, [10, 10, 20, 30, 30])
    np.testing.assert_array_equal(filled.Y, [20, 20, 40, 60, 60])


def test_vote_free_courts_preserve_successes_and_failures(tmp_path: Path) -> None:
    from dataset_builder.vision import load_court_vision

    video = item()
    corners = [[600, 400], [1300, 400], [1600, 980], [300, 980]]
    mirrored = [corners[index] for index in (1, 0, 3, 2)]
    np.testing.assert_array_equal(image_order(mirrored), corners)
    scenes = [
        {'start_frame': 0, 'end_frame': 70, 'scene_valid': False, 'status': 'outside_supplied_rallies'},
        {'start_frame': 70, 'end_frame': 151, 'scene_valid': True, 'status': 'court', 'corners_native_px': mirrored},
        {'start_frame': 151, 'end_frame': 240, 'scene_valid': False, 'status': 'court_detection_failed'},
    ]
    source, target = tmp_path / 'source', tmp_path / 'target'
    write_json(source / 'court_evidence.json.gz', {'scene_records': scenes})
    vote_free_courts(video, source, target)
    result = load_court_vision(target, video_id='fixture', frame_count=240, resolution=(1920., 1080.))
    assert result.evidence.court_present.sum() == 81
    assert not result.evidence.keep_vote.any()
    assert [record.scene_valid for record in result.evidence.scene_records] == [False, True, False]
    np.testing.assert_array_equal(result.evidence.scene_records[1].corners_native_px, corners)
    payload = read_json(target / 'court_evidence.json.gz')
    assert payload['scene_records'][1]['exactly_two_count'] == 0


def test_review_clip_has_exact_frame_count(tmp_path: Path) -> None:
    import subprocess

    from review_failures import make_clip

    source, destination = tmp_path / 'source.mp4', tmp_path / 'review.mp4'
    subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'testsrc2=size=640x360:rate=30:duration=2', '-c:v', 'libx264', str(source)], check=True)
    make_clip(source, destination, {'start_frame': 5, 'end_frame': 35, 'label_frames': [10, 20],
                                   'label_sides': ['Bot', 'Top']}, 30)
    count = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                                     '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', str(destination)],
                                    text=True)
    assert int(count.strip()) == 30
