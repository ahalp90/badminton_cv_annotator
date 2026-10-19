"""Check frozen populations and scoring of actual heuristic rally boundaries."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from annotator.config import BaseAnnotatorConfig
from annotator.outcomes.point_winner import Half
from annotator.run_video import AnnotatorResult, empty_result
from annotator.sequence.contacts import ContactEvent, ContactSequence
from annotator.training.sequences import LabelledRally
from dataset_builder.vision import load_json_gz
from experiments.annotator.heuristic_only_tune import data as search_data
from experiments.annotator.old_court_regression.retrain import stream_payload


def _video(rallies: tuple[LabelledRally, ...]) -> search_data.SearchVideo:
    return search_data.SearchVideo('example', 'ShuttleSet', 'train', 30.0, (1920.0, 1080.0), rallies)


@pytest.mark.parametrize(
    ('final_half', 'count', 'expected'),
    [
        (Half.TOP, 3, ['Top', 'Bot', 'Top']),
        (Half.TOP, 2, ['Bot', 'Top']),
        (Half.BOT, 2, ['Top', 'Bot']),
        (None, 2, [None, None]),
    ],
)
def test_heuristic_stream_uses_fitted_final_striker(
    final_half: Half | None, count: int, expected: list[str | None],
) -> None:
    frames = [10, 20, 30][:count]
    result = empty_result([(0, 40)], [])._replace(
        filtered_by_rally={0: frames}, striker_halves=[final_half],
    )
    stream = search_data.heuristic_stream(result)
    assert [event[2] for event in stream['events']] == expected
    assert stream['sequences'] == [{
        'span_id': 0, 'start_frame': 0, 'end_frame': 40, 'events': stream['events'],
    }]


def test_original_videos_keep_the_frozen_split(tmp_path: Path) -> None:
    rows = load_json_gz(search_data.TRACKED_SPLIT)['videos']
    master = tmp_path / 'shots.csv'
    with master.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['vid', 'set_id', 'rally', 'frame_num', 'player_side'])
        for row in rows:
            writer.writerow([row['video_id'], 1, 1, 20, 'Bottom'])
            writer.writerow([row['video_id'], 1, 1, 10, 'Top'])
    videos = search_data.load_original_videos(master)
    validation = {video.identity for video in videos if video.split == 'validation'}
    assert validation == {'sset_18', 'sset_22', 'sset_24', 'sset_25', 'sset_30', 'sset_31', 'sset_39', 'sset_40'}
    assert sum(video.split == 'train' for video in videos) == 32
    assert videos[0].rallies == (LabelledRally('1:1', (10, 20), (Half.TOP, Half.BOT)),)


def test_failed_empty_stream_keeps_all_label_denominators() -> None:
    video = _video((LabelledRally('rally', (10, 20), (Half.TOP, Half.BOT)),))
    scores = search_data.score_stream(video, stream_payload((), ()))
    for counts in scores.values():
        assert counts == {
            'labelled_rallies': 1, 'labelled_contacts': 2, 'predicted_contacts': 0,
            'matched_contacts': 0, 'correct_side_contacts': 0, 'fully_correct_rallies': 0,
        }
        rates = search_data.add_rates(counts)
        assert rates['recall'] == 0.0
        assert rates['fully_correct_rallies_percent'] == 0.0


def test_merged_rallies_match_contacts_without_counting_a_complete_rally() -> None:
    video = _video((
        LabelledRally('first', (10, 20), (Half.TOP, Half.BOT)),
        LabelledRally('second', (50, 60), (Half.TOP, Half.BOT)),
    ))
    events = tuple(ContactEvent(frame, 1.0, side) for frame, side in (
        (10, Half.TOP), (20, Half.BOT), (50, Half.TOP), (60, Half.BOT),
    ))
    stream = stream_payload((ContactSequence(0, 0, 70, events),), events)
    for counts in search_data.score_stream(video, stream).values():
        assert counts['matched_contacts'] == 4
        assert counts['correct_side_contacts'] == 4
        assert counts['fully_correct_rallies'] == 0


def test_split_rally_matches_contacts_without_counting_a_complete_rally() -> None:
    video = _video((LabelledRally('rally', (10, 20), (Half.TOP, Half.BOT)),))
    events = (ContactEvent(10, 1.0, Half.TOP), ContactEvent(20, 1.0, Half.BOT))
    sequences = (ContactSequence(0, 0, 15, events[:1]), ContactSequence(1, 15, 30, events[1:]))
    for counts in search_data.score_stream(video, stream_payload(sequences, events)).values():
        assert counts['matched_contacts'] == 2
        assert counts['fully_correct_rallies'] == 0


def test_contacts_outside_all_rallies_still_count_against_precision() -> None:
    video = _video((LabelledRally('rally', (10, 20), (Half.TOP, Half.BOT)),))
    own = (ContactEvent(10, 1.0, Half.TOP), ContactEvent(20, 1.0, Half.BOT))
    events = (*own, ContactEvent(100, 1.0, Half.TOP))
    stream = stream_payload((ContactSequence(0, 0, 30, own),), events)
    for counts in search_data.score_stream(video, stream).values():
        assert counts['predicted_contacts'] == 3
        assert counts['matched_contacts'] == 2
        assert counts['fully_correct_rallies'] == 1
        assert search_data.add_rates(counts)['precision'] == 2 / 3


def test_tolerances_scale_with_video_fps() -> None:
    video = search_data.SearchVideo(
        'example', 'ShuttleSet', 'validation', 60.0, (1920.0, 1080.0),
        (LabelledRally('rally', (30,), (Half.TOP,)),),
    )
    events = (ContactEvent(42, 1.0, Half.TOP),)
    stream = stream_payload((ContactSequence(0, 0, 60, events),), events)
    scores = search_data.score_stream(video, stream)
    assert scores['5']['matched_contacts'] == 0
    assert scores['10']['matched_contacts'] == 1


def test_load_inputs_rejects_labels_beyond_the_track(monkeypatch: pytest.MonkeyPatch) -> None:
    video = _video((LabelledRally('rally', (20,), (Half.TOP,)),))
    track = np.zeros((20, 3), dtype=np.float32)
    monkeypatch.setattr(search_data, 'load_npy_xz', lambda _path: track)
    with pytest.raises(ValueError, match='label outside the video'):
        search_data.load_original_inputs(video, Path('/stages'), Path('/courts'))


def test_run_configuration_discovers_boundaries_without_loading_models(monkeypatch: pytest.MonkeyPatch) -> None:
    base = BaseAnnotatorConfig()
    inputs = SimpleNamespace(track=object(), pose=SimpleNamespace(bboxes=1, scores=2, kps=3, ndet=4))
    recorded = {}

    def run(*positional: Any, **keyword: Any) -> AnnotatorResult:
        recorded.update(keyword)
        return empty_result([], [])

    monkeypatch.setattr(search_data, 'run_video', run)
    monkeypatch.setattr(search_data, 'run_video_kwargs', lambda _inputs: {'fps': 30.0})
    assert search_data.run_configuration(inputs, base) == {'sequences': [], 'events': []}
    assert recorded['base'] is base
    assert recorded['heuristic_only'] is True
    assert recorded['models'] is None
    assert 'spans' not in recorded
    assert 'contacts' not in recorded
