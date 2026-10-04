"""Paired coverage includes losses within already-incorrect rallies."""

import pytest

from annotator.outcomes.point_winner import Half
from annotator.training.sequences import LabelledRally
from experiments.annotator.good_court_refit.compare import (
    compare_video,
    expected_videos,
)


def stream(events: list[list]) -> dict:
    return {'events': events, 'sequences': [
        {'span_id': 0, 'start_frame': 0, 'end_frame': 200, 'events': events},
    ]}


def test_contact_and_side_losses_are_visible_without_correct_rally_loss() -> None:
    labels = (LabelledRally('r1', (20, 60, 100), (Half.TOP, Half.BOT, Half.TOP)),)
    left = stream([[20, 0.9, 'Top'], [60, 0.9, 'Bot']])
    right = stream([[20, 0.9, 'Bot'], [100, 0.9, 'Top']])
    result = compare_video(left, right, labels, 30)
    assert result['changes']['fully_correct_rallies'] == {'gained': [], 'lost': []}
    assert result['changes']['timing'] == {'gained': [('r1', 100)], 'lost': [('r1', 60)]}
    assert result['changes']['correct_side'] == {'gained': [('r1', 100)], 'lost': [('r1', 20), ('r1', 60)]}
    assert result['scores']['left']['reached_but_not_correct_rallies'] == 1
    assert result['scores']['right']['reached_but_not_correct_rallies'] == 1


def test_missed_rallies_and_unknown_sides_remain_explicit() -> None:
    labels = (LabelledRally('r1', (20,), (None,)), LabelledRally('r2', (300,), (Half.TOP,)))
    result = compare_video(stream([[20, 0.9, 'Top']]), {'events': [], 'sequences': []}, labels, 30)
    assert result['scores']['left']['correct_side_contacts'] == 0
    assert result['scores']['left']['wholly_missed_rallies'] == 1
    assert result['scores']['right']['wholly_missed_rallies'] == 2
    assert result['scores']['left']['whole_rally_unjudgeable'] == 1


@pytest.mark.parametrize(('split', 'count'), [('validation', 8), ('test', 46)])
def test_pinned_comparison_populations(split: str, count: int) -> None:
    videos = expected_videos(split)
    assert len(videos) == count
    assert '15' not in videos
    if split == 'test':
        assert '53' in videos


def test_saved_comparison_requires_complete_videos_and_matching_clocks(tmp_path, monkeypatch) -> None:
    from dataset_builder.vision import save_json_gz
    from experiments.annotator.good_court_refit import compare

    monkeypatch.setattr(compare, 'expected_videos', lambda _split: ('8',))
    labels = tmp_path / 'shared/labels/test'
    labels.mkdir(parents=True)
    (labels / '8.csv').write_text('rally_id,frame,side\nr1,20,Top\n')
    for arm in ('base', 'veto'):
        root = tmp_path / arm / 'eval/test/8'
        save_json_gz(root / 'scores.json.gz', {'fps': 30, 'frame_count': 200})
        save_json_gz(root / 'stream.json.gz', stream([[20, 0.9, 'Top']]))
    report = compare.compare_saved(tmp_path, 'test', 'base', 'veto')
    assert report['totals']['left']['fully_correct_rallies'] == 1
    assert report['totals']['right']['contact_recall'] == 1
    assert report['changes']['timing'] == {'gained': 0, 'lost': 0}

    save_json_gz(tmp_path / 'veto/eval/test/8/scores.json.gz', {'fps': 60, 'frame_count': 200})
    with pytest.raises(ValueError, match='clock or length'):
        compare.compare_saved(tmp_path, 'test', 'base', 'veto')
    (tmp_path / 'veto/eval/test/15').mkdir()
    with pytest.raises(ValueError, match='extra='):
        compare.compare_saved(tmp_path, 'test', 'base', 'veto')
