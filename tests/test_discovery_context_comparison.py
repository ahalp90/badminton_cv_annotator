"""The discovery comparison uses exact rally totals and keeps contact metrics whole."""

from dataclasses import replace

import pytest

from annotator.outcomes.point_winner import Half
from annotator.sequence.contacts import ContactEvent, ContactSequence, stream_payload
from annotator.training.sequences import LabelledRally, RallyBoundaryPolicy
from experiments.annotator.monotrack_benchmark.benchmark_scripts.common import (
    write_json,
)
from experiments.annotator.monotrack_benchmark.writeup_scripts import (
    discovery_boundary_surveys as surveys,
)
from experiments.annotator.monotrack_benchmark.writeup_scripts import (
    metrics,
)
from experiments.annotator.monotrack_benchmark.writeup_scripts.compare_discovery_context import (
    selection_key,
)


def test_equal_rally_totals_use_player_f1_despite_percentage_rounding() -> None:
    first = {'per_seed': [{'fully_correct': count} for count in (450, 381, 391, 431, 460)],
             'mean': {'correct_player_f1': 85.5, 'contact_f1': 89.0}}
    second = {'per_seed': [{'fully_correct': count} for count in (456, 381, 391, 425, 460)],
              'mean': {'correct_player_f1': 85.0, 'contact_f1': 90.0}}
    assert selection_key(first) > selection_key(second)


def test_boundary_only_survey_matches_full_scoring_with_excluded_rallies(tmp_path, monkeypatch) -> None:
    labels = (LabelledRally('rally', (300, 400), (Half.TOP, Half.BOT)),)
    events = (ContactEvent(300, .9, Half.TOP), ContactEvent(400, .9, Half.BOT), ContactEvent(750, .9, Half.TOP))
    sequences = (ContactSequence(0, 240, 491, events[:2]), ContactSequence(1, 700, 800, events[2:]))
    stream = stream_payload(sequences, events) | {'confidence': {'0': .8, '1': .7}}
    item = {'item_id': '8', 'video_id': 8, 'fps': 30, 'spans': [{'start_frame': 270, 'end_frame': 431}]}
    windows = [(650, 850)]
    monkeypatch.setattr(surveys, 'labels_for_item', lambda _item: labels)
    monkeypatch.setattr(metrics, 'labels_for_item', lambda _item: labels)
    path = tmp_path / '8.json.gz'
    write_json(path, stream)
    actual = surveys.judge_video((item, path, windows))
    for name, policy in surveys.POLICIES.items():
        counts, judged = metrics.score_stream_for_item(item, stream, 10, windows, supplied=False, boundary_policy=policy)
        assert counts['ignored_predictions'] == 1
        assert actual[name] == [row | {'video': 8} for row in judged]
    assert actual['1_1'][0]['outcome'] == 'fully_correct'
    assert actual['2_3'][0]['outcome'] == 'fully_correct'


def test_discovery_ignores_windows_even_when_they_overlap_another_label() -> None:
    labels = (LabelledRally('first', (100, 200), (Half.TOP, Half.BOT)),
              LabelledRally('second', (400, 500), (Half.TOP, Half.BOT)))
    events = (ContactEvent(100, .9, Half.TOP), ContactEvent(200, .9, Half.BOT))
    sequence = ContactSequence(0, 100, 201, events)
    wide = replace(sequence, start_frame=0, end_frame=600)
    for policy in (RallyBoundaryPolicy(None, None), RallyBoundaryPolicy(1, 1), RallyBoundaryPolicy(2, 3)):
        narrow_result = metrics.judge_found_rallies((sequence,), labels, 10, fps=30, boundary_policy=policy)
        wide_result = metrics.judge_found_rallies((wide,), labels, 10, fps=30, boundary_policy=policy)
        assert narrow_result == wide_result
        assert wide_result[0]['outcome'] == 'fully_correct'
    empty = replace(wide, events=())
    assert metrics.judge_found_rallies((empty,), labels, 10, fps=30)[0]['outcome'] == 'no_contacts'
    assert metrics.judge_found_rallies((wide,), labels, 10, fps=30, supplied=True)[0]['outcome'] == 'merged_rallies'


@pytest.mark.parametrize('fps', (25, 30, 60))
def test_endpoint_tolerance_and_context_limits(fps: float) -> None:
    from annotator.sequence.contacts import scale_frames

    first, last = int(10 * fps), int(20 * fps)
    tolerance = scale_frames(10, fps)
    labels = (LabelledRally('rally', (first, last), (Half.TOP, Half.BOT)),)

    def outcome(start: int, end: int, policy: RallyBoundaryPolicy) -> str:
        events = (ContactEvent(start, .9, Half.TOP), ContactEvent(end, .9, Half.BOT))
        sequence = ContactSequence(0, 0, int(30 * fps), events)
        return metrics.judge_found_rallies((sequence,), labels, tolerance, fps=fps,
                                           boundary_policy=policy)[0]['outcome']

    unlimited = RallyBoundaryPolicy(None, None)
    assert outcome(first + tolerance, last - tolerance, unlimited) == 'fully_correct'
    assert outcome(first + tolerance + 1, last, unlimited) == 'part_of_rally'
    assert outcome(first, last - tolerance - 1, unlimited) == 'part_of_rally'
    for before, after in ((1, 1), (2, 3)):
        policy = RallyBoundaryPolicy(before, after)
        start, end = first - int(before * fps), last + int(after * fps)
        assert outcome(start, end, policy) in metrics.BOUNDARY_RIGHT
        assert outcome(start - 1, end, policy) == 'overlong_rally'
        assert outcome(start, end + 1, policy) == 'overlong_rally'
        assert outcome(start - 1, end + 1, unlimited) in metrics.BOUNDARY_RIGHT
