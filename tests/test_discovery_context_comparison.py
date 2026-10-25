"""The discovery comparison uses exact rally totals and keeps contact metrics whole."""

from annotator.outcomes.point_winner import Half
from annotator.sequence.contacts import ContactEvent, ContactSequence, stream_payload
from annotator.training.sequences import LabelledRally
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
    surveys.write_json(path, stream)
    actual = surveys.judge_video((item, path, windows))
    for name, policy in surveys.POLICIES.items():
        counts, judged = metrics.score_stream_for_item(item, stream, 10, windows, supplied=False, boundary_policy=policy)
        assert counts['ignored_predictions'] == 1
        assert actual[name] == [row | {'video': 8} for row in judged]
    assert actual['1_1'][0]['outcome'] == 'overlong_rally'
    assert actual['2_3'][0]['outcome'] == 'fully_correct'
