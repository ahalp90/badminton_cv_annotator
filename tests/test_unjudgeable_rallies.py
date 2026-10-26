"""Unreliable rally labels must not change individual-contact assessment."""

from dataclasses import replace
from pathlib import Path

import pytest

from annotator.outcomes.point_winner import Half
from annotator.sequence.contacts import ContactEvent, ContactSequence, stream_payload
from annotator.training.sequences import LabelledRally, whole_rally_correctness
from annotator.training.workflow import load_labels
from experiments.annotator.annotator_closeout_new_courts.scripts.evaluate_saved import (
    score_video,
)
from experiments.annotator.monotrack_benchmark.writeup_scripts import metrics


@pytest.mark.parametrize('frames', [(100, 130), (100,), (110, 115, 120)])
def test_unreliable_rally_abstains_before_contact_or_boundary_errors(frames: tuple[int, ...]) -> None:
    rally = LabelledRally('first', (100, 130), (Half.TOP, Half.BOT), judgeable=False)
    events = tuple(ContactEvent(frame, .8, Half.TOP) for frame in frames)
    sequence = ContactSequence(0, 90, 121, events)
    assert whole_rally_correctness(sequence, (rally,), 30) == -1
    assert metrics.judge_found_rallies((sequence,), (rally,), 10, fps=30, supplied=True)[0]['outcome'] == (
        'unjudgeable_rally'
    )


def test_merged_rallies_remain_wrong_when_one_or_both_are_unreliable() -> None:
    first = LabelledRally('first', (100, 130), (Half.TOP, Half.BOT), judgeable=False)
    second = LabelledRally('second', (200, 230), (Half.TOP, Half.BOT))
    sequence = ContactSequence(0, 90, 240, ())
    for neighbour in (second, replace(second, judgeable=False)):
        assert whole_rally_correctness(sequence, (first, neighbour), 30) == 0


def test_rally_exclusion_preserves_contact_rows_and_matching() -> None:
    rally = LabelledRally('first', (100, 130), (Half.TOP, Half.BOT))
    events = (ContactEvent(100, .8, Half.TOP), ContactEvent(130, .8, Half.BOT))
    stream = stream_payload((ContactSequence(0, 90, 140, events),), events)
    original = score_video('fixture', 'validation', stream, (rally,), 30, 10)
    excluded = score_video('fixture', 'validation', stream, (replace(rally, judgeable=False),), 30, 10)
    assert original[1:3] == excluded[1:3]
    assert len(original[3]) == 1
    assert excluded[3] == []
    assert excluded[0][0]['outcome'] == 'unknown'


@pytest.mark.parametrize('events', [(), (ContactEvent(100, .8, Half.TOP),)])
def test_original_extent_excludes_dropped_endpoints_and_empty_detections(events: tuple) -> None:
    rally = LabelledRally('first', (130, 160), (Half.TOP, Half.BOT), judgeable=False, unjudgeable_span=(90, 201))
    sequence = ContactSequence(0, 95, 111, events)
    assert whole_rally_correctness(sequence, (rally,), 30) == -1
    assert metrics.judge_found_rallies((sequence,), (rally,), 10, fps=30)[0]['outcome'] == 'unjudgeable_rally'
    between = ContactSequence(0, 140, 145, ())
    assert whole_rally_correctness(between, (rally,), 30) == -1


def test_label_csv_can_mark_a_whole_rally_without_losing_contacts(tmp_path: Path) -> None:
    labels = tmp_path / 'labels.csv'
    labels.write_text('rally_id,frame,side,judgeable\nfirst,100,Top,0\nfirst,130,Bot,0\n')
    assert load_labels(labels, 200) == (LabelledRally('first', (100, 130), (Half.TOP, Half.BOT), judgeable=False),)
    labels.write_text('rally_id,frame,side,judgeable\nfirst,100,Top,0\nfirst,130,Bot,1\n')
    with pytest.raises(ValueError, match='constant within a rally'):
        load_labels(labels, 200)
