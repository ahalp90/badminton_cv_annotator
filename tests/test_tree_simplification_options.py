"""Full option-list coverage, raw evidence and finished-rally targets."""

from dataclasses import replace

import numpy as np

from annotator.contacts.build_features import CONTACT_FEATURE_NAMES
from annotator.rallies.build_contact_proposals import (
    RallyContext,
    build_rally_options,
    initial_sequences,
)
from annotator.training.sequences import LabelledRally, SequenceTrainingVideo
from annotator.types import ContactEvent, CourtHalf
from experiments.annotator.tree_simplification.options import (
    FEATURE_NAMES,
    expand_pool,
)


def example_video() -> SequenceTrainingVideo:
    frames = (10, 35, 60, 90, 120, 150, 180, 250, 280, 330, 345, 390, 405)
    events = tuple(ContactEvent(frame, 0.55 + index * 0.04, CourtHalf.TOP if index % 2 else CourtHalf.BOT)
                   for index, frame in enumerate(frames))
    initial = initial_sequences(((0, 50), (70, 210), (240, 300), (320, 360), (380, 420)), events)
    serve = tuple(ContactEvent(frame, 0.8, CourtHalf.TOP) for frame in (52, 75))
    # The close and out-of-bound later rows exercise rejection in new combinations.
    later = tuple(ContactEvent(frame, 0.7, CourtHalf.BOT) for frame in (40, 105, 135, 149, 165, 195))
    physical = {event.frame: np.full(len(CONTACT_FEATURE_NAMES), event.frame, dtype=float)
                for event in (*events, *serve, *later)}
    candidates = {1: tuple(sorted((*serve, *later), key=lambda event: event.frame))}
    rallies = tuple(LabelledRally(str(sequence.span_id), tuple(event.frame for event in sequence.events),
                                 tuple(event.side for event in sequence.events)) for sequence in initial)
    context = RallyContext(
        30.0, initial, events, candidates, physical, frame_count=430,
    )
    return SequenceTrainingVideo('fixture', 'group', context, rallies)


def test_propose_contact_additions_and_removals_cover_multiple_edits_and_keep_no_edit() -> None:
    video = example_video()
    options = expand_pool(video)
    middle = [options.sequences[row] for row in options.span_rows[1]]
    frame_lists = {tuple(event.frame for event in sequence.events) for sequence in middle}
    assert (90, 105, 120, 135, 150, 180) in frame_lists
    assert (150, 180) in frame_lists
    assert (52, 60, 150, 180) in frame_lists
    restricted = expand_pool(video, max_edits=0, deletion_candidates=0)
    assert restricted.sequences == video.context.initial


def test_runtime_proposals_need_no_labels_and_match_training_order() -> None:
    video = example_video()
    context = video.context
    runtime = build_rally_options(context)
    labelled = expand_pool(video)
    unlabelled = expand_pool(replace(video, rallies=()))
    assert runtime.sequences == labelled.sequences == unlabelled.sequences
    np.testing.assert_array_equal(runtime.matrix, labelled.matrix)
    np.testing.assert_array_equal(runtime.matrix, unlabelled.matrix)
    for span_id, rows in runtime.span_rows.items():
        np.testing.assert_array_equal(rows, labelled.span_rows[span_id])
    assert labelled.runtime.matrix is labelled.matrix


def test_new_combinations_obey_spacing_bounds_and_do_not_resurrect_deletions() -> None:
    video = example_video()
    options = expand_pool(video)
    for sequence in options.sequences:
        frames = tuple(event.frame for event in sequence.events)
        previous_end = video.context.initial[sequence.span_id - 1].end_frame if sequence.span_id else -1
        assert sequence.start_frame >= previous_end
        assert all(sequence.start_frame <= frame < sequence.end_frame for frame in frames)
        assert np.all(np.diff(frames) > 6)
    repaired = next(sequence for sequence in options.sequences
                    if tuple(event.frame for event in sequence.events) == (52, 60, 150, 180))
    assert repaired.start_frame == 52
    assert not {90, 120} & {event.frame for event in repaired.events}
    assert 60 in {event.frame for event in repaired.events}


def test_features_and_order_are_deterministic_and_independent_of_labels() -> None:
    video = example_video()
    first = expand_pool(video)
    second = expand_pool(video)
    changed = expand_pool(replace(video, rallies=(LabelledRally('different', (90,), (CourtHalf.TOP,)),)))
    assert first.sequences == second.sequences == changed.sequences
    np.testing.assert_array_equal(first.matrix, second.matrix)
    np.testing.assert_array_equal(first.matrix, changed.matrix)
    assert np.any(first.targets != changed.targets)
    assert first.matrix.shape == (len(first.sequences), len(FEATURE_NAMES))
    assert first.targets.dtype == np.int8
    assert set(first.targets.tolist()) <= {-1, 0, 1}
    for before in video.context.initial:
        rows = first.span_rows[before.span_id]
        assert first.sequences[rows[0]] == before
        frame_set = {event.frame for event in before.events}
        priorities = [(len(frame_set ^ {event.frame for event in first.sequences[row].events}),
                       tuple(event.frame for event in first.sequences[row].events)) for row in rows[1:]]
        assert priorities == sorted(priorities)
        assert first.matrix[rows[0], FEATURE_NAMES.index('added_count')] == 0
        assert np.isnan(first.matrix[rows[0], FEATURE_NAMES.index('added_probability_min')])
    repaired_row = next(row for row, sequence in enumerate(first.sequences)
                        if tuple(event.frame for event in sequence.events) == (52, 60, 150, 180))
    assert first.matrix[repaired_row, FEATURE_NAMES.index('added_count')] == 2
    assert first.matrix[repaired_row, FEATURE_NAMES.index('deleted_count')] == 2
    assert first.matrix[repaired_row, FEATURE_NAMES.index(f'new_first__{CONTACT_FEATURE_NAMES[0]}')] == 52


def test_proposal_targets_keep_merged_and_partial_rallies_wrong() -> None:
    video = example_video()
    merged_labels = (
        LabelledRally('first', (90, 120), (None, None)),
        LabelledRally('second', (150, 180), (None, None)),
    )
    merged = expand_pool(replace(video, rallies=merged_labels))
    assert merged.targets[merged.span_rows[1][0]] == 0
    partial = expand_pool(replace(video, rallies=(LabelledRally('partial', (60, 90, 120), (None, None, None)),)))
    assert partial.targets[partial.span_rows[1][0]] == 0


def test_supplied_bounds_reject_earlier_contacts_and_keep_original_extent() -> None:
    original = example_video()
    video = replace(original, context=replace(original.context, bounds_supplied=True))
    options = expand_pool(video)
    initial = {sequence.span_id: sequence for sequence in video.context.initial}
    for sequence in options.sequences:
        before = initial[sequence.span_id]
        assert (sequence.start_frame, sequence.end_frame) == (before.start_frame, before.end_frame)
        assert all(before.start_frame <= event.frame < before.end_frame for event in sequence.events)
    assert all(52 not in [event.frame for event in options.sequences[row].events] for row in options.span_rows[1])
