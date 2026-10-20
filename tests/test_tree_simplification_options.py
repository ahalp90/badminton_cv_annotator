"""Expanded proposal coverage, raw evidence and finished-rally targets."""

from dataclasses import replace

import numpy as np

from annotator.outcomes.point_winner import Half
from annotator.sequence.boundaries import extend_boundaries
from annotator.sequence.candidates import ServeCandidate
from annotator.sequence.contacts import ContactEvent, initial_sequences
from annotator.sequence.edits import apply_choices, option_pool, sequence_edits
from annotator.sequence.features import PHYSICAL_FEATURE_NAMES, ServeEditInputs
from annotator.sequence.rally_options import RallyContext, build_rally_options
from annotator.sequence.refine import OptionPool
from annotator.sequence.sides import alternate_sides
from annotator.training.sequences import LabelledRally, SequenceTrainingVideo
from annotator.training.workflow import whole_rally_correctness
from experiments.annotator.tree_simplification.options import (
    FEATURE_NAMES,
    expand_pool,
    finish_in_initial_context,
    sequence_key,
)


def example_video() -> SequenceTrainingVideo:
    frames = (10, 35, 60, 90, 120, 150, 180, 250, 280, 330, 345, 390, 405)
    events = tuple(ContactEvent(frame, 0.55 + index * 0.04, Half.TOP if index % 2 else Half.BOT)
                   for index, frame in enumerate(frames))
    initial = initial_sequences(((0, 50), (70, 210), (240, 300), (320, 360), (380, 420)), events)
    serve = tuple(ContactEvent(frame, 0.8, Half.TOP) for frame in (52, 75))
    # The close and out-of-bound later rows exercise rejection in new combinations.
    later = tuple(ContactEvent(frame, 0.7, Half.BOT) for frame in (40, 105, 135, 149, 165, 195))
    serve_candidates = [ServeCandidate(1, event, 90, ()) for event in serve]
    edits = {}
    previous_end = -1
    for sequence in initial:
        edits[sequence.span_id] = sequence_edits(
            sequence, serve_candidates if sequence.span_id == 1 else (), events, previous_end,
        )
        previous_end = sequence.end_frame
    options = option_pool(edits, {1: later}, 30.0)
    physical = {event.frame: np.full(len(PHYSICAL_FEATURE_NAMES), event.frame, dtype=float)
                for event in (*events, *serve, *later)}
    pool = OptionPool(
        options, {1: later}, ServeEditInputs({}, np.empty((0, 0))),
        np.full((len(options), 1), np.nan), np.full((len(options), 1), np.nan), physical,
    )
    rallies = tuple(LabelledRally(str(sequence.span_id), tuple(event.frame for event in sequence.events),
                                 tuple(event.side for event in sequence.events)) for sequence in initial)
    return SequenceTrainingVideo('fixture', 'group', 30.0, initial, events, pool, rallies, 430)


def test_multiple_additions_and_deletions_preserve_all_legacy_choices() -> None:
    video = example_video()
    expanded = expand_pool(video)
    middle = [expanded.sequences[row] for row in expanded.span_rows[1]]
    frame_lists = {tuple(event.frame for event in sequence.events) for sequence in middle}
    assert (90, 105, 120, 135, 150, 180) in frame_lists
    assert (150, 180) in frame_lists
    assert (52, 60, 150, 180) in frame_lists
    keys = {(sequence.span_id, sequence_key(sequence)) for sequence in expanded.sequences}
    assert {(option.span_id, sequence_key(option.sequence)) for option in video.pool.options} <= keys
    restricted = expand_pool(video, max_edits=0, deletion_candidates=0)
    restricted_keys = {(sequence.span_id, sequence_key(sequence)) for sequence in restricted.sequences}
    assert {(option.span_id, sequence_key(option.sequence)) for option in video.pool.options} <= restricted_keys


def test_runtime_proposals_need_no_labels_and_match_training_order() -> None:
    video = example_video()
    context = RallyContext(video.fps, video.initial, video.events, video.pool, video.frame_count)
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
    expanded = expand_pool(video)
    legacy = {(option.span_id, sequence_key(option.sequence)) for option in video.pool.options}
    for sequence in expanded.sequences:
        if (sequence.span_id, sequence_key(sequence)) in legacy:
            continue
        frames = tuple(event.frame for event in sequence.events)
        previous_end = video.initial[sequence.span_id - 1].end_frame if sequence.span_id else -1
        assert sequence.start_frame >= previous_end
        assert all(sequence.start_frame <= frame < sequence.end_frame for frame in frames)
        assert np.all(np.diff(frames) > 6)
    repaired = next(sequence for sequence in expanded.sequences
                    if tuple(event.frame for event in sequence.events) == (52, 60, 150, 180))
    assert repaired.start_frame == 52
    assert not {90, 120} & {event.frame for event in repaired.events}
    assert 60 in {event.frame for event in repaired.events}


def test_features_and_order_are_deterministic_and_independent_of_labels() -> None:
    video = example_video()
    first = expand_pool(video)
    second = expand_pool(video)
    changed = expand_pool(replace(video, rallies=(LabelledRally('different', (90,), (Half.TOP,)),)))
    assert first.sequences == second.sequences == changed.sequences
    np.testing.assert_array_equal(first.matrix, second.matrix)
    np.testing.assert_array_equal(first.matrix, changed.matrix)
    assert np.any(first.targets != changed.targets)
    assert first.matrix.shape == (len(first.sequences), len(FEATURE_NAMES))
    assert first.targets.dtype == np.int8
    assert set(first.targets.tolist()) <= {-1, 0, 1}
    for before in video.initial:
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
    assert first.matrix[repaired_row, FEATURE_NAMES.index(f'new_first__{PHYSICAL_FEATURE_NAMES[0]}')] == 52


def test_local_boundary_finishing_matches_full_video_and_targets_keep_known_bad_spans() -> None:
    video = example_video()
    expanded = expand_pool(video)
    stream_frames = np.asarray([event.frame for event in video.events])
    for after in expanded.sequences:
        chosen = {sequence.span_id: sequence for sequence in video.initial}
        chosen[after.span_id] = after
        raw, events = apply_choices(video.initial, video.events, chosen)
        expected = extend_boundaries(raw, events, video.fps, frame_count=video.frame_count)
        actual = finish_in_initial_context(video, after.span_id, after, stream_frames)
        assert actual == expected[after.span_id]
        finished, _events = alternate_sides(expected, events)
        assert whole_rally_correctness(actual, video.rallies, video.fps) == whole_rally_correctness(
            finished[after.span_id], video.rallies, video.fps,
        )
    merged_labels = (
        LabelledRally('first', (90, 120), (None, None)),
        LabelledRally('second', (150, 180), (None, None)),
    )
    merged = expand_pool(replace(video, rallies=merged_labels))
    assert merged.targets[merged.span_rows[1][0]] == 0
    partial = expand_pool(replace(video, rallies=(LabelledRally('partial', (60, 90, 120), (None, None, None)),)))
    assert partial.targets[partial.span_rows[1][0]] == 0
