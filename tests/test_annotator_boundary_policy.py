"""Training windows and supplied-boundary scoring share frame conventions."""

from dataclasses import replace

import pytest

from annotator.outcomes.point_winner import Half
from annotator.sequence.contacts import ContactEvent, ContactSequence, scale_frames
from annotator.training.sequences import (
    LabelledRally,
    RallyBoundaryPolicy,
    judge_rally_boundaries,
    whole_rally_correctness,
)
from experiments.annotator.monotrack_benchmark.writeup_scripts.metrics import (
    judge_found_rallies,
)


@pytest.mark.parametrize('fps', [25.0, 30.0, 60.0])
@pytest.mark.parametrize('policy', [RallyBoundaryPolicy(2, 3), RallyBoundaryPolicy(5, 5)])
def test_endpoint_limits_use_final_included_frame_and_scale_with_fps(fps, policy):
    first, last = round(10 * fps), round(12 * fps)
    rally = LabelledRally('rally', (first, last), (Half.TOP, Half.BOT))
    events = (ContactEvent(first, .9, Half.TOP), ContactEvent(last, .9, Half.BOT))
    sequence = ContactSequence(0, first - round(policy.before_seconds * fps),
                               last + round(policy.after_seconds * fps) + 1, events)
    for candidate, expected in (
        (sequence, 1),
        (replace(sequence, start_frame=sequence.start_frame - 1), 0),
        (replace(sequence, end_frame=sequence.end_frame + 1), 0),
        (replace(sequence, start_frame=first, end_frame=last + 1), 1),
    ):
        assert whole_rally_correctness(candidate, (rally,), fps, boundary_policy=policy) == expected
        result = judge_found_rallies(
            (candidate,), (rally,), scale_frames(10, fps), fps=fps, boundary_policy=policy, supplied=True,
        )[0]
        assert (result['outcome'] == 'fully_correct') == bool(expected)


def test_short_inward_errors_and_neighbouring_rallies_use_shared_rule():
    rally = LabelledRally('rally', (300, 400), (Half.TOP, Half.BOT))
    sequence = ContactSequence(0, 305, 397, (ContactEvent(305, .9, Half.TOP), ContactEvent(395, .9, Half.BOT)))
    assert whole_rally_correctness(sequence, (rally,), 30) == 1
    neighbour = LabelledRally('neighbour', (420, 440), (Half.TOP, Half.BOT))
    merged = replace(sequence, end_frame=441)
    result = judge_rally_boundaries(merged, (rally, neighbour), 30, tolerance=10, policy=RallyBoundaryPolicy(5, 5))
    assert result.outcome == 'merged_rallies'
    assert whole_rally_correctness(merged, (rally, neighbour), 30, boundary_policy=RallyBoundaryPolicy(5, 5)) == 0


def test_overlong_boundary_is_wrong_even_when_hitter_labels_are_unknown():
    rally = LabelledRally('rally', (300, 400), (None, None))
    sequence = ContactSequence(0, 0, 401, (ContactEvent(300, .9, None), ContactEvent(400, .9, None)))
    assert whole_rally_correctness(sequence, (rally,), 30) == 0
    assert whole_rally_correctness(replace(sequence, start_frame=290), (rally,), 30) == -1


def test_timing_allowance_does_not_merge_a_neighbour_outside_the_span():
    rally = LabelledRally('rally', (100, 200, 300), (Half.TOP, Half.BOT, Half.TOP))
    neighbour = LabelledRally('neighbour', (398, 450), (Half.BOT, Half.TOP))
    events = tuple(ContactEvent(frame, .9, side) for frame, side in zip(rally.frames, rally.sides, strict=True))
    sequence = ContactSequence(0, 40, 391, events)
    for tolerance in (5, 10):
        assert whole_rally_correctness(sequence, (rally, neighbour), 30, tolerance_at_30_fps=tolerance) == 1
        assert judge_found_rallies((sequence,), (rally, neighbour), tolerance, fps=30)[0]['outcome'] == 'fully_correct'


def test_unlimited_context_preserves_rally_and_contact_checks():
    policy = RallyBoundaryPolicy(None, None)
    rally = LabelledRally('rally', (300, 400, 500), (Half.TOP, Half.BOT, Half.TOP))
    events = tuple(ContactEvent(frame, .9, side) for frame, side in zip(rally.frames, rally.sides, strict=True))
    sequence = ContactSequence(0, 0, 1000, events)
    assert whole_rally_correctness(sequence, (rally,), 30, boundary_policy=policy) == 1
    assert judge_found_rallies((sequence,), (rally,), 10, fps=30, boundary_policy=policy)[0]['outcome'] == 'fully_correct'
    neighbour = LabelledRally('neighbour', (800, 900), (Half.BOT, Half.TOP))
    assert whole_rally_correctness(sequence, (rally, neighbour), 30, boundary_policy=policy) == 0
    for changed in (
        replace(sequence, start_frame=350),
        replace(sequence, events=events[:-1]),
        replace(sequence, events=(*events, ContactEvent(600, .9, Half.BOT))),
        replace(sequence, events=tuple(replace(event, side=Half.BOT if event.side == Half.TOP else Half.TOP)
                                       for event in events)),
    ):
        assert whole_rally_correctness(changed, (rally,), 30, boundary_policy=policy) == 0
