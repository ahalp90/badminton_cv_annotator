"""Focused tests for the label-aware candidate-union ceiling."""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import score_contact_candidate_union_ceiling as scorer


def _event(
    frame: int,
    *,
    side: str | None = "Top",
    fixture: str = "sset_01",
) -> scorer.rally_scorer.FixedEvent:
    return scorer.rally_scorer.FixedEvent(fixture, frame, 0.9, side)


def _span(
    *events: scorer.rally_scorer.FixedEvent,
    start: int = 0,
    end: int = 100,
    span_id: int = 0,
) -> scorer.rally_scorer.FixedSpan:
    return scorer.rally_scorer.FixedSpan("sset_01", span_id, start, end, tuple(events))


def _rally(*frames: int, rally_id: str = "set1:1") -> scorer.rally_scorer.RallyReference:
    return scorer.rally_scorer.RallyReference("sset_01", 0, rally_id, tuple(frames))


def _sides(*frames: int, value: str = "Top") -> dict[tuple[str, int], str]:
    return {("sset_01", frame): value for frame in frames}


def _source_flags(*events: scorer.rally_scorer.FixedEvent) -> dict[tuple[str, int], str]:
    return {scorer._event_identity(event): scorer.SOURCE_ANCHOR for event in events}


def test_inclusive_tolerance_selects_a_candidate_on_the_boundary() -> None:
    result = scorer._search_assignment(
        _span(_event(12), _event(30)),
        _rally(10, 30),
        tolerance_frames=2,
        target_sides=_sides(10, 30),
        require_correct_side=True,
    )

    assert result.selected_event_indices == (0, 1)
    assert result.score is not None and result.score.fully_correct


def test_oracle_can_drop_anchor_for_alternative_and_reports_actions() -> None:
    events = (_event(9, side="Bot"), _event(10), _event(20))
    span = _span(*events)
    result = scorer._search_assignment(
        span,
        _rally(10, 20),
        tolerance_frames=1,
        target_sides=_sides(10, 20),
        require_correct_side=True,
    )
    flags = {
        ("sset_01", 9): scorer.SOURCE_ANCHOR,
        ("sset_01", 10): scorer.SOURCE_ALTERNATIVE,
        ("sset_01", 20): scorer.SOURCE_ANCHOR_AND_ALTERNATIVE,
    }

    detail = scorer._span_detail(span, _rally(10, 20), result, flags)

    assert detail["selected_frames"] == [10, 20]
    assert detail["selected_predicted_sides"] == ["Top", "Top"]
    assert detail["action_counts"]["retain"] == 2
    assert detail["action_counts"]["drop"] == 1
    assert detail["action_counts"]["dropped_anchors"] == 1
    assert detail["action_counts"]["selected_non_anchor_alternatives"] == 1
    assert detail["selected_source_flags"] == [scorer.SOURCE_ALTERNATIVE, scorer.SOURCE_ANCHOR_AND_ALTERNATIVE]


def test_competing_contacts_use_deterministic_smallest_domain_order() -> None:
    span = _span(_event(9), _event(11), _event(20))

    result = scorer._search_assignment(
        span,
        _rally(10, 20),
        tolerance_frames=2,
        target_sides=_sides(10, 20),
        require_correct_side=True,
    )

    assert result.selected_event_indices == (0, 2)
    assert result.attempts == 2  # one locally accepted subset per component


def test_disconnected_components_search_alternatives_additively() -> None:
    # Each timing component has two rejected/accepted local choices.  A
    # whole-span product would combine those choices, while the decomposed
    # search only walks the two local alternatives in each component.
    span = _span(
        _event(8),
        _event(11),
        _event(13),
        _event(108),
        _event(111),
        _event(113),
        end=200,
    )
    rally = _rally(10, 12, 110, 112)
    result = scorer._search_assignment(
        span,
        rally,
        tolerance_frames=3,
        target_sides=_sides(*rally.frames),
        require_correct_side=True,
    )

    assert result.selected_event_indices == (1, 2, 4, 5)
    assert result.component_count == 2
    assert result.maximum_component_contacts == 2
    assert result.maximum_component_candidates == 3
    assert result.attempts == 4
    assert result.rejected_assignments == 2
    assert result.score is not None and result.score.fully_correct


def test_timing_components_keep_opposite_side_cross_edges_connected() -> None:
    span = _span(_event(10, side="Top"), _event(20, side="Bot"))
    rally = _rally(10, 20)
    sides = {("sset_01", 10): "Top", ("sset_01", 20): "Bot"}

    result = scorer._search_assignment(
        span,
        rally,
        tolerance_frames=10,
        target_sides=sides,
        require_correct_side=True,
    )

    # The side-filtered domains are separate, but both timing edges connect
    # the contacts.  The production matcher therefore sees one component.
    assert result.selected_event_indices == (0, 1)
    assert result.component_count == 1
    assert result.maximum_component_contacts == 2
    assert result.maximum_component_candidates == 2


def test_side_compatible_assignment_can_fail_actual_greedy_side_pairing() -> None:
    # The only side-compatible assignment is 8 -> GT 10 and 10 -> GT 11.
    # Production greedy matching instead pairs 10 to GT 10 and 8 to GT 11,
    # which preserves timing but swaps the two sides.
    span = _span(_event(8, side="Top"), _event(10, side="Bot"))
    rally = _rally(10, 11)
    sides = {("sset_01", 10): "Top", ("sset_01", 11): "Bot"}

    result = scorer._search_assignment(
        span,
        rally,
        tolerance_frames=3,
        target_sides=sides,
        require_correct_side=True,
    )

    assert result.selected_event_indices is None
    assert result.attempts == 1
    assert result.rejected_assignments == 1
    assert result.component_count == 1


def test_exhaustive_search_rejects_production_greedy_matching_failure() -> None:
    # The production matcher takes 11 for GT 10 because its offset is one.
    # That leaves GT 12 unmatched.  The complete assignment is 8 -> 10 and
    # 11 -> 12, but the unchanged production matcher still rejects that
    # selected set.  The ceiling must report no accepted assignment.
    span = _span(_event(8), _event(11))

    result = scorer._search_assignment(
        span,
        _rally(10, 12),
        tolerance_frames=3,
        target_sides=_sides(10, 12),
        require_correct_side=True,
    )

    assert result.selected_event_indices is None
    assert result.attempts == 1
    assert result.rejected_assignments == 1


def test_timing_only_accepts_wrong_or_unanswered_side_but_full_does_not() -> None:
    span = _span(_event(10, side="Bot"), _event(20, side=None))
    sides = _sides(10, 20)

    timing = scorer._search_assignment(
        span,
        _rally(10, 20),
        tolerance_frames=0,
        target_sides=sides,
        require_correct_side=False,
    )
    full = scorer._search_assignment(
        span,
        _rally(10, 20),
        tolerance_frames=0,
        target_sides=sides,
        require_correct_side=True,
    )

    assert timing.selected_event_indices == (0, 1)
    assert timing.score is not None and not timing.score.fully_correct
    assert full.selected_event_indices is None


def test_half_open_assignment_leaves_end_boundary_unassigned() -> None:
    events = {"sset_01": (_event(10), _event(20), _event(30))}
    evidence = {
        "fixtures": [
            {
                "fixture": "sset_01",
                "spans": [{"span_id": 0, "start_frame": 10, "end_frame": 30}],
            }
        ]
    }

    spans = scorer.rally_scorer.fixed_spans_from_evidence(evidence, events)

    assert [event.frame for event in spans[0].events] == [10, 20]
    assert [event.frame for event in scorer.rally_scorer.unassigned_events(spans, events)] == [30]


def test_false_and_multi_rally_spans_abstain() -> None:
    spans = (
        _span(_event(10), span_id=0),
        _span(_event(10), span_id=1),
    )
    rallies = {
        "sset_01": (
            _rally(10, rally_id="set1:1"),
            _rally(10, rally_id="set1:2"),
        )
    }
    report = scorer._ceiling_for_tolerance(
        spans,
        rallies,
        _sides(10),
        tolerance_base30=10,
        source_flags={},
        require_correct_side=True,
    )

    assert report["feasible_span_count"] == 0
    assert report["false_spans"] == 0
    assert report["multiple_rally_spans"] == 2


def test_source_flags_mark_an_alternative_that_is_also_an_anchor() -> None:
    rows = np.zeros(2, dtype=scorer.tree_scorer.CANDIDATE_SCORE_DTYPE)
    rows["fixture"] = b"sset_01"
    rows["frame"] = [10, 20]
    frozen = SimpleNamespace(
        nplus_rows=rows,
        anchor_indices=np.asarray([0], dtype=np.int32),
        alternative_indices=np.asarray([0, 1], dtype=np.int32),
        shortlist_indices=np.asarray([0, 1], dtype=np.int32),
    )

    flags = scorer._source_flags(frozen)

    assert flags == {
        ("sset_01", 10): scorer.SOURCE_ANCHOR_AND_ALTERNATIVE,
        ("sset_01", 20): scorer.SOURCE_ALTERNATIVE,
    }


def test_exhaustive_search_agrees_with_brute_force_on_a_small_span() -> None:
    span = _span(_event(8), _event(11), _event(19))
    rally = _rally(10, 20)
    tolerance = 3
    sides = _sides(10, 20)
    result = scorer._search_assignment(
        span,
        rally,
        tolerance,
        sides,
        require_correct_side=True,
    )

    brute_force_solutions = []
    for selected in itertools.permutations(range(len(span.events)), len(rally.frames)):
        if len(set(selected)) != len(selected):
            continue
        if any(abs(span.events[event_index].frame - gt_frame) > tolerance for event_index, gt_frame in zip(selected, rally.frames, strict=True)):
            continue
        if any(span.events[event_index].predicted_side != sides[("sset_01", gt_frame)] for event_index, gt_frame in zip(selected, rally.frames, strict=True)):
            continue
        selected_span = scorer.replace(
            span,
            events=tuple(span.events[index] for index in sorted(selected)),
        )
        evaluated = scorer.rally_scorer.evaluate_span(
            selected_span,
            (rally,),
            sides,
            tolerance,
            0.0,
        )
        if evaluated.fully_correct and not evaluated.rejection_reasons:
            brute_force_solutions.append(tuple(sorted(selected)))

    assert brute_force_solutions
    assert result.selected_event_indices in brute_force_solutions


def test_decomposed_search_agrees_with_brute_force_on_two_components() -> None:
    span = _span(_event(8), _event(12), _event(108), _event(112), end=200)
    rally = _rally(10, 110)
    tolerance = 2
    sides = _sides(*rally.frames)
    result = scorer._search_assignment(
        span,
        rally,
        tolerance,
        sides,
        require_correct_side=True,
    )

    brute_force_solutions = []
    for selected in itertools.permutations(range(len(span.events)), len(rally.frames)):
        if any(
            abs(span.events[event_index].frame - gt_frame) > tolerance
            for event_index, gt_frame in zip(selected, rally.frames, strict=True)
        ):
            continue
        if any(
            span.events[event_index].predicted_side != sides[("sset_01", gt_frame)]
            for event_index, gt_frame in zip(selected, rally.frames, strict=True)
        ):
            continue
        selected_span = scorer.replace(
            span,
            events=tuple(span.events[index] for index in sorted(selected)),
        )
        evaluated = scorer.rally_scorer.evaluate_span(
            selected_span,
            (rally,),
            sides,
            tolerance,
            0.0,
        )
        if evaluated.fully_correct and not evaluated.rejection_reasons:
            brute_force_solutions.append(tuple(sorted(selected)))

    assert result.component_count == 2
    assert brute_force_solutions
    assert result.selected_event_indices in brute_force_solutions


def test_hall_prune_rejects_many_shared_domains_before_factorial_search() -> None:
    # Every contact can use every one of five events, but the rally has ten
    # contacts.  A naive assignment walk would explore every partial event
    # permutation before discovering that it cannot reach a complete
    # assignment.  The exact residual matching check proves the failure at
    # the root, without evaluating a made-up span.
    span = _span(*(_event(frame) for frame in range(5)))
    result = scorer._search_assignment(
        span,
        _rally(*range(10)),
        tolerance_frames=100,
        target_sides=_sides(*range(10)),
        require_correct_side=True,
    )

    assert result.selected_event_indices is None
    assert result.attempts == 0
    assert result.rejected_assignments == 0
    assert result.feasibility_prunes == 1
    assert result.visited_states == 1
    assert result.memo_prunes == 0


def test_score_loads_labels_after_label_blind_freeze(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        scorer,
        "_freeze_label_blind",
        lambda _arguments: calls.append("freeze") or object(),
    )
    monkeypatch.setattr(
        scorer,
        "_load_labels",
        lambda: calls.append("labels") or ({}, {}),
    )
    monkeypatch.setattr(
        scorer,
        "_result_payload",
        lambda *_arguments: calls.append("score") or {},
    )

    assert scorer.score(object()) == {}
    assert calls == ["freeze", "labels", "score"]
