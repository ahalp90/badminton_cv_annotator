"""Focused checks for the combined label-guided event-edit ceiling."""

from scratch.contact_det.scripts.score_contact_rallies import (
    FixedEvent,
    FixedSpan,
    RallyReference,
)
from scratch.contact_det_followup.scripts.audit_combined_best_case import (
    CombinedAction,
    _apply_actions,
    _option_fully_correct,
    delete_actions,
    section_actions,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import HumanLabels


def _span() -> FixedSpan:
    return FixedSpan(
        "sset_01",
        3,
        100,
        200,
        (
            FixedEvent("sset_01", 120, 0.95, "Top"),
            FixedEvent("sset_01", 150, 0.96, "Bot"),
        ),
    )


def _candidate_list() -> dict[str, object]:
    return {
        "fixture": "sset_01",
        "span_id": 3,
        "section_start_frame": 100,
        "section_end_frame": 200,
        "fixed_contact_frame": 120,
        "candidates": [
            {"frame": 120, "is_fixed_contact": True},
            {
                "frame": 90,
                "is_fixed_contact": False,
                "contact_score": 0.8,
                "predicted_side": "Bot",
            },
        ],
    }


def test_delete_action_removes_exactly_one_existing_event() -> None:
    span = _span()
    delete = delete_actions(span)[0]

    revised = _apply_actions(
        (span,),
        {"sset_01": span.events},
        {("sset_01", 3): delete},
    )

    assert [event.frame for event in revised.spans[0].events] == [150]
    assert [event.frame for event in revised.events_by_fixture["sset_01"]] == [150]
    assert delete.deleted_frame == 120


def test_section_action_pool_is_the_union_of_start_and_delete_actions() -> None:
    span = _span()
    fixture_events = (
        FixedEvent("sset_01", 95, 0.7, "Top"),
        *span.events,
    )

    actions = section_actions(
        span, _candidate_list(), fixture_events, previous_span_end=80
    )

    assert [action.kind for action in actions] == [
        "keep",
        "add",
        "replace",
        "delete",
        "delete",
        "add_delete",
        "add_delete",
        "replace_delete",
        "replace_delete",
    ]
    assert [event.frame for event in actions[1].span.events] == [90, 95, 120, 150]
    assert [event.frame for event in actions[2].span.events] == [90, 95, 150]
    assert [action.deleted_frame for action in actions[3:5]] == [120, 150]
    assert all(
        action.deleted_frame != 90
        for action in actions
        if action.kind in {"add_delete", "replace_delete"}
    )


def test_combined_action_applies_start_edit_then_deletes_one_retained_event() -> None:
    span = _span()
    fixture_events = (
        FixedEvent("sset_01", 95, 0.7, "Top"),
        *span.events,
    )
    combined = next(
        action
        for action in section_actions(span, _candidate_list(), fixture_events, 80)
        if action.kind == "add_delete" and action.deleted_frame == 95
    )

    revised = _apply_actions(
        (span,),
        {"sset_01": fixture_events},
        {("sset_01", 3): combined},
    )

    assert [event.frame for event in revised.spans[0].events] == [90, 120, 150]
    assert [event.frame for event in revised.events_by_fixture["sset_01"]] == [90, 120, 150]


def test_replace_delete_keeps_the_replacement_and_removes_the_second_event() -> None:
    span = _span()
    combined = next(
        action
        for action in section_actions(
            span,
            _candidate_list(),
            span.events,
            previous_span_end=80,
        )
        if action.kind == "replace_delete" and action.deleted_frame == 150
    )

    revised = _apply_actions(
        (span,),
        {"sset_01": span.events},
        {("sset_01", 3): combined},
    )

    assert [event.frame for event in revised.spans[0].events] == [90]
    assert [event.frame for event in revised.events_by_fixture["sset_01"]] == [90]


def test_ceiling_tries_both_side_phases_instead_of_using_the_vote() -> None:
    span = FixedSpan(
        "sset_01",
        3,
        100,
        200,
        (
            FixedEvent("sset_01", 120, 0.95, "Top"),
            FixedEvent("sset_01", 150, 0.96, "Top"),
        ),
    )
    labels = HumanLabels(
        {"sset_01": (RallyReference("sset_01", 0, "set:1", (120, 150)),)},
        {("sset_01", 120): "Bot", ("sset_01", 150): "Top"},
    )

    assert _option_fully_correct(CombinedAction("keep", None, None, span), labels, {"sset_01": 30.0}, 5)
