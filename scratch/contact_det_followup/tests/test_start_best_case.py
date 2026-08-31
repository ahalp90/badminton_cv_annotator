"""Checks for first-contact keep, add, and replace actions."""

from scratch.contact_det.scripts.score_contact_rallies import FixedEvent, FixedSpan
from scratch.contact_det_followup.scripts.score_start_best_case import start_actions


def test_start_actions_add_and_replace_an_earlier_candidate() -> None:
    span = FixedSpan(
        "sset_01",
        3,
        100,
        200,
        (
            FixedEvent("sset_01", 120, 0.95, "Top"),
            FixedEvent("sset_01", 150, 0.96, "Bot"),
        ),
    )
    candidate_list = {
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

    fixture_events = (
        FixedEvent("sset_01", 95, 0.7, "Top"),
        *span.events,
    )
    actions = start_actions(span, candidate_list, fixture_events, previous_span_end=80)

    assert [action.kind for action in actions] == ["keep", "add", "replace"]
    assert tuple(event.frame for event in actions[1].span.events) == (90, 95, 120, 150)
    assert tuple(event.frame for event in actions[2].span.events) == (90, 95, 150)
    assert actions[1].span.start_frame == 90
