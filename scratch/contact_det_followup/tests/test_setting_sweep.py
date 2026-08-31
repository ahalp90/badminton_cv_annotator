"""Checks for the timing-only contact-setting sweep."""

from scratch.contact_det.scripts.score_contact_rallies import (
    FixedEvent,
    FixedSpan,
    RallyReference,
)
from scratch.contact_det_followup.scripts.score_setting_sweep import (
    SettingEvaluation,
    assign_events_to_spans,
    choose_setting,
    timing_complete_ids,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import HumanLabels


def test_timing_complete_ignores_side_but_requires_exact_matching() -> None:
    labels = HumanLabels(
        rallies={"sset_01": (RallyReference("sset_01", 0, "1:1", (10, 20)),)},
        target_sides={("sset_01", 10): "Top", ("sset_01", 20): "Bot"},
    )
    exact = FixedSpan(
        "sset_01",
        1,
        0,
        30,
        (
            FixedEvent("sset_01", 11, 0.9, None),
            FixedEvent("sset_01", 19, 0.9, None),
        ),
    )
    extra = FixedSpan(
        "sset_01",
        2,
        30,
        60,
        (FixedEvent("sset_01", 40, 0.9, None),),
    )

    assert timing_complete_ids((exact, extra), labels, {"sset_01": 30.0}) == {
        ("sset_01", 1)
    }


def test_timing_complete_uses_requested_base30_tolerance() -> None:
    labels = HumanLabels(
        rallies={"sset_01": (RallyReference("sset_01", 0, "1:1", (10, 20)),)},
        target_sides={("sset_01", 10): "Top", ("sset_01", 20): "Bot"},
    )
    span = FixedSpan(
        "sset_01",
        1,
        0,
        30,
        (
            FixedEvent("sset_01", 11, 0.9, None),
            FixedEvent("sset_01", 26, 0.9, None),
        ),
    )

    assert timing_complete_ids((span,), labels, {"sset_01": 30.0}, 5) == frozenset()
    assert timing_complete_ids((span,), labels, {"sset_01": 30.0}, 10) == {
        ("sset_01", 1)
    }


def test_span_assignment_uses_half_open_bounds_and_counts_unassigned() -> None:
    templates = (
        FixedSpan("sset_01", 1, 10, 20, ()),
        FixedSpan("sset_01", 2, 20, 30, ()),
    )
    events = {
        "sset_01": (
            FixedEvent("sset_01", 9, 0.9, None),
            FixedEvent("sset_01", 10, 0.9, None),
            FixedEvent("sset_01", 20, 0.9, None),
        )
    }

    spans, unassigned = assign_events_to_spans(templates, events)

    assert tuple(event.frame for event in spans[0].events) == (10,)
    assert tuple(event.frame for event in spans[1].events) == (20,)
    assert unassigned == 1


def _evaluation(
    cutoff: float,
    group_a_complete: int,
    group_b_complete: int,
    group_a_matches: int,
    group_b_matches: int,
) -> SettingEvaluation:
    def contact(matched: int) -> dict[str, int | float | None]:
        return {
            "contact_count": 10,
            "prediction_count": 10,
            "matched": matched,
            "first_contact_count": 2,
            "first_contact_matched": 1,
        }

    return SettingEvaluation(
        score_cutoff=cutoff,
        duplicate_distance_at_30_fps=6,
        timing_complete_ids=frozenset(),
        timing_complete_by_group={"A": group_a_complete, "B": group_b_complete},
        contact_by_group={"A": contact(group_a_matches), "B": contact(group_b_matches)},
        unassigned_by_group={"A": 0, "B": 0},
    )


def test_setting_choice_uses_only_named_training_groups() -> None:
    strong_on_a = _evaluation(0.8, 5, 100, 8, 10)
    strong_on_b = _evaluation(0.9, 4, 200, 9, 10)

    assert choose_setting((strong_on_a, strong_on_b), ("A",)) is strong_on_a
    assert choose_setting((strong_on_a, strong_on_b), ("B",)) is strong_on_b
