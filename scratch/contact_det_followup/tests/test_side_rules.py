"""Checks for whole-rally player-side choices."""

import pytest

from annotator.point_winner import Half
from scratch.contact_det.scripts.score_contact_rallies import FixedEvent, FixedSpan
from scratch.contact_det_followup.scripts.audit_opposite_side_duplicates import (
    count_opposite_side_pairs,
)
from scratch.contact_det_followup.scripts.score_side_audit import (
    validate_decision_record,
)
from scratch.contact_det_followup.scripts.side_rules import (
    alternating_pattern,
    choose_simple_alternation,
)


def _span(sides: list[str | None]) -> FixedSpan:
    events = tuple(
        FixedEvent("1", frame, 0.9, side)
        for frame, side in enumerate(sides, start=10)
    )
    return FixedSpan("1", 3, 0, 100, events)


def test_alternating_pattern_uses_the_named_final_side() -> None:
    assert alternating_pattern(Half.TOP, 4) == ("Bot", "Top", "Bot", "Top")


def test_simple_alternation_ignores_unknown_guesses() -> None:
    decision = choose_simple_alternation(_span(["Top", None, "Top", "Bot"]))

    assert decision is not None
    assert decision.sides_after == ("Top", "Bot", "Top", "Bot")
    assert decision.score_gap == 3


def test_simple_alternation_leaves_a_tie_unchanged() -> None:
    assert choose_simple_alternation(_span(["Top", "Top"])) is None


def test_final_scorer_rejects_decisions_from_another_vote_gap() -> None:
    config = {
        "schema": "contact-detector-side-rule/1",
        "rule": "simple_alternation_vote",
        "minimum_vote_gap": 1,
    }
    decision = {
        "status": "complete",
        "labels_read": False,
        "config": "scratch/contact_det_followup/configs/side_rule.json",
        "prediction_source_commit": "baseline",
        "sections_seen": 10,
        "rule": "simple_alternation_vote",
        "minimum_vote_gap": 2,
    }

    with pytest.raises(ValueError, match="chosen rule"):
        validate_decision_record(decision, config, "baseline", 10)


def test_duplicate_audit_counts_only_close_opposite_sides() -> None:
    events_by_fixture = {
        "1": _span(["Top", "Bot"]).events,
        "2": (
            FixedEvent("2", 10, 0.9, "Top"),
            FixedEvent("2", 12, 0.9, "Top"),
            FixedEvent("2", 14, 0.9, None),
            FixedEvent("2", 17, 0.9, "Bot"),
        ),
    }

    assert count_opposite_side_pairs(events_by_fixture) == {
        "pair_count": 1,
        "pair_count_by_frame_gap": {"0": 0, "1": 1, "2": 0},
        "affected_videos": 1,
    }
