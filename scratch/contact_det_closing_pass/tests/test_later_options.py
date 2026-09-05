"""Exercise later-insertion boundaries, duplicate competition and side voting."""

import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import (
    FixedEvent,
    FixedSpan,
    RallyReference,
)
from scratch.contact_det_closing_pass.scripts.evaluation import section_result
from scratch.contact_det_closing_pass.scripts.later_evaluation import opportunity
from scratch.contact_det_closing_pass.scripts.later_options import (
    LaterOption,
    apply_options,
    build_later_options,
    select_options,
    shortlist_frames,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_evaluation import (
    contact_edit_effect,
)
from scratch.contact_det_followup.scripts.audit_combined_best_case import CombinedAction
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import HumanLabels


def event(frame: int, side: str = "Top") -> FixedEvent:
    return FixedEvent("fixture", frame, 0.8, side)


def test_shortlist_uses_real_scores_and_scaled_distance() -> None:
    span = FixedSpan("fixture", 0, 0, 200, (event(20), event(140)))
    scores = np.array([(10, .99), (20, .99), (27, .99), (60, .8), (62, .9), (95, .7), (200, 1.)],
                      dtype=[("frame", "i4"), ("contact_score", "f8")])
    assert shortlist_frames(span, scores, 60) == [62, 95]
    assert shortlist_frames(span, scores, 30, limit=1) == [27]


def test_duplicate_insertion_cannot_claim_a_distinct_contact() -> None:
    span = FixedSpan("fixture", 0, 0, 100, (event(20), event(60)))
    base = CombinedAction("keep", None, None, span)
    candidates = {("fixture", 0): (event(20), event(25), event(28), event(100))}
    options = build_later_options((base,), candidates, {"fixture": 30})
    assert [option.inserted.frame for option in options if option.inserted] == [28]
    effects = contact_edit_effect(span, options[1].span, (25, 60), 10)
    assert effects["newly_matched_contacts"] == 0
    assert effects["unnecessary_added"] == 1


def test_insertion_changes_alternation_and_preserves_raw_guesses() -> None:
    span = FixedSpan("fixture", 0, 0, 120, (event(20), event(80)))
    base = CombinedAction("keep", None, None, span)
    options = build_later_options((base,), {("fixture", 0): (event(50, "Bot"),)}, {"fixture": 30})
    labels = HumanLabels(
        {"fixture": (RallyReference("fixture", 0, "one", (20, 50, 80)),)},
        {("fixture", 20): "Top", ("fixture", 50): "Bot", ("fixture", 80): "Top"},
    )
    assert not section_result(span, labels, 10)["side_rule_fully_correct"]
    assert section_result(options[1].span, labels, 10)["side_rule_fully_correct"]
    assert options[1].span.events[-1] == span.events[-1]


def test_start_and_delete_combinations_keep_insertion_and_fullstream_events() -> None:
    original = FixedSpan("fixture", 0, 20, 100, (event(20), event(70), event(90)))
    edited = FixedSpan("fixture", 0, 10, 100, (event(10, "Bot"), event(20), event(70)))
    base = CombinedAction("add_delete", 10, 90, edited)
    options = build_later_options((base,), {("fixture", 0): (event(45, "Bot"),)}, {"fixture": 30})
    selected = select_options(options, np.array([.3, .8]))
    streams = apply_options((original,), {"fixture": (*original.events, event(150))}, selected)
    assert [contact.frame for contact in streams.events_by_fixture["fixture"]] == [10, 20, 45, 70, 150]
    assert streams.spans[0].start_frame == 10
    assert select_options(options, np.array([.8, .8]))[("fixture", 0)].inserted is None


def test_empty_sections_remain_visible() -> None:
    span = FixedSpan("fixture", 0, 0, 100, ())
    options = build_later_options((CombinedAction("keep", None, None, span),), {}, {"fixture": 30})
    selected = select_options(options, np.array([.1]))
    assert apply_options((span,), {"fixture": ()}, selected).spans == (span,)


def test_opportunity_includes_missing_opening_and_later_hit_together() -> None:
    span = FixedSpan("fixture", 0, 20, 100, (event(20, "Bot"), event(80, "Bot")))
    keep = CombinedAction("keep", None, None, span)
    opening_span = FixedSpan("fixture", 0, 10, 100, (event(10), *span.events))
    opening = CombinedAction("add", 10, None, opening_span)
    options = build_later_options((keep, opening), {("fixture", 0): (event(50),)}, {"fixture": 30})
    labels = HumanLabels(
        {"fixture": (RallyReference("fixture", 0, "one", (10, 20, 50, 80)),)},
        {("fixture", 10): "Top", ("fixture", 20): "Bot", ("fixture", 50): "Top", ("fixture", 80): "Bot"},
    )
    result = opportunity(options, {("fixture", 0): LaterOption(keep, None, span)},
                         labels, {"fixture": 30}, {"fixture": "A"})
    assert result["10"]["counts"]["repair_with_same_base"] == 0
    assert result["10"]["counts"]["repair_with_start_delete_combinations"] == 1
    assert result["10"]["sections"][0]["distinct_local_candidate_frames"] == [50]
