"""Focused checks for the first-contact action chooser."""

import inspect

import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import (
    FixedEvent,
    FixedSpan,
    RallyReference,
)
from scratch.contact_det_followup.scripts.score_start_model import (
    ActionRow,
    StartChoiceEvaluation,
    apply_selected_actions,
    assign_action_targets,
    build_action_rows,
    choose_deployable_configuration,
    predict_action_scores,
    select_actions,
    start_actions,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    CandidateRow,
    HumanLabels,
)


def _candidate(frame: int, score: float = 0.8) -> CandidateRow:
    return CandidateRow(
        fixture="sset_01",
        group="A",
        fps=30.0,
        span_id=0,
        section_start_frame=100,
        section_end_frame=200,
        prefix_start_frame=80,
        fixed_contact_frame=120,
        frame=frame,
        contact_score=score,
        fixed_contact_score=0.95,
        kept=False,
        predicted_side=None,
        fixed_predicted_side=None,
        features=(score, 0.95, 10.0, 5.0, 100.0, 0.0, 0.0, 0.0, 0.0),
    )


def _video() -> dict[str, object]:
    return {
        "group": "A",
        "video": {"fixture": "sset_01", "fps": 30.0},
        "spans": [{"span_id": 0, "start_frame": 100, "end_frame": 200}],
        "kept_contacts": [
            {"frame": 120},
            {"frame": 150},
        ],
    }


def _baseline_span() -> FixedSpan:
    return FixedSpan(
        "sset_01",
        0,
        100,
        200,
        (
            FixedEvent("sset_01", 120, 0.95, "Top"),
            FixedEvent("sset_01", 150, 0.95, "Bot"),
        ),
    )


def test_action_target_is_timing_only_and_prefers_the_repairing_action() -> None:
    candidate = _candidate(90)
    action_rows = build_action_rows((candidate,))
    labels = HumanLabels(
        {"sset_01": (RallyReference("sset_01", 0, "set:1", (90, 120, 150)),)},
        {},
    )

    targets = assign_action_targets(
        action_rows,
        (_baseline_span(),),
        {"sset_01": _baseline_span().events},
        (_video(),),
        labels,
        {"sset_01": 30.0},
        default_group="A",
    )

    add_target = targets.by_action[candidate.identity + ("add",)]
    replace_target = targets.by_action[candidate.identity + ("replace",)]
    assert add_target.positive is True
    assert add_target.action_timing_complete is True
    assert replace_target.positive is False
    assert replace_target.action_timing_complete is False
    assert targets.section_statuses[("sset_01", 0)] == "usable_candidate"


def test_add_and_replace_retain_prefix_events_with_different_fixed_contact_semantics() -> (
    None
):
    candidate = _candidate(90)
    action_rows = build_action_rows((candidate,))
    baseline = _baseline_span()
    events = {
        "sset_01": (
            FixedEvent("sset_01", 95, 0.7, "Top"),
            *baseline.events,
        )
    }

    added = apply_selected_actions(
        (baseline,),
        events,
        {("sset_01", 0): next(row for row in action_rows if row.action == "add")},
    )
    replaced = apply_selected_actions(
        (baseline,),
        events,
        {("sset_01", 0): next(row for row in action_rows if row.action == "replace")},
    )

    assert [event.frame for event in added.spans[0].events] == [90, 95, 120, 150]
    assert [event.frame for event in replaced.spans[0].events] == [90, 95, 150]
    assert [event.frame for event in added.events_by_fixture["sset_01"]] == [
        90,
        95,
        120,
        150,
    ]
    assert [event.frame for event in replaced.events_by_fixture["sset_01"]] == [
        90,
        95,
        150,
    ]


def test_start_actions_rejects_a_candidate_that_overlaps_the_predecessor() -> None:
    span = _baseline_span()
    assert (
        start_actions(span, (_candidate(90),), span.events, previous_span_end=95) == ()
    )


def test_selection_keeps_one_action_per_section_and_prefers_add_on_a_tie() -> None:
    rows = build_action_rows((_candidate(90, 0.8), _candidate(95, 0.8)))
    scores = {row.identity: 0.75 for row in rows}

    selected = select_actions(rows, scores, cutoff=0.7)

    assert list(selected) == [("sset_01", 0)]
    assert selected[("sset_01", 0)].action == "add"
    assert selected[("sset_01", 0)].candidate.frame == 90


def test_prediction_and_selection_interfaces_do_not_accept_labels() -> None:
    rows = build_action_rows((_candidate(90),))

    class FakeModel:
        classes_ = np.asarray([0, 1])

        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            assert features.shape == (2, 10)
            return np.asarray([[0.2, 0.8], [0.6, 0.4]])

    scores = predict_action_scores(FakeModel(), rows)

    assert scores[rows[0].identity] == 0.8
    assert scores[rows[1].identity] == 0.4
    assert "labels" not in inspect.signature(predict_action_scores).parameters
    assert "labels" not in inspect.signature(select_actions).parameters
    assert isinstance(rows[0], ActionRow)


def test_deployable_choice_applies_the_repair_and_break_gate() -> None:
    descriptive = StartChoiceEvaluation(
        "shallow_hgb",
        0.7,
        100,
        130,
        frozenset(("sset_01", index) for index in range(40)),
        frozenset(("sset_02", index) for index in range(10)),
        200,
        {"add": 200, "replace": 0},
    )
    cautious = StartChoiceEvaluation(
        "shallow_hgb",
        0.9,
        100,
        120,
        frozenset(("sset_01", index) for index in range(20)),
        frozenset(),
        50,
        {"add": 50, "replace": 0},
    )

    assert choose_deployable_configuration((descriptive, cautious)) == cautious
