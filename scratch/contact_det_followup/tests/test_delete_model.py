"""Focused checks for the bounded delete-event chooser."""

import inspect

import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import (
    FixedEvent,
    FixedSpan,
    RallyReference,
)
from scratch.contact_det_followup.scripts.score_delete_model import (
    DeleteEvaluation,
    DeleteRow,
    apply_selected_deletions,
    assign_delete_targets,
    build_delete_rows,
    choose_deployable_configuration,
    predict_delete_scores,
    select_deletions,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import HumanLabels


def _span() -> FixedSpan:
    return FixedSpan(
        "sset_01",
        0,
        100,
        200,
        (
            FixedEvent("sset_01", 115, 0.40, "Top"),
            FixedEvent("sset_01", 125, 0.90, "Bot"),
            FixedEvent("sset_01", 150, 0.80, "Top"),
        ),
    )


def _rows() -> tuple[DeleteRow, ...]:
    span = _span()
    return build_delete_rows((span,), {"sset_01": "A"}, {"sset_01": 30.0})


def test_delete_stream_removes_only_selected_event() -> None:
    span = _span()
    rows = _rows()

    revised = apply_selected_deletions(
        (span,),
        {"sset_01": span.events},
        {("sset_01", 0): rows[1]},
    )

    assert [event.frame for event in revised.spans[0].events] == [115, 150]
    assert [event.frame for event in revised.events_by_fixture["sset_01"]] == [115, 150]


def test_delete_features_cover_event_position_gaps_and_section_statistics() -> None:
    rows = _rows()

    assert len(rows) == 3
    assert rows[0].features == (
        0.40,
        3.0,
        100.0,
        0.0,
        2.0,
        1.0,
        0.0,
        0.0,
        -0.40,
        0.0,
        10.0,
        1.0,
    )
    assert rows[1].features[9:11] == (10.0, 25.0)
    assert rows[2].features[4:7] == (0.0, 0.0, 1.0)


def test_delete_target_marks_one_lowest_score_repair() -> None:
    span = _span()
    labels = HumanLabels(
        {"sset_01": (RallyReference("sset_01", 0, "set:1", (125, 150)),)},
        {("sset_01", 125): "Bot", ("sset_01", 150): "Top"},
    )

    targets = assign_delete_targets(
        _rows(),
        (span,),
        labels,
        {"sset_01": 30.0},
    )

    positives = [identity for identity, target in targets.by_event.items() if target.positive]
    assert positives == [("sset_01", 0, 115)]
    assert targets.by_event[("sset_01", 0, 115)].deletion_fully_correct is True
    assert targets.by_event[("sset_01", 0, 125)].positive is False


def test_delete_selection_is_label_free_and_one_per_section() -> None:
    rows = _rows()
    scores = {row.identity: score for row, score in zip(rows, (0.8, 0.95, 0.90), strict=True)}

    selected = select_deletions(rows, scores, cutoff=0.7)

    assert list(selected) == [("sset_01", 0)]
    assert selected[("sset_01", 0)].frame == 125
    assert "labels" not in inspect.signature(predict_delete_scores).parameters
    assert "labels" not in inspect.signature(select_deletions).parameters

    class FakeModel:
        classes_ = np.asarray([0, 1])

        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            assert features.shape == (3, 12)
            return np.asarray(
                [[0.2, 0.8], [0.6, 0.4], [0.1, 0.9]],
            )

    predicted = predict_delete_scores(FakeModel(), rows)
    assert predicted[rows[0].identity] == 0.8


def test_delete_gate_requires_thirty_net_and_one_break_per_five_repairs() -> None:
    passing = DeleteEvaluation(
        "shallow_hgb",
        0.9,
        100,
        130,
        frozenset(("sset_01", index) for index in range(30)),
        frozenset(),
        {},
        100,
        50,
    )
    failing = DeleteEvaluation(
        "shallow_hgb",
        0.7,
        100,
        130,
        frozenset(("sset_01", index) for index in range(31)),
        frozenset(("sset_02", index) for index in range(7)),
        {},
        100,
        50,
    )

    assert choose_deployable_configuration((passing, failing)) == passing
