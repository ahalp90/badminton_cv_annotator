"""Focused checks for the bounded keep-or-review development audit."""

import inspect

import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import FixedEvent, FixedSpan
from scratch.contact_det_followup.scripts.score_keep_review import (
    EVENT_COUNT_BINS,
    FEATURE_NAMES,
    GROUPS,
    KeepReviewRow,
    build_feature_vector,
    choose_candidate_threshold,
    cross_fit_probabilities,
    keep_review_curve,
    predict_keep_probabilities,
)


def _span(
    events: tuple[FixedEvent, ...], *, start: int = 100, end: int = 200
) -> FixedSpan:
    return FixedSpan("sset_01", 0, start, end, events)


def test_features_use_finite_zero_values_for_empty_and_one_event_sections() -> None:
    empty = build_feature_vector(_span(()), 25.0)
    one = build_feature_vector(
        _span((FixedEvent("sset_01", 125, 0.8, None),)),
        25.0,
    )

    assert len(empty) == len(FEATURE_NAMES) == 10
    assert empty == (0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert one == (1.0, 4.0, 0.8, 0.8, 0.8, 0.0, 0.0, 1.0, 3.0, 1.0)
    assert np.isfinite(empty).all()
    assert np.isfinite(one).all()


def test_features_summarise_multi_event_scores_and_gaps_in_seconds() -> None:
    span = _span(
        (
            FixedEvent("sset_01", 110, 0.9, "Top"),
            FixedEvent("sset_01", 130, 0.4, "Bot"),
            FixedEvent("sset_01", 180, 0.7, None),
            FixedEvent("sset_01", 190, 0.5, "Top"),
        ),
        start=100,
        end=200,
    )

    assert build_feature_vector(span, 25.0) == (
        4.0,
        4.0,
        0.4,
        0.6,
        (0.4 + 0.5 + 0.7) / 3,
        0.4,
        2.0,
        0.4,
        0.4,
        1.0,
    )


def test_cross_fit_predicts_every_section_from_a_model_holding_out_its_group() -> None:
    rows = tuple(
        KeepReviewRow(
            fixture=f"sset_{index + 1:02d}",
            group=group,
            span_id=0,
            fps=25.0,
            features=(float(index),) + (0.0,) * 9,
        )
        for index, group in enumerate(GROUPS)
    )
    targets = {row.identity: bool(index % 2) for index, row in enumerate(rows)}
    fit_sizes: list[int] = []

    class FakeModel:
        classes_ = np.asarray([0, 1])

        def fit(self, features: np.ndarray, labels: np.ndarray) -> "FakeModel":
            fit_sizes.append(len(features))
            assert len(features) == len(labels)
            return self

        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            return np.tile(np.asarray([[0.25, 0.75]]), (len(features), 1))

    probabilities = cross_fit_probabilities(rows, targets, model_factory=FakeModel)

    assert set(probabilities) == {row.identity for row in rows}
    assert all(probability == 0.75 for probability in probabilities.values())
    assert fit_sizes == [3, 3, 3, 3]


def test_curve_reports_overall_group_and_event_bin_arithmetic() -> None:
    rows = tuple(
        KeepReviewRow(
            fixture=f"sset_{index + 1:02d}",
            group=GROUPS[index // 2],
            span_id=0,
            fps=25.0,
            features=(float(index),) + (0.0,) * 9,
        )
        for index in range(8)
    )
    probabilities = {row.identity: index / 7 for index, row in enumerate(rows)}
    targets = {row.identity: index in {5, 6, 7} for index, row in enumerate(rows)}

    curve = keep_review_curve(rows, probabilities, targets, thresholds=(0.5, 0.9))

    assert curve[0]["accepted_count"] == 4
    assert curve[0]["fully_correct_accepted"] == 3
    assert curve[0]["precision"] == 0.75
    assert curve[0]["coverage"] == 0.5
    assert curve[0]["by_group"]["A"]["accepted_count"] == 0
    assert set(curve[0]["by_event_count"]) == set(EVENT_COUNT_BINS)
    assert choose_candidate_threshold(curve) == 0.9
    assert (
        choose_candidate_threshold(
            (
                {"threshold": 0.5, "precision": 0.91, "coverage": 0.2},
                {"threshold": 0.6, "precision": 0.95, "coverage": 0.1},
            ),
        )
        == 0.5
    )


def test_same_oof_probabilities_can_be_scored_against_five_and_ten_frame_targets() -> None:
    rows = tuple(
        KeepReviewRow(f"sset_{index + 1:02d}", GROUPS[index], 0, 25.0, (0.0,) * 10)
        for index in range(4)
    )
    probabilities = {row.identity: 0.75 for row in rows}
    targets_at_5 = {row.identity: index < 2 for index, row in enumerate(rows)}
    targets_at_10 = {row.identity: index < 3 for index, row in enumerate(rows)}

    curve_at_5 = keep_review_curve(rows, probabilities, targets_at_5, thresholds=(0.5,))
    curve_at_10 = keep_review_curve(rows, probabilities, targets_at_10, thresholds=(0.5,))

    assert curve_at_5[0]["accepted_count"] == curve_at_10[0]["accepted_count"] == 4
    assert curve_at_5[0]["fully_correct_accepted"] == 2
    assert curve_at_10[0]["fully_correct_accepted"] == 3


def test_feature_and_prediction_interfaces_do_not_accept_labels() -> None:
    assert "labels" not in inspect.signature(build_feature_vector).parameters
    assert "labels" not in inspect.signature(predict_keep_probabilities).parameters

    row = KeepReviewRow("sset_01", "A", 0, 25.0, (0.0,) * 10)

    class FakeModel:
        classes_ = np.asarray([0, 1])

        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            assert features.shape == (1, 10)
            return np.asarray([[0.4, 0.6]])

    assert predict_keep_probabilities(FakeModel(), (row,)) == {row.identity: 0.6}
