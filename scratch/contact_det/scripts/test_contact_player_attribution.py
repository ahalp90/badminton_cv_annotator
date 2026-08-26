"""Focused tests for contact player-attribution scoring."""

# ruff: noqa: E402

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import score_contact_player_attribution as scorer
import score_tree_contact_detector as tree_scorer


def _match(
    *,
    serve: bool,
    predicted_half: str | None,
    target_half: str = "Top",
    frame: int = 10,
) -> scorer.AttributedMatch:
    return scorer.AttributedMatch("sset_01", frame, frame, serve, predicted_half, target_half)


def test_metric_slice_separates_timing_coverage_accuracy_and_joint_score() -> None:
    matches = [
        _match(serve=True, predicted_half="Top", frame=10),
        _match(serve=False, predicted_half="Bot", frame=20),
        _match(serve=False, predicted_half=None, frame=30),
    ]

    metrics = scorer._metric_slice(matches, ground_truth_count=4, prediction_count=5)

    assert metrics == {
        "ground_truth_contacts": 4,
        "timing_matches": 3,
        "timing_recall": 0.75,
        "side_answers": 2,
        "side_answer_coverage": 2 / 3,
        "correct_side_answers": 1,
        "side_accuracy": 0.5,
        "timing_and_correct_side_recall": 0.25,
        "event_predictions": 5,
        "joint_event_and_side_precision": 0.2,
        "joint_event_and_side_recall": 0.25,
        "joint_event_and_side_f1": 2 * 0.2 * 0.25 / 0.45,
    }


def test_score_slices_uses_matched_gt_identity_for_serve_split() -> None:
    matches = [
        _match(serve=True, predicted_half="Top", frame=10),
        _match(serve=False, predicted_half="Top", frame=20),
        _match(serve=False, predicted_half="Bot", frame=30),
    ]

    score = scorer._score_slices(matches, {"all": 5, "serve": 2, "non_serve": 3}, prediction_count=4)

    assert score["serve"]["timing_matches"] == 1
    assert score["serve"]["timing_recall"] == 0.5
    assert score["non_serve"]["timing_matches"] == 2
    assert score["non_serve"]["timing_recall"] == 2 / 3
    assert score["all"]["event_predictions"] == 4
    assert "event_predictions" not in score["serve"]


@pytest.mark.parametrize(
    ("top_gap", "bot_gap", "expected"),
    [
        (0.2, 0.3, "Top"),
        (0.3, 0.2, "Bot"),
        (0.2, 0.2, "Top"),
        (0.2, np.nan, "Top"),
        (np.nan, 0.2, "Bot"),
        (np.nan, np.nan, None),
        (np.inf, np.inf, None),
    ],
)
def test_nearest_tracked_player_matches_nanargmin_convention(
    top_gap: float, bot_gap: float, expected: str | None
) -> None:
    assert scorer.nearest_tracked_player(top_gap, bot_gap) == expected


def test_timing_count_guard_accepts_retained_counts_and_rejects_drift() -> None:
    ground_truth = tree_scorer.GroundTruth(
        frames={"sset_01": np.array([10, 20], dtype=np.int32)},
        serves={"sset_01": {10}},
        rally_count=1,
    )
    predictions = {"sset_01": np.array([11, 40], dtype=np.int32)}
    metrics = {
        str(tolerance): tree_scorer._event_counts(
            ground_truth, predictions, tolerance, ["sset_01"]
        )
        for tolerance in scorer.TOLERANCES_BASE30
    }

    scorer._assert_timing_metrics(predictions, ground_truth, {"metrics": metrics}, "test")

    metrics["10"] = dict(metrics["10"], matched=2)
    with pytest.raises(AssertionError, match="retained timing counts changed"):
        scorer._assert_timing_metrics(predictions, ground_truth, {"metrics": metrics}, "test")


def test_tree_match_uses_attribution_at_prediction_frame() -> None:
    ground_truth = tree_scorer.GroundTruth(
        frames={"sset_01": np.array([10], dtype=np.int32)},
        serves={"sset_01": {10}},
        rally_count=1,
    )
    predictions = {"sset_01": np.array([12], dtype=np.int32)}
    attribution = {("sset_01", 12): "Bot"}
    sides = {("sset_01", 10): "Top"}

    matches = scorer._tree_matches(predictions, attribution, sides, ground_truth, tolerance_base30=5)

    assert matches == [scorer.AttributedMatch("sset_01", 10, 12, True, "Bot", "Top")]


def test_write_results_is_deterministic_gzip(tmp_path: Path) -> None:
    first = tmp_path / "first.json.gz"
    second = tmp_path / "second.json.gz"
    payload = {"schema": scorer.RESULTS_SCHEMA, "value": [2, 1]}

    scorer.write_results(first, payload)
    scorer.write_results(second, payload)

    assert first.read_bytes() == second.read_bytes()
    with gzip.open(first, "rt", encoding="utf-8") as source:
        assert json.load(source) == payload
