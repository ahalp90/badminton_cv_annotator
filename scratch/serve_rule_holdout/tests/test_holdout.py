"""Focused checks for the frozen issue #90 evaluation."""

from __future__ import annotations

from dataclasses import fields

import numpy as np

from annotator.inpaint_guard import DEGRADED, FABRICATED, NO_FLAG
from scratch.serve_rule_holdout.frozen_predictions import (
    MeasuredPath,
    PreVerdict,
    PredictionRow,
    h3_outgoing,
    h3_verdict,
    rebuild_guard_codes,
)
from scratch.serve_rule_holdout.score_predictions import (
    TemporalScore,
    _pr88_temporal,
    _span_overlap_counts,
    _temporal_score,
)
from scratch.serve_start_trajectory_exploration.trajectory_features import (
    IncomingMotion,
    RobustDistanceTrend,
)


def _path(*, ratio: float, fitted_decrease: float) -> MeasuredPath:
    motion = IncomingMotion(5, 1.0, 0.5, 0.5, 1.0, 0.5, ratio)
    trend = RobustDistanceTrend(0.0, 0.0, fitted_decrease, 0.0, 0.0)
    return MeasuredPath(motion, trend, 2)


def test_h3_verdict_keeps_frozen_inclusive_boundaries() -> None:
    assert h3_verdict(_path(ratio=8.0, fitted_decrease=0.05), 2) is PreVerdict.INCOMING
    assert h3_verdict(_path(ratio=8.0, fitted_decrease=0.049), 2) is PreVerdict.NOT_INCOMING
    assert h3_verdict(_path(ratio=8.001, fitted_decrease=0.05), 2) is PreVerdict.UNAVAILABLE


def test_h3_outgoing_uses_the_frozen_negative_trend_threshold() -> None:
    assert h3_outgoing(_path(ratio=8.0, fitted_decrease=-0.05), 2)
    assert not h3_outgoing(_path(ratio=8.0, fitted_decrease=-0.049), 2)


def test_three_frame_guard_only_shrinks_the_production_halo() -> None:
    track = np.column_stack(
        (
            np.arange(30, dtype=float),
            np.arange(30, dtype=float),
            np.ones(30, dtype=float),
        )
    )
    track[15, :2] = (4.0, 4.0)
    production = np.full(30, DEGRADED, dtype=np.uint8)
    production[15] = FABRICATED
    rebuilt = rebuild_guard_codes(track, production, 3)
    assert np.all(rebuilt[12:15] == DEGRADED)
    assert rebuilt[15] == FABRICATED
    assert np.all(rebuilt[16:19] == DEGRADED)
    assert rebuilt[11] == NO_FLAG
    assert rebuilt[19] == NO_FLAG
    assert rebuilt[4] == DEGRADED


def test_prediction_schema_has_no_ground_truth_or_score_fields() -> None:
    names = {field.name.lower() for field in fields(PredictionRow)}
    forbidden = ("gt_", "truth", "label", "correct")
    assert not {name for name in names if any(token in name for token in forbidden)}


def test_temporal_scoring_uses_the_predeclared_tolerance() -> None:
    serve = _temporal_score("serve", 102, (100, 130), 30.0)
    returned = _temporal_score("return", 130, (100, 130), 30.0)
    unmatched = _temporal_score("serve", 111, (100, 130), 30.0)
    assert serve.correct
    assert returned.correct
    assert unmatched.gt_label == "unmatched"
    assert not unmatched.correct


def test_pr88_unresolved_category_preserves_the_pr82_temporal_answer() -> None:
    baseline = TemporalScore("serve", 100, "contact_1", True, False)
    prediction = {
        "pr88_category": "not_enough_shuttle_trajectory_to_tell",
        "pr88_selected_frame": "130",
    }
    assert _pr88_temporal(prediction, baseline, (100, 130), 30.0) is baseline


def test_partial_rally_overlap_prevents_a_one_to_one_span() -> None:
    spans = [(100, 200), (300, 400)]
    rally_extents = [(90, 110), (150, 180), (320, 380)]
    assert _span_overlap_counts(spans, rally_extents) == {0: 2, 1: 1}
