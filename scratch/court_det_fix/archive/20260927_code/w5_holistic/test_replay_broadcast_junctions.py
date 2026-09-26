from __future__ import annotations

import numpy as np
import pytest

from scratch.court_det_fix.w5_holistic.replay_broadcast_junctions import (
    MASK_POLICIES,
    median_two_of_three_boxes,
    no_mask_boxes,
    summarise,
    validate_historical_orders,
)


def _sample(frame_index: int, boxes: list[list[float]], scores: list[float]) -> dict:
    return {"frame_index": frame_index, "bboxes": boxes, "scores": scores}


def test_no_mask_has_box_shape() -> None:
    boxes = no_mask_boxes()

    assert boxes.shape == (0, 4)
    assert boxes.dtype == float


def test_median_mask_uses_two_frames_and_filters_low_scores() -> None:
    case = {
        "id": "median",
        "dimensions": {"width": 100, "height": 80},
        "provenance": {"person_score_cutoff": 0.2},
        "pose_samples": [
            _sample(1, [[0, 0, 10, 10], [40, 40, 60, 60]], [0.9, 0.1]),
            _sample(2, [[5, 0, 15, 10], [30, 30, 35, 35]], [0.8, 0.2]),
            _sample(3, [[5, 5, 10, 15]], [0.7]),
        ],
    }

    boxes = median_two_of_three_boxes(case)

    np.testing.assert_array_equal(boxes, np.asarray([[5, 0, 10, 10], [5, 5, 10, 10]], dtype=float))


def test_median_mask_deduplicates_pair_intersections() -> None:
    case = {
        "id": "same",
        "dimensions": {"width": 100, "height": 80},
        "provenance": {"person_score_cutoff": 0.2},
        "pose_samples": [
            _sample(1, [[1, 2, 9, 10]], [1.0]),
            _sample(2, [[1, 2, 9, 10]], [1.0]),
            _sample(3, [[1, 2, 9, 10]], [1.0]),
        ],
    }

    boxes = median_two_of_three_boxes(case)

    np.testing.assert_array_equal(boxes, np.asarray([[1, 2, 9, 10]], dtype=float))


def test_median_mask_keeps_shared_box_boundaries() -> None:
    case = {
        "id": "boundary",
        "dimensions": {"width": 20, "height": 20},
        "provenance": {"person_score_cutoff": 0.2},
        "pose_samples": [
            _sample(1, [[0, 0, 5, 5]], [1.0]),
            _sample(2, [[5, 0, 10, 5]], [1.0]),
            _sample(3, [], []),
        ],
    }

    boxes = median_two_of_three_boxes(case)

    np.testing.assert_array_equal(boxes, np.asarray([[5, 0, 5, 5]], dtype=float))


def test_intersection_boxes_match_pointwise_two_of_three_rule() -> None:
    samples = [
        _sample(1, [[0, 0, 6, 6], [12, 0, 18, 6]], [1.0, 1.0]),
        _sample(2, [[3, 2, 9, 8]], [1.0]),
        _sample(3, [[4, 4, 14, 10]], [1.0]),
    ]
    case = {
        "id": "pointwise",
        "dimensions": {"width": 20, "height": 12},
        "provenance": {"person_score_cutoff": 0.2},
        "pose_samples": samples,
    }
    intersections = median_two_of_three_boxes(case)
    points = np.asarray([[x, y] for x in range(20) for y in range(12)], dtype=float)

    frame_coverage = []
    for sample in samples:
        boxes = np.asarray(sample["bboxes"], dtype=float)
        inside = ((points[:, None] >= boxes[None, :, :2]) & (points[:, None] <= boxes[None, :, 2:])).all(axis=2)
        frame_coverage.append(inside.any(axis=1))
    expected = np.stack(frame_coverage).sum(axis=0) >= 2
    actual = (
        (points[:, None] >= intersections[None, :, :2])
        & (points[:, None] <= intersections[None, :, 2:])
    ).all(axis=2).any(axis=1)

    np.testing.assert_array_equal(actual, expected)


def test_historical_order_gate_fails_closed() -> None:
    measured = {"orders": {"original": ["new"], "paint_qualified": ["new"]}}
    expected = {"original": ["old"], "paint_qualified": ["old"]}

    with pytest.raises(ValueError, match="do not reproduce"):
        validate_historical_orders("case", measured, expected)


def test_summary_keeps_unverified_and_empty_cases_out_of_denominator() -> None:
    def policy(error: float | None, same: bool = True) -> dict:
        first = None if error is None else {"id": "winner", "metrics": {"corner_max_error_px": error}}
        second_id = "winner" if same else "other"
        second = None if error is None else {"id": second_id, "metrics": {"corner_max_error_px": error}}
        return {"winners": {"original": first, "paint_qualified": second}}

    cases = [
        {
            "id": "accurate",
            "scoreable": True,
            "eligible_legacy_count": 1,
            "policies": {name: policy(10.0) for name in MASK_POLICIES},
        },
        {
            "id": "wrong",
            "scoreable": True,
            "eligible_legacy_count": 1,
            "policies": {name: policy(20.0, same=False) for name in MASK_POLICIES},
        },
        {
            "id": "unverified",
            "scoreable": False,
            "eligible_legacy_count": 1,
            "policies": {name: policy(1.0) for name in MASK_POLICIES},
        },
        {
            "id": "empty",
            "scoreable": False,
            "eligible_legacy_count": 0,
            "policies": {name: policy(None) for name in MASK_POLICIES},
        },
    ]

    summary = summarise(cases)

    assert summary["scoreable_case_count"] == 2
    assert summary["unverified_case_ids"] == ["unverified", "empty"]
    assert summary["empty_pool_case_ids"] == ["empty"]
    for policy_name in MASK_POLICIES:
        assert summary["policies"][policy_name]["original"] == {"within_15_px": 1, "denominator": 2}
        assert summary["policies"][policy_name]["original_vs_paint_qualified"] == {
            "same_winner": 2,
            "comparable_cases": 3,
        }
