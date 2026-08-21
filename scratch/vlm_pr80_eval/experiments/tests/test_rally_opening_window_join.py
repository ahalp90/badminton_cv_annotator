from __future__ import annotations

import pytest

from scratch.vlm_pr80_eval.experiments.rally_opening_window_join import (
    _best_span_match,
    _reject_truth_keys,
    _scene_cut_frames,
    opening_window,
)


def test_opening_window_uses_cut_before_first_three_contacts() -> None:
    result = opening_window(
        [1_000, 1_020, 1_040, 1_060],
        [940, 980, 1_030, 1_100],
        fps=30.0,
        total_frames=2_000,
    )

    assert result == {
        "route_selected": True,
        "route_reason": "qualifying_opening_cut",
        "early_contact_frames": [1_000, 1_020, 1_040],
        "qualifying_cut_frames": [940, 980, 1_030],
        "window_start_frame": 790,
        "window_end_frame_exclusive": 1_191,
    }


def test_opening_window_does_not_use_cut_after_third_contact() -> None:
    result = opening_window(
        [1_000, 1_020, 1_040, 1_060],
        [1_041],
        fps=30.0,
        total_frames=2_000,
    )

    assert result["route_selected"] is False
    assert result["route_reason"] == "no_qualifying_cut"
    assert result["window_start_frame"] is None


def test_opening_window_uses_all_contacts_when_fewer_than_three() -> None:
    result = opening_window(
        [50, 75],
        [25],
        fps=25.0,
        total_frames=180,
    )

    assert result["early_contact_frames"] == [50, 75]
    assert result["window_start_frame"] == 0
    assert result["window_end_frame_exclusive"] == 180


def test_opening_window_retains_contactless_rally_with_reason() -> None:
    result = opening_window([], [25], fps=25.0, total_frames=180)

    assert result["route_selected"] is False
    assert result["route_reason"] == "no_accepted_contacts"
    assert result["early_contact_frames"] == []


def test_opening_window_requires_ordered_contacts() -> None:
    with pytest.raises(ValueError, match="must be ordered"):
        opening_window([100, 90], [95], fps=25.0, total_frames=180)


def test_scene_cut_frames_requires_full_contiguous_coverage() -> None:
    assert _scene_cut_frames([[0, 20], [20, 50]], 50) == [20]

    with pytest.raises(ValueError, match="contiguous coverage"):
        _scene_cut_frames([[0, 20], [21, 50]], 50)


def test_inference_manifest_rejects_nested_truth() -> None:
    with pytest.raises(ValueError, match="truth fields"):
        _reject_truth_keys({"cases": [{"reviewed_visibility": "visible"}]})


def test_best_span_match_keeps_unmatched_truth_explicit() -> None:
    cases = [
        {
            "case_id": "case-1",
            "automatic_span_start_frame": 100,
            "automatic_span_end_frame_exclusive": 200,
        }
    ]

    assert _best_span_match(300, 400, cases) is None
