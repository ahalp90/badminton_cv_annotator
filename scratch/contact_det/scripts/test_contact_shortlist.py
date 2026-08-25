"""Focused tests for the frozen Phase 3 local contact shortlist."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import score_contact_shortlist as scorer


def _rows(
    frames: tuple[int, ...],
    scores: tuple[float, ...],
    *,
    intervals: tuple[int, ...] | None = None,
    fixture: str = "sset_21",
    threshold: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    if intervals is None:
        intervals = (0,) * len(frames)
    candidates = np.zeros(len(frames), dtype=scorer.tree_scorer.CANDIDATE_SCORE_DTYPE)
    candidates["fixture"] = fixture.encode("ascii")
    candidates["interval_id"] = intervals
    candidates["frame"] = frames
    candidates["timing_score"] = scores
    candidates["threshold"] = threshold

    model_dtype = np.dtype(
        [
            ("fixture", "S7"),
            ("interval_id", "<i2"),
            ("frame", "<i4"),
            (scorer.decision_scorer.START_REGION_FIELD, "u1"),
        ]
    )
    model = np.zeros(len(frames), dtype=model_dtype)
    model["fixture"] = candidates["fixture"]
    model["interval_id"] = candidates["interval_id"]
    model["frame"] = candidates["frame"]

    baseline = scorer.decision_scorer.replay_trial(
        candidates,
        model,
        scorer.decision_scorer.TRIALS[0],
    )
    candidates["decision"] = baseline["decision"]
    return candidates, model


def _frames(rows: np.ndarray, indices: np.ndarray) -> list[int]:
    return rows["frame"][indices].tolist()


def test_shortlist_keeps_anchor_and_best_below_cutoff_alternative() -> None:
    candidates, model = _rows(
        (0, 6, 7, 10, 11),
        (0.90, 0.74, 0.70, 0.71, 0.74),
    )

    frozen = scorer.freeze_shortlist(candidates, model)

    assert _frames(frozen.nplus_rows, frozen.anchor_indices) == [0]
    assert _frames(frozen.nplus_rows, frozen.alternative_indices) == [10]
    assert _frames(frozen.nplus_rows, frozen.shortlist_indices) == [0, 10]
    assert frozen.nplus_rows["decision"][3] == scorer.tree_scorer.CANDIDATE_BELOW_THRESHOLD


def test_equal_score_uses_the_earlier_distinct_frame() -> None:
    candidates, model = _rows(
        (0, 7, 10),
        (0.90, 0.70, 0.70),
    )

    frozen = scorer.freeze_shortlist(candidates, model)

    assert _frames(frozen.nplus_rows, frozen.alternative_indices) == [7]


def test_25_fps_scales_the_inclusive_area_and_exclusion_boundaries() -> None:
    candidates, model = _rows(
        (0, 5, 6, 8, 9),
        (0.90, 0.74, 0.70, 0.71, 0.79),
        fixture="sset_01",
        threshold=0.7999999999999999,
    )

    frozen = scorer.freeze_shortlist(candidates, model)

    assert _frames(frozen.nplus_rows, frozen.alternative_indices) == [8]


def test_shortlist_area_never_crosses_interval_boundary() -> None:
    candidates, model = _rows(
        (0, 7, 8),
        (0.90, 0.74, 0.70),
        intervals=(0, 1, 0),
    )

    frozen = scorer.freeze_shortlist(candidates, model)

    assert _frames(frozen.nplus_rows, frozen.alternative_indices) == [8]


def test_overlapping_anchor_areas_deduplicate_alternative_identities() -> None:
    candidates, model = _rows(
        (0, 7, 14),
        (0.90, 0.70, 0.85),
    )

    frozen = scorer.freeze_shortlist(candidates, model)

    assert _frames(frozen.nplus_rows, frozen.anchor_indices) == [0, 14]
    assert _frames(frozen.nplus_rows, frozen.alternative_indices) == [7, 7]
    assert _frames(frozen.nplus_rows, frozen.shortlist_indices) == [0, 7, 14]


def test_shortlist_union_preserves_source_row_order() -> None:
    candidates, model = _rows(
        (7, 0, 10),
        (0.70, 0.90, 0.69),
    )

    frozen = scorer.freeze_shortlist(candidates, model)

    assert frozen.shortlist_indices.tolist() == [0, 1]
    assert _frames(frozen.nplus_rows, frozen.shortlist_indices) == [7, 0]


def test_shortlist_fails_loudly_when_an_anchor_has_no_distinct_alternative() -> None:
    candidates, model = _rows(
        (0, 6),
        (0.90, 0.70),
    )

    with pytest.raises(ValueError, match="no distinct shortlist alternative"):
        scorer.freeze_shortlist(candidates, model)


def test_frozen_count_gate_rejects_any_other_identity_set() -> None:
    candidates, model = _rows(
        (0, 7),
        (0.90, 0.70),
    )
    frozen = scorer.freeze_shortlist(candidates, model)

    with pytest.raises(ValueError, match="identity counts differ"):
        scorer.validate_frozen_shortlist(frozen)


def test_gate_requires_both_bounded_size_and_half_the_remaining_misses() -> None:
    assert scorer._gate_result(3_238, 6_305, 152)["decision"] == "continue"
    assert scorer._gate_result(3_238, 6_305, 151)["decision"] == "stop"
    assert scorer._gate_result(3_238, 6_477, 152)["decision"] == "stop"


def test_matched_contact_identities_use_fixture_and_ground_truth_row() -> None:
    frames = {
        fixture: np.empty(0, dtype=np.int32)
        for fixture in scorer.tree_scorer.FIXTURE_SPECS
    }
    predictions = {fixture: values.copy() for fixture, values in frames.items()}
    frames["sset_21"] = np.asarray([0, 20], dtype=np.int32)
    predictions["sset_21"] = np.asarray([1, 19], dtype=np.int32)
    ground_truth = scorer.tree_scorer.GroundTruth(
        frames=frames,
        serves={fixture: set() for fixture in scorer.tree_scorer.FIXTURE_SPECS},
        rally_count=0,
    )

    identities = scorer._matched_contact_identities(ground_truth, predictions, 10)

    assert identities == {("sset_21", 0), ("sset_21", 1)}


def test_score_freezes_and_validates_shortlist_before_loading_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    verified = SimpleNamespace(manifest={"feature_sha256": "feature"})
    verified_candidates = SimpleNamespace(
        rows=np.empty(0),
        tree_result={},
        manifest={"tree_result_sha256": "tree", "candidate_sha256": "candidate"},
    )
    frozen = SimpleNamespace()

    monkeypatch.setattr(scorer.tree_scorer, "verify_freeze", lambda _path: verified)
    monkeypatch.setattr(
        scorer.tree_scorer,
        "verify_candidate_scores",
        lambda *_arguments: verified_candidates,
    )
    monkeypatch.setattr(
        scorer.tree_scorer,
        "_result_variant",
        lambda _result: scorer.tree_scorer.CANDIDATE_VARIANT,
    )
    monkeypatch.setattr(scorer.decision_scorer, "_model_rows", lambda _verified: np.empty(0))

    def freeze(_candidate_rows: object, _model_rows: object) -> object:
        calls.append("freeze")
        return frozen

    def validate(_frozen: object) -> None:
        calls.append("validate")

    def load_ground_truth() -> object:
        calls.append("labels")
        return object()

    monkeypatch.setattr(scorer, "freeze_shortlist", freeze)
    monkeypatch.setattr(scorer, "validate_frozen_shortlist", validate)
    monkeypatch.setattr(scorer.tree_scorer, "_load_ground_truth", load_ground_truth)
    monkeypatch.setattr(scorer, "_selected_trial", lambda: object())
    monkeypatch.setattr(
        scorer,
        "_result_payload",
        lambda *_arguments: calls.append("score") or {},
    )

    result = scorer.score(
        SimpleNamespace(
            feature_manifest="features",
            candidate_manifest="candidates",
            tree_results="tree",
        )
    )

    assert result == {}
    assert calls == ["freeze", "validate", "labels", "score"]
