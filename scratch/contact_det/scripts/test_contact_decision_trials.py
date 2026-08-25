"""Focused tests for the fixed held-out-score decision trials."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import score_contact_decision_trials as scorer


def _rows(
    frames: tuple[int, ...],
    scores: tuple[float, ...],
    *,
    intervals: tuple[int, ...] | None = None,
    starts: tuple[bool, ...] | None = None,
    threshold: float = 0.75,
    fixture: str = "sset_21",
) -> tuple[np.ndarray, np.ndarray]:
    if intervals is None:
        intervals = (0,) * len(frames)
    if starts is None:
        starts = (False,) * len(frames)
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
            (scorer.START_REGION_FIELD, "u1"),
        ]
    )
    model = np.zeros(len(frames), dtype=model_dtype)
    model["fixture"] = candidates["fixture"]
    model["interval_id"] = candidates["interval_id"]
    model["frame"] = candidates["frame"]
    model[scorer.START_REGION_FIELD] = starts
    return candidates, model


def _retained_frames(rows: np.ndarray) -> list[int]:
    return rows[rows["decision"] == scorer.tree_scorer.CANDIDATE_RETAINED]["frame"].tolist()


def test_trial_list_is_the_frozen_one_factor_grid() -> None:
    assert [trial.trial_id for trial in scorer.TRIALS] == ["B0", "T−", "N−", "N+", "S−"]
    assert [trial.nms_radius_base30 for trial in scorer.TRIALS] == [5, 5, 4, 6, 5]
    assert [trial.threshold_mode for trial in scorer.TRIALS] == [
        scorer.BASELINE_THRESHOLD,
        scorer.LOWER_THRESHOLD,
        scorer.BASELINE_THRESHOLD,
        scorer.BASELINE_THRESHOLD,
        scorer.LOWER_START_THRESHOLD,
    ]


def test_threshold_neighbours_come_from_the_exact_frozen_grid() -> None:
    assert scorer._previous_threshold(0.7999999999999999) == 0.75
    assert scorer._previous_threshold(0.75) == 0.7
    assert scorer._next_threshold(0.75) == 0.7999999999999999
    with pytest.raises(ValueError, match="no preceding"):
        scorer._previous_threshold(0.05)
    with pytest.raises(ValueError, match="no preceding"):
        scorer._previous_threshold(0.77)
    with pytest.raises(ValueError, match="no preceding"):
        scorer._previous_threshold(0.7500000000005)


def test_lower_threshold_and_start_only_threshold_are_distinct() -> None:
    candidates, model = _rows(
        (0, 10, 20),
        (0.72, 0.72, 0.80),
        starts=(True, False, False),
    )

    baseline = scorer.replay_trial(candidates, model, scorer.TRIALS[0])
    lower = scorer.replay_trial(candidates, model, scorer.TRIALS[1])
    start_only = scorer.replay_trial(candidates, model, scorer.TRIALS[4])

    assert _retained_frames(baseline) == [20]
    assert _retained_frames(lower) == [0, 10, 20]
    assert _retained_frames(start_only) == [0, 20]
    np.testing.assert_array_equal(start_only["threshold"], [0.7, 0.75, 0.75])


def test_nms_rows_use_one_global_pool_and_the_existing_tie_order() -> None:
    candidates, model = _rows(
        (0, 4, 5, 6),
        (0.90, 0.80, 0.85, 0.84),
    )

    narrow = scorer.replay_trial(candidates, model, scorer.TRIALS[2])
    baseline = scorer.replay_trial(candidates, model, scorer.TRIALS[0])
    wide = scorer.replay_trial(candidates, model, scorer.TRIALS[3])

    assert _retained_frames(narrow) == [0, 5]
    assert _retained_frames(baseline) == [0, 6]
    assert _retained_frames(wide) == [0]


def test_nms_never_suppresses_across_intervals() -> None:
    candidates, model = _rows(
        (10, 11),
        (0.90, 0.80),
        intervals=(0, 1),
    )

    rows = scorer.replay_trial(candidates, model, scorer.TRIALS[3])

    assert _retained_frames(rows) == [10, 11]


def test_25_fps_nms_scales_and_suppresses_at_the_inclusive_radius() -> None:
    candidates, model = _rows(
        (0, 4, 5),
        (0.90, 0.85, 0.80),
        threshold=0.7999999999999999,
        fixture="sset_01",
    )

    rows = scorer.replay_trial(candidates, model, scorer.TRIALS[0])

    assert _retained_frames(rows) == [0, 5]


def test_replay_requires_stable_feature_identities_and_exact_b0() -> None:
    candidates, model = _rows((0, 10), (0.90, 0.80))
    baseline = scorer.replay_trial(candidates, model, scorer.TRIALS[0])
    candidates["decision"] = baseline["decision"]

    rows_by_trial = scorer.replay_all_trials(candidates, model)
    assert np.array_equal(rows_by_trial["B0"], candidates)

    changed = candidates.copy()
    changed["decision"][0] = scorer.tree_scorer.CANDIDATE_BELOW_THRESHOLD
    with pytest.raises(ValueError, match="B0 replay differs"):
        scorer.replay_all_trials(changed, model)

    model["frame"][0] = 1
    with pytest.raises(ValueError, match="identities differ"):
        scorer.replay_trial(candidates, model, scorer.TRIALS[0])


def test_ground_truth_is_derived_from_the_single_timing_rally_load() -> None:
    rallies = {
        "sset_01": (
            scorer.rally_scorer.RallyReference("sset_01", 0, "one", (10, 20)),
        ),
        "sset_15": (
            scorer.rally_scorer.RallyReference("sset_15", 0, "two", (30,)),
        ),
        "sset_21": (
            scorer.rally_scorer.RallyReference("sset_21", 0, "three", (40, 50, 60)),
        ),
    }

    ground_truth = scorer._ground_truth_from_rallies(rallies)

    assert ground_truth.rally_count == 3
    assert ground_truth.frames["sset_01"].tolist() == [10, 20]
    assert ground_truth.serves == {
        "sset_01": {10},
        "sset_15": {30},
        "sset_21": {40},
    }


def test_trial_streams_are_fully_constructed_before_label_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates, model = _rows((0, 10), (0.90, 0.80))
    candidates["decision"] = scorer.replay_trial(candidates, model, scorer.TRIALS[0])["decision"]
    attribution = {("sset_21", 0): "Top", ("sset_21", 10): "Bot"}
    captured: dict[str, object] = {}

    def record_predictions(_arguments: object, predictions: object) -> dict[tuple[str, int], str]:
        captured["predictions"] = predictions
        return attribution

    monkeypatch.setattr(scorer, "_replay_attribution", record_predictions)
    evidence = SimpleNamespace(evidence={"fixtures": []})

    frozen = scorer._freeze_trial_streams(
        SimpleNamespace(),
        evidence,
        candidates,
        model,
    )

    assert [trial.spec.trial_id for trial in frozen] == ["B0", "T−", "N−", "N+", "S−"]
    assert set(captured["predictions"]) == {"B0", "T−", "N−", "N+", "S−"}
    assert all(trial.spans == () for trial in frozen)
    assert all(len(trial.unassigned) == len(trial.predictions["sset_21"]) for trial in frozen)
