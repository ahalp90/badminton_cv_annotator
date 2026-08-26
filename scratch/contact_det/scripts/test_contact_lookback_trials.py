"""Focused tests for the bounded serve-lookback decision replay."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import score_contact_lookback_trials as scorer


def _rows(
    frames: tuple[int, ...],
    scores: tuple[float, ...],
    *,
    intervals: tuple[int, ...] | None = None,
    starts: tuple[bool, ...] | None = None,
    lookbacks: tuple[bool, ...] | None = None,
    threshold: float = 0.75,
    fixture: str = "sset_21",
) -> tuple[np.ndarray, np.ndarray]:
    if intervals is None:
        intervals = (0,) * len(frames)
    if starts is None:
        starts = (False,) * len(frames)
    if lookbacks is None:
        lookbacks = (False,) * len(frames)
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
            (scorer.SERVE_LOOKBACK_REGION_FIELD, "u1"),
        ]
    )
    model = np.zeros(len(frames), dtype=model_dtype)
    model["fixture"] = candidates["fixture"]
    model["interval_id"] = candidates["interval_id"]
    model["frame"] = candidates["frame"]
    model[scorer.START_REGION_FIELD] = starts
    model[scorer.SERVE_LOOKBACK_REGION_FIELD] = lookbacks
    return candidates, model


def _retained_frames(rows: np.ndarray) -> list[int]:
    return rows[rows["decision"] == scorer.tree_scorer.CANDIDATE_RETAINED]["frame"].tolist()


def test_lookback_output_keeps_only_bounded_controls_and_policies() -> None:
    assert [trial.trial_id for trial in scorer.TRIALS] == ["B0", "N+", "S−", "L−", "SL−"]
    assert [trial.nms_radius_base30 for trial in scorer.TRIALS] == [5, 6, 5, 5, 5]
    assert [trial.threshold_mode for trial in scorer.TRIALS] == [
        scorer.BASELINE_THRESHOLD,
        scorer.BASELINE_THRESHOLD,
        scorer.LOWER_START_THRESHOLD,
        scorer.LOWER_SERVE_LOOKBACK_THRESHOLD,
        scorer.LOWER_START_OR_SERVE_LOOKBACK_THRESHOLD,
    ]


def test_lookback_masks_lower_exact_grid_point_once_for_overlap() -> None:
    candidates, model = _rows(
        (0, 10, 20, 30),
        (0.72, 0.72, 0.72, 0.72),
        starts=(True, False, True, False),
        lookbacks=(False, True, True, False),
    )

    lookback = scorer._effective_thresholds(candidates, model, scorer.TRIALS[3])
    either_region = scorer._effective_thresholds(candidates, model, scorer.TRIALS[4])

    np.testing.assert_allclose(lookback, [0.75, 0.70, 0.70, 0.75])
    np.testing.assert_allclose(either_region, [0.70, 0.70, 0.70, 0.75])


def test_lookback_replay_keeps_nms_shared_by_interval() -> None:
    candidates, model = _rows(
        (0, 4, 5, 6, 10),
        (0.90, 0.80, 0.85, 0.84, 0.83),
        intervals=(0, 0, 1, 1, 0),
        lookbacks=(True,) * 5,
    )

    rows = scorer.replay_trial(candidates, model, scorer.TRIALS[3])

    # Frames 0, 4 and 10 share interval 0. Frames 5 and 6 share interval 1.
    # The radius is inclusive, so only the strongest row in each local
    # neighbourhood survives.
    assert _retained_frames(rows) == [0, 5, 10]


def test_event_side_metrics_reuse_attribution_match_helpers() -> None:
    empty = np.empty(0, dtype=np.int32)
    predictions = {
        "sset_01": empty,
        "sset_15": empty,
        "sset_21": np.asarray([9, 31], dtype=np.int32),
    }
    ground_truth = scorer.tree_scorer.GroundTruth(
        frames={
            "sset_01": empty,
            "sset_15": empty,
            "sset_21": np.asarray([10, 30], dtype=np.int32),
        },
        serves={"sset_01": set(), "sset_15": set(), "sset_21": {10}},
        rally_count=1,
    )
    trial = scorer.FrozenTrial(
        scorer.TRIALS[0],
        np.empty(0, dtype=scorer.tree_scorer.CANDIDATE_SCORE_DTYPE),
        predictions,
        {("sset_21", 9): "Top", ("sset_21", 31): "Bot"},
        (),
        (),
    )
    target_sides = {("sset_21", 10): "Top", ("sset_21", 30): "Top"}

    result = scorer._event_side_metrics(trial, ground_truth, target_sides)

    pooled = result["10"]["pooled"]
    assert pooled["all"]["side_accuracy"] == pytest.approx(0.5)
    assert pooled["all"]["timing_and_correct_side_recall"] == pytest.approx(0.5)
    assert pooled["all"]["joint_event_and_side_f1"] == pytest.approx(0.5)
    assert pooled["serve"]["timing_and_correct_side_recall"] == pytest.approx(1.0)
    assert result["10"]["fixtures"]["sset_21"]["all"]["correct_side_answers"] == 1


def test_score_freezes_and_replays_side_before_loading_timing_or_side_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    verified = SimpleNamespace(manifest={"feature_sha256": "feature"})
    verified_candidates = SimpleNamespace(
        manifest={
            "tree_result_sha256": "tree",
            "candidate_sha256": "candidate",
        },
        tree_result={},
        rows=np.empty(0),
    )
    evidence = SimpleNamespace(manifest={"evidence_sha256": "evidence"}, evidence={})

    monkeypatch.setattr(
        scorer.tree_scorer,
        "verify_freeze",
        lambda _path: calls.append("verify_features") or verified,
    )
    monkeypatch.setattr(
        scorer.tree_scorer,
        "verify_candidate_scores",
        lambda *_arguments: calls.append("verify_candidates") or verified_candidates,
    )
    monkeypatch.setattr(
        scorer.tree_scorer,
        "_result_variant",
        lambda _result: scorer.tree_scorer.CANDIDATE_VARIANT,
    )
    monkeypatch.setattr(
        scorer.evidence_scorer,
        "verify_freeze",
        lambda _path: calls.append("verify_evidence") or evidence,
    )
    monkeypatch.setattr(
        scorer,
        "_freeze_trial_streams",
        lambda *_arguments: calls.append("freeze_and_replay_side") or (),
    )
    monkeypatch.setattr(scorer, "_model_rows", lambda _verified: np.empty(0))
    monkeypatch.setattr(
        scorer.rally_scorer,
        "_load_timing_rallies",
        lambda: calls.append("timing_labels") or {},
    )
    monkeypatch.setattr(
        scorer.decision_scorer,
        "_ground_truth_from_rallies",
        lambda _rallies: calls.append("ground_truth") or object(),
    )
    monkeypatch.setattr(
        scorer.rally_scorer,
        "_load_side_labels",
        lambda: calls.append("side_labels") or {},
    )
    monkeypatch.setattr(scorer.tree_scorer, "_sha256", lambda _path: "sha")

    scorer.score(
        SimpleNamespace(
            feature_manifest="features",
            tree_results="tree",
            candidate_manifest="candidates",
            evidence_manifest="evidence",
            region_v1_manifest="v1-features",
            region_v1_results="v1-results",
            data_root="data",
        )
    )

    assert calls == [
        "verify_features",
        "verify_candidates",
        "verify_evidence",
        "freeze_and_replay_side",
        "timing_labels",
        "ground_truth",
        "side_labels",
    ]


def test_trial_streams_fix_side_answers_before_any_label_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates, model = _rows((0, 10), (0.90, 0.80))
    candidates["decision"] = scorer.replay_trial(
        candidates,
        model,
        scorer.TRIALS[0],
    )["decision"]
    attribution = {
        ("sset_21", 0): "Top",
        ("sset_21", 10): "Bot",
    }
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

    assert [trial.spec.trial_id for trial in frozen] == ["B0", "N+", "S−", "L−", "SL−"]
    assert set(captured["predictions"]) == {"B0", "N+", "S−", "L−", "SL−"}
    assert all(trial.spans == () for trial in frozen)
    assert all(len(trial.unassigned) == len(trial.predictions["sset_21"]) for trial in frozen)
    assert all(trial.attribution for trial in frozen)
