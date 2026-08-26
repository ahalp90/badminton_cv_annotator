"""Pure contract tests for the strict rally evaluator."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import score_contact_rallies as scorer


def _rally(*frames: int, rally_id: str = "set1:1") -> scorer.RallyReference:
    return scorer.RallyReference("sset_01", 0, rally_id, tuple(frames))


def _span(*events: scorer.FixedEvent, start: int = 0, end: int = 100) -> scorer.FixedSpan:
    return scorer.FixedSpan("sset_01", 0, start, end, tuple(events))


def _event(
    frame: int,
    *,
    score: float = 0.9,
    side: str | None = "Top",
) -> scorer.FixedEvent:
    return scorer.FixedEvent("sset_01", frame, score, side)


def _score(
    span: scorer.FixedSpan,
    rallies: list[scorer.RallyReference],
    *,
    target_sides: dict[tuple[str, int], str] | None = None,
    confidence_requirement: float = 0.0,
    tolerance: int = 2,
) -> scorer.SpanScore:
    return scorer.evaluate_span(
        span,
        rallies,
        target_sides or {("sset_01", frame): "Top" for frame in (10, 20)},
        tolerance,
        confidence_requirement,
    )


def test_half_open_span_boundaries_and_unassigned_events() -> None:
    events = {
        "sset_01": (
            _event(10),
            _event(20),
            _event(30),
        )
    }
    evidence = {
        "fixtures": [
            {
                "fixture": "sset_01",
                "spans": [{"span_id": 0, "start_frame": 10, "end_frame": 30}],
            }
        ]
    }

    spans = scorer.fixed_spans_from_evidence(evidence, events)

    assert spans[0].events == (events["sset_01"][0], events["sset_01"][1])
    assert scorer.unassigned_events(spans, events) == (events["sset_01"][2],)


def test_normalise_rallies_preserves_shuttleset_identity() -> None:
    raw = {"sset_01": [SimpleNamespace(set_id="set1", rally=4, stroke_frames=(10, 20))]}

    rallies = scorer.normalise_rallies(raw)

    assert rallies["sset_01"][0] == scorer.RallyReference("sset_01", 0, "set1:4", (10, 20))


def test_exact_span_is_kept_and_fully_correct() -> None:
    result = _score(_span(_event(10), _event(20)), [_rally(10, 20)])

    assert result.kept is True
    assert result.fully_correct is True
    assert result.timing_matches == 2
    assert result.rejection_reasons == ()


def test_empty_span_is_abstained_and_cannot_be_fully_correct() -> None:
    result = _score(_span(), [_rally(10, 20)])

    assert result.kept is False
    assert result.fully_correct is False
    assert result.timing_confidence is None
    assert scorer.REASON_NO_EVENTS in result.rejection_reasons


def test_span_with_no_matching_rally_is_rejected() -> None:
    result = _score(_span(_event(10), _event(20), end=40), [_rally(50, 60)])

    assert result.kept is True
    assert result.fully_correct is False
    assert result.rally_id is None
    assert result.rejection_reasons == (scorer.REASON_NO_RALLY,)


def test_span_overlapping_two_rallies_is_rejected() -> None:
    result = _score(_span(_event(10), _event(20)), [_rally(10, 20), _rally(20, 30, rally_id="set1:2")])

    assert result.kept is True
    assert result.fully_correct is False
    assert scorer.REASON_MULTIPLE_RALLIES in result.rejection_reasons


def test_missing_contact_rejects_without_dropping_the_span() -> None:
    result = _score(_span(_event(10)), [_rally(10, 20)])

    assert result.kept is True
    assert result.fully_correct is False
    assert result.event_count == 1
    assert result.timing_matches == 1
    assert scorer.REASON_MISSING_CONTACT in result.rejection_reasons


def test_extra_event_rejects_without_dropping_the_span() -> None:
    result = _score(_span(_event(10), _event(20), _event(30)), [_rally(10, 20)])

    assert result.kept is True
    assert result.fully_correct is False
    assert result.event_count == 3
    assert scorer.REASON_EXTRA_EVENT in result.rejection_reasons


def test_unanswered_side_rejects_the_whole_span() -> None:
    result = _score(_span(_event(10, side=None), _event(20)), [_rally(10, 20)])

    assert result.kept is False
    assert result.fully_correct is False
    assert result.event_count == 2
    assert result.rejection_reasons == (scorer.REASON_SIDE_UNANSWERED,)


def test_wrong_side_rejects_a_timing_correct_span() -> None:
    result = _score(
        _span(_event(10, side="Bot"), _event(20)),
        [_rally(10, 20)],
    )

    assert result.kept is True
    assert result.fully_correct is False
    assert scorer.REASON_SIDE_INCORRECT in result.rejection_reasons


def test_low_timing_confidence_abstains_on_the_whole_span() -> None:
    result = _score(
        _span(_event(10, score=0.4), _event(20, score=0.9)),
        [_rally(10, 20)],
        confidence_requirement=0.5,
    )

    assert result.kept is False
    assert result.fully_correct is False
    assert result.event_count == 2
    assert result.timing_confidence == 0.4
    assert scorer.REASON_LOW_TIMING_CONFIDENCE in result.rejection_reasons


def test_timing_mismatch_rejects_even_when_event_counts_match() -> None:
    result = _score(_span(_event(10), _event(30)), [_rally(10, 20)], tolerance=1)

    assert result.kept is True
    assert result.fully_correct is False
    assert result.timing_matches == 1
    assert scorer.REASON_MISSING_CONTACT in result.rejection_reasons
    assert scorer.REASON_EXTRA_EVENT in result.rejection_reasons
    assert scorer.REASON_TIMING_MISMATCH in result.rejection_reasons


def test_confidence_curve_abstains_whole_spans_and_counts_full_rallies() -> None:
    spans = (
        _span(_event(10, score=0.4), _event(20, score=0.9)),
        scorer.FixedSpan(
            "sset_01",
            1,
            100,
            200,
            (_event(110, score=0.8), _event(120, score=0.8)),
        ),
    )
    rallies = {"sset_01": (_rally(10, 20), scorer.RallyReference("sset_01", 1, "set1:2", (110, 120)))}
    sides = {("sset_01", frame): "Top" for frame in (10, 20, 110, 120)}

    curve = scorer.confidence_curve(spans, rallies, sides, {"sset_01": 2}, (0.0, 0.5, 0.9))

    assert curve[0]["rallies_kept"] == 2
    assert curve[0]["fully_correct_kept_rallies"] == 2
    assert curve[1]["rallies_kept"] == 1
    assert curve[1]["fully_correct_kept_rallies"] == 1
    assert curve[2]["rallies_kept"] == 0
    assert curve[2]["fully_correct_kept_rallies"] == 0


def test_primary_and_sensitivity_tolerances_use_base30_scaling() -> None:
    spans = (_span(_event(10), _event(20)),)
    rallies = {"sset_01": (_rally(10, 20),)}
    sides = {("sset_01", 10): "Top", ("sset_01", 20): "Top"}

    primary = scorer.score_strict_rallies(
        spans,
        rallies,
        sides,
        {"sset_01": 25.0},
        tolerance_base30=10,
        requirements=(0.0,),
    )
    sensitivity = scorer.score_strict_rallies(
        spans,
        rallies,
        sides,
        {"sset_01": 25.0},
        tolerance_base30=5,
        requirements=(0.0,),
    )

    assert primary["tolerance_frames"] == {"sset_01": 8}
    assert sensitivity["tolerance_frames"] == {"sset_01": 4}


def test_retained_events_keep_score_and_replayed_side() -> None:
    dtype = np.dtype(
        [
            ("fixture", "S7"),
            ("interval_id", "i2"),
            ("frame", "i4"),
            ("timing_score", "f8"),
            ("threshold", "f8"),
            ("decision", "u1"),
        ]
    )
    rows = np.array([(b"sset_01", 0, 10, 0.8, 0.5, 2), (b"sset_01", 0, 20, 0.9, 0.5, 1)], dtype=dtype)

    events = scorer.retained_events_from_scores(rows, {("sset_01", 10): "Bottom"})

    assert events == {"sset_01": (scorer.FixedEvent("sset_01", 10, 0.8, "Bot"),)}


def test_retained_events_reject_duplicate_identities() -> None:
    dtype = np.dtype(
        [
            ("fixture", "S7"),
            ("frame", "i4"),
            ("timing_score", "f8"),
            ("decision", "u1"),
        ]
    )
    rows = np.array([(b"sset_01", 10, 0.8, 2), (b"sset_01", 10, 0.9, 2)], dtype=dtype)

    with pytest.raises(ValueError, match="duplicated"):
        scorer.retained_events_from_scores(rows, {})


def _variant_folds(offset: int) -> list[dict[str, object]]:
    return [
        {
            "test_fixture": fixture,
            "prediction_count": 1,
            "prediction_frames": [offset + index],
        }
        for index, fixture in enumerate(scorer.evidence_freezer.FIXTURE_SPECS)
    ]


def test_prediction_frames_default_to_the_baseline_variant() -> None:
    result = {
        "models": {
            "histogram_boosting": {
                "physics": {"folds": _variant_folds(100)},
            }
        }
    }

    frames = scorer._prediction_frames_from_result(result)

    assert frames["sset_01"].tolist() == [100]


@pytest.mark.parametrize(
    "variant_name",
    (
        scorer.PHYSICS_WITHOUT_RAW_MOTION_VARIANT,
        scorer.PHYSICS_BASE30_MOTION_VARIANT,
    ),
)
def test_prediction_frames_follow_the_verified_trial_variant(variant_name: str) -> None:
    result = {
        "models": {
            "histogram_boosting": {
                "physics": {"folds": _variant_folds(200)},
            }
        }
    }

    frames = scorer._prediction_frames_from_result(
        result,
        variant_name,
    )

    assert frames["sset_01"].tolist() == [200]


def test_candidate_variant_comes_from_the_verified_manifest() -> None:
    verified = SimpleNamespace(
        manifest={"variant": scorer.PHYSICS_WITHOUT_RAW_MOTION_VARIANT},
        tree_result={"selected_variant": scorer.PHYSICS_WITHOUT_RAW_MOTION_VARIANT},
    )

    assert scorer._candidate_variant(verified) == scorer.PHYSICS_WITHOUT_RAW_MOTION_VARIANT


def test_candidate_variant_rejects_inconsistent_tree_result() -> None:
    verified = SimpleNamespace(
        manifest={"variant": scorer.PHYSICS_WITHOUT_RAW_MOTION_VARIANT},
        tree_result={"selected_variant": scorer.PHYSICS_BASE30_MOTION_VARIANT},
    )

    with pytest.raises(ValueError, match="variant differs"):
        scorer._candidate_variant(verified)


def test_prediction_frames_reject_unknown_variant() -> None:
    result = {"models": {"histogram_boosting": {"physics": {"folds": _variant_folds(100)}}}}

    with pytest.raises(ValueError, match="not an allowed rally-scoring variant"):
        scorer._prediction_frames_from_result(result, "region_v2/histogram_boosting/unknown")


def test_shipped_attribution_receives_the_selected_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    import score_contact_player_attribution as attribution_scorer

    captured: dict[str, object] = {}

    monkeypatch.setattr(attribution_scorer, "_load_tree_freezes", lambda _arguments: {})

    def record_variants(_data_root: Path, _freezes: object, variants: object) -> dict[tuple[str, int], str]:
        captured["variants"] = variants
        return {("sset_01", 12): "Top"}

    monkeypatch.setattr(attribution_scorer, "_shipped_attribution_map", record_variants)
    arguments = SimpleNamespace(
        feature_manifest=Path("features.json"),
        tree_results=Path("tree.json.gz"),
        region_v1_manifest=Path("region-v1-features.json"),
        region_v1_results=Path("region-v1-tree.json.gz"),
        data_root=Path("inputs"),
    )
    retained_frames = {"sset_01": np.asarray([12], dtype=np.int32)}

    attribution = scorer._shipped_attribution(
        arguments,
        retained_frames,
        scorer.PHYSICS_WITHOUT_RAW_MOTION_VARIANT,
    )

    assert attribution == {("sset_01", 12): "Top"}
    assert captured["variants"] == {
        scorer.PHYSICS_WITHOUT_RAW_MOTION_VARIANT: retained_frames,
    }
