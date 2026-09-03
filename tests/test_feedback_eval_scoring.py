"""Scoring, aggregation, and the comparability guard.

A fake scorer stands in for BERTScore throughout, so these run on CPU with no
transformers install and no model download.
"""
import pytest

from feedback_eval.contracts import (
    ComparabilityError,
    PredictionRecord,
    ReferenceRecord,
    RunScore,
    ScoredClip,
    ScorerConfig,
    assert_comparable,
)
from feedback_eval.report import as_dict, render_comparison
from feedback_eval.scoring import score_run


def _pair(clip_id, feedback, player_ids=("player_1",), references=("reference text",)):
    return (
        ReferenceRecord(
            clip_id=clip_id,
            player_ids=tuple(player_ids),
            references=tuple(references),
            source="template",
        ),
        PredictionRecord(clip_id=clip_id, feedback=feedback),
    )


def _fixed_scorer(scores):
    """Scorer returning canned per-candidate scores, recording what it was given."""
    calls = []

    def scorer(candidates, references):
        calls.append((list(candidates), [list(reference) for reference in references]))
        taken = scores[: len(candidates)]
        return taken, taken, taken

    scorer.calls = calls
    return scorer


def _run(clips, model_version="A", config=None):
    return RunScore(
        model_version=model_version,
        scorer=config or ScorerConfig(),
        clips=tuple(clips),
    )


def test_score_run_maps_scores_back_to_their_clips():
    pairs = [_pair("clip_1", "a"), _pair("clip_2", "b")]
    run = score_run(
        pairs,
        model_version="A",
        scorer=_fixed_scorer([0.25, 0.75]),
        config=ScorerConfig(),
    )
    assert [(clip.clip_id, clip.f1) for clip in run.clips] == [("clip_1", 0.25), ("clip_2", 0.75)]
    assert run.mean_f1 == 0.5


def test_score_run_passes_every_reference_phrasing_to_the_scorer():
    """BERTScore keeps the best-matching reference, so all phrasings must reach it."""
    scorer = _fixed_scorer([0.5])
    score_run(
        [_pair("clip_1", "a", references=("first phrasing", "second phrasing"))],
        model_version="A",
        scorer=scorer,
        config=ScorerConfig(),
    )
    assert scorer.calls[0][1] == [["first phrasing", "second phrasing"]]


def test_empty_prediction_scores_zero_and_never_reaches_the_scorer():
    """Dropping empty predictions instead would reward a model for staying silent."""
    scorer = _fixed_scorer([0.8, 0.6])
    run = score_run(
        [_pair("clip_1", "a"), _pair("clip_2", ""), _pair("clip_3", "c")],
        model_version="A",
        scorer=scorer,
        config=ScorerConfig(),
    )
    assert scorer.calls[0][0] == ["a", "c"]
    by_clip = {clip.clip_id: clip for clip in run.clips}
    assert by_clip["clip_2"].f1 == 0.0
    assert by_clip["clip_2"].empty_prediction is True
    assert by_clip["clip_1"].f1 == 0.8
    assert by_clip["clip_3"].f1 == 0.6
    assert run.n_clips == 3
    assert run.n_empty == 1
    assert run.mean_f1 == pytest.approx((0.8 + 0.0 + 0.6) / 3)


def test_all_predictions_empty_still_scores_every_clip():
    run = score_run(
        [_pair("clip_1", ""), _pair("clip_2", "")],
        model_version="A",
        scorer=_fixed_scorer([]),
        config=ScorerConfig(),
    )
    assert run.n_clips == 2
    assert run.mean_f1 == 0.0


def test_score_run_rejects_a_scorer_returning_the_wrong_count():
    with pytest.raises(ValueError, match="scorer returned"):
        score_run(
            [_pair("clip_1", "a"), _pair("clip_2", "b")],
            model_version="A",
            scorer=_fixed_scorer([0.5]),
            config=ScorerConfig(),
        )


def test_score_run_rejects_an_empty_run():
    with pytest.raises(ValueError, match="cannot score an empty run"):
        score_run([], model_version="A", scorer=_fixed_scorer([]), config=ScorerConfig())


def test_mean_f1_by_player_groups_and_averages():
    run = _run(
        [
            ScoredClip("clip_1", ("player_1",), 0.0, 0.0, 0.2),
            ScoredClip("clip_2", ("player_1",), 0.0, 0.0, 0.4),
            ScoredClip("clip_3", ("player_2",), 0.0, 0.0, 0.9),
        ]
    )
    assert run.mean_f1_by_player() == {"player_1": pytest.approx(0.3), "player_2": 0.9}
    assert run.n_players == 2


def test_stdev_f1_is_zero_for_a_single_clip():
    assert _run([ScoredClip("clip_1", ("player_1",), 0.0, 0.0, 0.5)]).stdev_f1 == 0.0


def test_assert_comparable_rejects_a_different_scorer_config():
    """Raw and baseline-rescaled BERTScore are different scales, not different runs."""
    left = _run([ScoredClip("clip_1", ("player_1",), 0.0, 0.0, 0.5)], "A")
    right = _run(
        [ScoredClip("clip_1", ("player_1",), 0.0, 0.0, 0.9)],
        "B",
        config=ScorerConfig(rescale_with_baseline=True),
    )
    with pytest.raises(ComparabilityError, match="scorer config differs"):
        assert_comparable(left, right)


def test_assert_comparable_rejects_a_different_clip_set():
    left = _run([ScoredClip("clip_1", ("player_1",), 0.0, 0.0, 0.5)], "A")
    right = _run([ScoredClip("clip_2", ("player_1",), 0.0, 0.0, 0.9)], "B")
    with pytest.raises(ComparabilityError, match="clip sets differ"):
        assert_comparable(left, right)


def test_assert_comparable_rejects_comparing_a_version_with_itself():
    left = _run([ScoredClip("clip_1", ("player_1",), 0.0, 0.0, 0.5)], "A")
    right = _run([ScoredClip("clip_1", ("player_1",), 0.0, 0.0, 0.9)], "A")
    with pytest.raises(ComparabilityError, match="both runs are model version"):
        assert_comparable(left, right)


def test_assert_comparable_accepts_two_versions_on_one_clip_set():
    left = _run([ScoredClip("clip_1", ("player_1",), 0.0, 0.0, 0.5)], "A")
    right = _run([ScoredClip("clip_1", ("player_1",), 0.0, 0.0, 0.9)], "B")
    assert_comparable(left, right)


def test_render_comparison_reports_the_delta():
    left = _run([ScoredClip("clip_1", ("player_1",), 0.0, 0.0, 0.50)], "A")
    right = _run([ScoredClip("clip_1", ("player_1",), 0.0, 0.0, 0.62)], "B")
    assert "+0.1200" in render_comparison(left, right)


def test_as_dict_stamps_the_scorer_config_into_the_payload():
    """A saved run has to carry the settings that make it comparable later."""
    run = _run(
        [ScoredClip("clip_1", ("player_1",), 0.1, 0.2, 0.3)],
        "A",
        config=ScorerConfig(model_type="roberta-large", rescale_with_baseline=True),
    )
    payload = as_dict(run)
    assert payload["scorer"] == {
        "model_type": "roberta-large",
        "lang": "en",
        "rescale_with_baseline": True,
        "batch_size": 16,
    }
    assert payload["aggregates"]["n_clips"] == 1
    assert payload["clips"][0]["clip_id"] == "clip_1"
