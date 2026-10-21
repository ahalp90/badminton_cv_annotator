"""The fitted layer bundle works through training, storage and inference."""

import json
from dataclasses import replace

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from threadpoolctl import threadpool_limits

from annotator.models import AnnotatorModels, load_models, save_models
from annotator.outcomes.point_winner import Half
from annotator.sequence.contacts import ContactEvent, ContactSequence
from annotator.sequence.features import (
    PHYSICAL_FEATURE_NAMES,
)
from annotator.sequence.rally_model import (
    RallyLayerModels,
    build_layer_review_features,
    predict_rally_options,
    review_feature_names,
    score_rally_review,
)
from annotator.sequence.rally_options import (
    FEATURE_NAMES,
    RallyContext,
    build_rally_options,
)
from annotator.sequence.repairs import RallyRepairs
from annotator.training.rally_layers import (
    RallyFitConfig,
    fit_rally_models,
    fit_rally_review_model,
)
from annotator.training.sequences import (
    LabelledRally,
    SequenceTrainingVideo,
)
from annotator.training.trees import TreeFitConfig


def training_video(identity: str, group: str, correct: bool) -> SequenceTrainingVideo:
    event = ContactEvent(40, .9, Half.TOP)
    sequence = ContactSequence(7, 10, 80, (event,))
    pool = RallyRepairs(
        (), {},
        {40: np.zeros(len(PHYSICAL_FEATURE_NAMES))},
    )
    label = LabelledRally('rally', (40 if correct else 60,), (Half.TOP,))
    return SequenceTrainingVideo(identity, group, 30.0, (sequence,), (event,), pool, (label,), 100)


def runtime_options():
    video = training_video('validation', 'held', True)
    return build_rally_options(RallyContext(video.fps, video.initial, video.events, video.repairs, video.frame_count))


def dummy_layers() -> RallyLayerModels:
    layers = tuple(
        DummyClassifier(strategy='prior').fit(np.zeros((3, len(FEATURE_NAMES) + 8 * depth)), [0, 1, 1])
        for depth in range(2)
    )
    return RallyLayerModels(layers)


def test_layer_bundle_roundtrip_preserves_predictions_and_review(tmp_path) -> None:
    models = AnnotatorModels(dummy_layers().layers[0], dummy_layers(), None)
    pool = runtime_options()
    before = predict_rally_options(models.sequences, pool)
    save_models(models, tmp_path)
    restored = load_models(tmp_path)
    assert isinstance(restored.sequences, RallyLayerModels)
    after = predict_rally_options(restored.sequences, pool)
    assert after.sequences == before.sequences
    assert after.events == before.events
    for actual, expected in zip(after.scores, before.scores, strict=True):
        np.testing.assert_array_equal(actual, expected)
    review = score_rally_review(after, restored.confidence)
    assert review.features.span_ids == (7,)
    assert review.features.matrix.shape == (1, len(review_feature_names(2)))
    np.testing.assert_allclose(review.scores, [2 / 3])
    metadata_path = tmp_path / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['rally_layers']['option_features'].reverse()
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='features or proposal settings'):
        load_models(tmp_path)


def test_empty_video_keeps_empty_annotation_and_review_shapes() -> None:
    context = replace(runtime_options().context, initial=(), events=())
    result = predict_rally_options(dummy_layers(), build_rally_options(context))
    assert result.sequences == result.events == ()
    review = score_rally_review(result, None)
    assert review.features.matrix.shape == (0, len(review_feature_names(2)))
    assert review.scores.shape == (0,)


def test_review_features_keep_raw_sides_when_finished_sides_change() -> None:
    prediction = predict_rally_options(dummy_layers(), runtime_options())
    features = build_layer_review_features(prediction)
    changed = tuple(replace(sequence, events=tuple(replace(event, side=Half.BOT) for event in sequence.events))
                    for sequence in prediction.sequences)
    revised = build_layer_review_features(replace(prediction, sequences=changed))
    np.testing.assert_array_equal(features.matrix, revised.matrix)
    assert features.matrix.shape[1] == len(review_feature_names(2))


@pytest.mark.parametrize('depth', [1, 2])
def test_real_layer_fit_and_review_use_one_prediction_per_training_video(depth: int) -> None:
    videos = [training_video(f'{group}-{correct}', group, correct) for group in 'ABCD' for correct in (False, True)]
    pytest.importorskip('xgboost')
    config = RallyFitConfig(tree=TreeFitConfig(max_iter=3, min_child_weight=0), depth=depth)
    with threadpool_limits(limits=1):
        result = fit_rally_models(videos, config)
        assert set(result.cross_fitted) == {video.identity for video in videos}
        assert result.span_weight == 1.0
        expected_widths = [len(FEATURE_NAMES) + 8 * layer for layer in range(depth)]
        assert [model.n_features_in_ for model in result.models.layers] == expected_widths
        # The shared contact evidence cannot recover the deliberately absent label.
        ranker = fit_rally_review_model(
            videos, result.cross_fitted,
            TreeFitConfig(max_iter=3, min_child_weight=0),
        )
        prediction = predict_rally_options(result.models, runtime_options())
        review = score_rally_review(prediction, ranker)
    assert review.scores.shape == (1,)
    assert np.isfinite(review.scores).all()


def test_review_cross_fitting_requires_an_extra_group() -> None:
    videos = [training_video(group, group, True) for group in 'AB']
    with pytest.raises(ValueError, match='more groups than layers'):
        fit_rally_models(videos, RallyFitConfig(depth=2))
