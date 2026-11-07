"""The fitted layer bundle works through training, storage and inference."""

import json
from dataclasses import replace

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from threadpoolctl import threadpool_limits

from annotator.contacts.build_features import CONTACT_FEATURE_NAMES
from annotator.models import AnnotatorModels, load_models, save_models
from annotator.rallies.build_contact_proposals import (
    FEATURE_NAMES,
    RallyContext,
    build_rally_options,
)
from annotator.rallies.refine_rallies import (
    REVIEW_FEATURE_NAMES,
    RallySelector,
    build_review_features,
    predict_rally_options,
    score_rally_review,
)
from annotator.training.rally import (
    RallyFitConfig,
    fit_rally_models,
    fit_rally_review_model,
)
from annotator.training.sequences import (
    LabelledRally,
    SequenceTrainingVideo,
)
from annotator.training.trees import TreeFitConfig
from annotator.types import ContactEvent, ContactSequence, CourtHalf


def training_video(identity: str, group: str, correct: bool) -> SequenceTrainingVideo:
    event = ContactEvent(40, .9, CourtHalf.TOP)
    sequence = ContactSequence(7, 10, 80, (event,))
    physical = {40: np.zeros(len(CONTACT_FEATURE_NAMES))}
    label = LabelledRally('rally', (40 if correct else 60,), (CourtHalf.TOP,))
    context = RallyContext(
        30.0, (sequence,), (event,), {}, physical, frame_count=100,
    )
    return SequenceTrainingVideo(identity, group, context, (label,))


def runtime_options():
    video = training_video('validation', 'held', True)
    return build_rally_options(video.context)


def dummy_layers() -> RallySelector:
    model = DummyClassifier(strategy='prior').fit(np.zeros((3, len(FEATURE_NAMES))), [0, 1, 1])
    return RallySelector(model)


def test_layer_bundle_roundtrip_preserves_predictions_and_review(tmp_path) -> None:
    review = DummyClassifier(strategy='prior').fit(np.zeros((3, len(REVIEW_FEATURE_NAMES))), [0, 1, 1])
    models = AnnotatorModels(dummy_layers().selector, dummy_layers(), review)
    pool = runtime_options()
    before = predict_rally_options(models.rally_selector, pool)
    save_models(models, tmp_path)
    restored = load_models(tmp_path)
    assert isinstance(restored.rally_selector, RallySelector)
    after = predict_rally_options(restored.rally_selector, pool)
    assert after.sequences == before.sequences
    assert after.events == before.events
    np.testing.assert_array_equal(after.scores, before.scores)
    review = score_rally_review(after, restored.review_model)
    assert review.features.span_ids == (7,)
    assert review.features.matrix.shape == (1, len(REVIEW_FEATURE_NAMES))
    np.testing.assert_allclose(review.scores, [2 / 3])
    metadata_path = tmp_path / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['rally_selector']['option_features'].reverse()
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='features or proposal settings'):
        load_models(tmp_path)


def test_empty_video_keeps_empty_annotation_and_review_shapes() -> None:
    context = replace(runtime_options().context, initial=(), events=())
    result = predict_rally_options(dummy_layers(), build_rally_options(context))
    assert result.sequences == result.events == ()
    review = build_review_features(result)
    assert review.matrix.shape == (0, len(REVIEW_FEATURE_NAMES))


def test_review_features_keep_raw_sides_when_finished_sides_change() -> None:
    prediction = predict_rally_options(dummy_layers(), runtime_options())
    features = build_review_features(prediction)
    changed = tuple(
        replace(rally, sequence=replace(
            rally.sequence, events=tuple(replace(event, side=CourtHalf.BOT) for event in rally.sequence.events),
        ))
        for rally in prediction.rallies
    )
    revised = build_review_features(replace(prediction, rallies=changed))
    np.testing.assert_array_equal(features.matrix, revised.matrix)
    assert features.matrix.shape[1] == len(REVIEW_FEATURE_NAMES)


def test_real_layer_fit_and_review_use_one_prediction_per_training_video() -> None:
    videos = [training_video(f'{group}-{correct}', group, correct) for group in 'ABCD' for correct in (False, True)]
    pytest.importorskip('xgboost')
    config = RallyFitConfig(tree=TreeFitConfig(max_iter=3, min_child_weight=0))
    with threadpool_limits(limits=1):
        result = fit_rally_models(videos, config)
        assert set(result.held_group_predictions) == {video.video_id for video in videos}
        assert result.selector.selector.n_features_in_ == len(FEATURE_NAMES)
        # The shared contact evidence cannot recover the deliberately absent label.
        ranker = fit_rally_review_model(
            videos, result.held_group_predictions,
            TreeFitConfig(max_iter=3, min_child_weight=0),
        )
        prediction = predict_rally_options(result.selector, runtime_options())
        review = score_rally_review(prediction, ranker)
    assert review.scores.shape == (1,)
    assert np.isfinite(review.scores).all()


def test_review_cross_fitting_requires_an_extra_group() -> None:
    videos = [training_video('A', 'A', True)]
    with pytest.raises(ValueError, match='at least two groups'):
        fit_rally_models(videos, RallyFitConfig())


@pytest.mark.parametrize(('fps', 'shift'), [(25.0, 4), (30.0, 5), (60.0, 10)])
def test_review_tolerance_changes_correctness_labels(fps, shift, monkeypatch):
    video = training_video('video', 'A', True)
    video = replace(
        video, context=replace(video.context, fps=fps),
        rallies=(LabelledRally('rally', (40 + shift,), (CourtHalf.TOP,)),),
    )
    options = build_rally_options(video.context)
    prediction = predict_rally_options(dummy_layers(), options)
    monkeypatch.setattr('annotator.training.rally.fit_review_model', lambda _matrix, labels, _config: labels)
    strict = fit_rally_review_model([video], {'video': prediction}, TreeFitConfig(), tolerance_at_30_fps=3)
    relaxed = fit_rally_review_model([video], {'video': prediction}, TreeFitConfig(), tolerance_at_30_fps=6)
    np.testing.assert_array_equal(strict, [0])
    np.testing.assert_array_equal(relaxed, [1])
