"""The fitted layer bundle works through training, storage and inference."""

import json
from dataclasses import replace

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

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
from annotator.training.sequences import (
    LabelledRally,
    SequenceTrainingVideo,
)
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
    models = AnnotatorModels(dummy_layers().selector, dummy_layers(), dummy_layers(), review)
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
