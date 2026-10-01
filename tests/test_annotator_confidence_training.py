"""Check confidence fitting uses judged rows and the selected tree settings."""

from dataclasses import replace

import numpy as np
import pytest

from annotator.sequence.confidence import (
    BASE_FEATURE_NAMES,
    GAP_FEATURE_NAMES,
    ConfidenceFeatures,
)
from annotator.sequence.features import positive_probabilities
from annotator.training.confidence import ConfidenceFitConfig, fit_confidence_model


def training_example(row_count: int = 60) -> tuple[ConfidenceFeatures, np.ndarray]:
    random = np.random.default_rng(2)
    base = random.normal(size=(row_count, len(BASE_FEATURE_NAMES)))
    extra = random.normal(size=(row_count, len(GAP_FEATURE_NAMES) - len(BASE_FEATURE_NAMES)))
    labels = np.tile([0, 1, -1], row_count // 3)
    base[labels == 1, 0] += 4
    base[labels == 0, 0] -= 4
    features = ConfidenceFeatures(tuple(range(row_count)), base, np.column_stack((base, extra)))
    return features, labels


def test_fitted_ranking_distinguishes_correct_and_wrong_sections() -> None:
    features, labels = training_example()
    model = fit_confidence_model(features, labels)
    scores = positive_probabilities(model, features.gap)
    assert scores.shape == labels.shape
    assert np.mean(scores[labels == 1]) > np.mean(scores[labels == 0])


def test_unjudgeable_rows_are_excluded_before_fitting() -> None:
    features, labels = training_example()
    config = ConfidenceFitConfig(max_iter=4, min_samples_leaf=2)
    model = fit_confidence_model(features, labels, config)
    judged = labels != -1
    filtered = ConfidenceFeatures(tuple(np.flatnonzero(judged).tolist()), features.base[judged], features.gap[judged])
    expected = fit_confidence_model(filtered, labels[judged], config)
    np.testing.assert_array_equal(model.predict_proba(features.gap), expected.predict_proba(features.gap))


@pytest.mark.parametrize("labels", [np.full(60, -1), np.zeros(60), np.ones(60)])
def test_fit_requires_correct_and_wrong_judged_examples(labels: np.ndarray) -> None:
    features, _ = training_example()
    with pytest.raises(ValueError, match="both correct and wrong"):
        fit_confidence_model(features, labels)


def test_fit_rejects_misaligned_or_invalid_labels() -> None:
    features, labels = training_example()
    with pytest.raises(ValueError, match="align"):
        fit_confidence_model(features, labels[:-1])
    labels[0] = 2
    with pytest.raises(ValueError, match="-1, 0 or 1"):
        fit_confidence_model(features, labels)


def test_fit_rejects_wrong_feature_layout() -> None:
    features, labels = training_example()
    wrong = replace(features, gap=features.gap[:, :-1])
    with pytest.raises(ValueError, match="supported columns"):
        fit_confidence_model(wrong, labels)
