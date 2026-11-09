"""Compare selector confidence with the learned review model."""
from annotator.rallies.refine_rallies import (
    RallyPrediction,
    RallyReview,
    build_review_features,
)


def score_selector_review(prediction: RallyPrediction) -> RallyReview:
    """Expose selector scores for experimental ranking comparisons."""
    features = build_review_features(prediction)
    scores = prediction.scores[list(prediction.chosen.values())]
    return RallyReview(features, scores)
