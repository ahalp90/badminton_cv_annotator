"""Compare selector confidence with the learned review model."""
from annotator.rallies.refine_rallies import (
    REVIEW_FEATURE_NAMES,
    RallyReview,
    RallyReviewFeatures,
)


def score_selector_review(features: RallyReviewFeatures) -> RallyReview:
    """Read selector scores already measured for the final rallies."""
    scores = features.matrix[:, REVIEW_FEATURE_NAMES.index('option_score')]
    return RallyReview(features, scores)
