"""Score contact candidates, select rally contact lists, then rank rallies for review."""

from dataclasses import dataclass

from annotator.contacts.build_features import ContactFeatures, build_contact_features
from annotator.contacts.score_candidates import (
    ScoredContacts,
    positive_probabilities,
    score_contact_features,
)
from annotator.inference.prepare_annotation_inputs import PreparedAnnotationInputs
from annotator.models import AnnotatorModels
from annotator.rallies.build_contact_proposals import (
    RallyOptions,
    build_rally_context,
    build_rally_options,
)
from annotator.rallies.refine_rallies import (
    RallyPrediction,
    RallyReview,
    predict_rally_options,
)
from annotator.rallies.split_and_correct_rallies import split_and_correct_rallies
from annotator.types import ContactEvent, ContactSequence, FinishedRally
from shared.progress import report


@dataclass(frozen=True)
class LearnedModelPrediction:
    """Finished rallies and review scores, with inputs for inspection and training.

    ``refinement`` retains the first-pass choices and proposal pool. ``review``
    describes the final rallies after splitting and correction. ``source_span_ids``
    maps those rallies to the original spans for comparisons.
    """

    contact_features: ContactFeatures
    scored_candidates: ScoredContacts
    refinement: RallyPrediction
    review: RallyReview
    rallies: tuple[FinishedRally, ...]
    events: tuple[ContactEvent, ...]
    source_span_ids: tuple[int, ...]

    @property
    def sequences(self) -> tuple[ContactSequence, ...]:
        """Final contact sequences, in rally order."""
        return tuple(rally.sequence for rally in self.rallies)

    def scores_for_rallies(self, review: RallyReview) -> tuple[float, ...]:
        """Return scores aligned to the final rally IDs."""
        by_span = dict(zip(review.features.span_ids, map(float, review.scores), strict=True))
        return tuple(by_span[rally.sequence.span_id] for rally in self.rallies)

    @property
    def rally_confidence(self) -> tuple[float, ...]:
        """Review scores for the final rallies in output order."""
        return self.scores_for_rallies(self.review)


def predict_rallies_and_contacts(inputs: PreparedAnnotationInputs, models: AnnotatorModels) -> LearnedModelPrediction:
    """Build contact features, choose rally contacts, then score rallies for review."""
    report("building contact features")
    features = contact_features_from_inputs(inputs)
    return predict_from_contact_features(inputs, features, models)


def predict_from_contact_features(
    inputs: PreparedAnnotationInputs, features: ContactFeatures, models: AnnotatorModels,
) -> LearnedModelPrediction:
    """Reuse measured contact features when comparing fitted models."""
    report("scoring contact candidates")
    scored_candidates = score_contact_features(
        features.rows, models.contact_model, inputs.fps, models.contact_settings,
    )
    report("selecting rally contact lists")
    context = build_rally_context(inputs, scored_candidates)
    options = build_rally_options(context, models.rally_selector.max_edits, models.rally_selector.deletion_candidates)
    return predict_from_rally_options(inputs, features, scored_candidates, options, models)


def predict_from_rally_options(
    inputs: PreparedAnnotationInputs, features: ContactFeatures, scored_candidates: ScoredContacts,
    options: RallyOptions, models: AnnotatorModels,
) -> LearnedModelPrediction:
    """Select rally contacts, correct split pieces, then score the final rallies."""
    refinement = predict_rally_options(models.rally_selector, options)
    corrected = split_and_correct_rallies(
        inputs, scored_candidates, refinement, models.followup_selector,
    )
    report("scoring rallies for review")
    review = RallyReview(
        corrected.review_features, positive_probabilities(models.review_model, corrected.review_features.matrix),
    )
    return LearnedModelPrediction(
        features, scored_candidates, refinement, review, corrected.rallies, corrected.events, corrected.source_span_ids,
    )


def contact_features_from_inputs(inputs: PreparedAnnotationInputs) -> ContactFeatures:
    """Measure contact features from the prepared shuttle, players and initial rallies."""
    return build_contact_features(
        track=inputs.track,
        pose_kps=inputs.pose_kps,
        players=inputs.players,
        wrist_gaps=inputs.wrist_gaps,
        tracker_intervals=inputs.tracker_intervals,
        exclusion_mask=inputs.exclusion_mask,
        heuristic_spans=inputs.heuristic_spans,
        raw_contact_frames=inputs.raw_contact_frames,
        scene_spans=[(scene.start_frame, scene.end_frame) for scene in inputs.scene_courts],
        fps=inputs.fps,
        resolution=inputs.resolution,
    )
