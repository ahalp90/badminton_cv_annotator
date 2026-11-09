"""Score contact candidates, select rally contact lists, then rank rallies for review."""

from dataclasses import dataclass

from annotator.contacts.build_features import ContactFeatures, build_contact_features
from annotator.contacts.score_candidates import ScoredContacts, score_contact_features
from annotator.inference.prepare_annotation_inputs import PreparedAnnotationInputs
from annotator.models import AnnotatorModels
from annotator.rallies.build_contact_proposals import (
    build_rally_context,
    build_rally_options,
)
from annotator.rallies.refine_rallies import (
    RallyPrediction,
    RallyReview,
    predict_rally_options,
    score_rally_review,
)
from shared.progress import report


@dataclass(frozen=True)
class LearnedModelPrediction:
    """Finished rallies and review scores, with inputs for inspection and training.

    ``rally_prediction`` keeps the proposals, chosen rallies and contact events.
    ``rally_review`` scores those chosen rallies in the same order.
    """

    contact_features: ContactFeatures
    scored_candidates: ScoredContacts
    rally_prediction: RallyPrediction
    rally_review: RallyReview


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
    rally_prediction = predict_rally_options(models.rally_selector, options)
    report("scoring rallies for review")
    rally_review = score_rally_review(rally_prediction, models.review_model)
    return LearnedModelPrediction(features, scored_candidates, rally_prediction, rally_review)


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
