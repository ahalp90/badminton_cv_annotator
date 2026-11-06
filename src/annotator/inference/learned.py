"""Run contact detection, rally refinement and review scoring for one video.

The contact model scores individual candidate frames. The rally selector scores
alternative contact lists. The review model ranks the finished rallies.
"""

from dataclasses import dataclass

from annotator.contacts.build_features import ContactFeatures, build_contact_features
from annotator.contacts.score_candidates import ScoredContacts, score_contact_features
from annotator.inference.prepare_contact_inference import AnnotationEvidence
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
    """Results of all three learned stages, with their measured inputs.

    Contact features/scores support inspection and training. ``refined`` keeps
    both the proposals and finished rallies; ``confidence`` ranks those rallies
    for review.
    """

    features: ContactFeatures
    contacts: ScoredContacts
    refined: RallyPrediction
    confidence: RallyReview


def predict_contacts(evidence: AnnotationEvidence, models: AnnotatorModels) -> LearnedModelPrediction:
    """Build features once, score contacts and refine their rally sequences."""
    report("building contact features")
    features = features_from_evidence(evidence)
    return predict_contact_features(evidence, features, models)


def predict_contact_features(
    evidence: AnnotationEvidence, features: ContactFeatures, models: AnnotatorModels,
) -> LearnedModelPrediction:
    """Score precomputed features, choose rally contacts, then score rally confidence."""
    report("scoring contact candidates")
    contacts = score_contact_features(
        features.rows, models.contact, evidence.fps, models.contact_settings,
    )
    report("refining rally sequences")
    context = build_rally_context(evidence, contacts)
    pool = build_rally_options(context, models.sequences.max_edits, models.sequences.deletion_candidates)
    refined = predict_rally_options(models.sequences, pool)
    report("scoring rally confidence")
    confidence = score_rally_review(refined, models.confidence)
    return LearnedModelPrediction(features, contacts, refined, confidence)


def features_from_evidence(evidence: AnnotationEvidence) -> ContactFeatures:
    """Measure learned contact features from prepared shuttle, player and rally evidence."""
    return build_contact_features(
        track=evidence.track,
        pose_kps=evidence.pose_kps,
        sticky=evidence.sticky,
        tracker_intervals=evidence.tracker_intervals,
        exclusion_mask=evidence.exclusion_mask,
        heuristic_spans=evidence.heuristic_spans,
        raw_contact_frames=evidence.raw_contact_frames,
        scene_spans=[(scene.start_frame, scene.end_frame) for scene in evidence.scene_courts],
        fps=evidence.fps,
        resolution=evidence.resolution,
    )
