"""Apply the contact tree and sequence refinements to one video's evidence."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import numpy as np

from annotator.contacts.features import ContactFeatures, build_contact_features
from annotator.contacts.model import ScoredContacts, score_contact_features
from annotator.courts.scenes import court_at_frame
from annotator.models import AnnotatorModels, SideGeometry
from annotator.outcomes.point_winner import Half, attribute_half
from annotator.preparation import AnnotationEvidence
from annotator.sequence import (
    CandidateScores,
    ContactEvent,
)
from annotator.sequence.rally_model import (
    RallyLayerPrediction,
    RallyReview,
    score_rally_review,
    select_rally_sequences,
)
from shared.progress import report


@dataclass(frozen=True)
class HybridPrediction:
    """Final sequences and the already-computed inputs needed for a retune."""

    features: ContactFeatures
    contacts: ScoredContacts
    refined: RallyLayerPrediction
    confidence: RallyReview


def features_from_evidence(evidence: AnnotationEvidence) -> ContactFeatures:
    """Build model features from the already-computed heuristic inputs."""
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


def sequence_inputs(
    evidence: AnnotationEvidence, contacts: ScoredContacts, side_geometry: SideGeometry,
) -> tuple[CandidateScores, tuple[ContactEvent, ...], Callable[[int], Half | None]]:
    """Share scored candidates and cached raw side guesses between fitting and inference."""
    @cache
    def side_for_frame(frame: int) -> Half | None:
        net_band = evidence.net_band
        if side_geometry is SideGeometry.SCENE:
            # Serve look-back frames can precede the accepted scene and its tracker.
            if evidence.track[frame, 2] != 1 or not np.isfinite(evidence.sticky.distances_per_slot[frame]).any():
                return None
            net_band = court_at_frame(evidence.scene_courts, frame).net_band
        return attribute_half(frame, evidence.track, evidence.sticky, evidence.bboxes, net_band)

    kept = np.zeros(len(contacts.candidates), dtype=bool)
    kept[contacts.kept_indices] = True
    scores = CandidateScores(
        contacts.candidates['frame'], contacts.candidates['interval_id'], contacts.probabilities, kept,
    )
    events = tuple(
        ContactEvent(int(scores.frames[row]), float(scores.probabilities[row]), side_for_frame(int(scores.frames[row])))
        for row in contacts.kept_indices
    )
    return scores, events, side_for_frame


def predict_contacts(evidence: AnnotationEvidence, models: AnnotatorModels) -> HybridPrediction:
    """Build features once, score contacts and refine their rally sequences."""
    report("building contact features")
    features = features_from_evidence(evidence)
    return predict_contact_features(evidence, features, models)


def predict_contact_features(
    evidence: AnnotationEvidence, features: ContactFeatures, models: AnnotatorModels,
) -> HybridPrediction:
    """Annotate precomputed contact features through the same production path."""
    report("scoring contact candidates")
    contacts = score_contact_features(
        features.rows, models.contact, evidence.fps, models.contact_settings,
        shuttle_hallucination_mask=evidence.shuttle_hallucination_mask,
    )
    scores, events, side_for_frame = sequence_inputs(evidence, contacts, models.side_geometry)
    report("refining rally sequences")
    refined = select_rally_sequences(
        evidence.heuristic_spans, events, scores, features.rows, features.search_intervals,
        fps=evidence.fps, side_for_frame=side_for_frame, models=models.sequences, frame_count=len(evidence.track),
        bounds_supplied=evidence.bounds_supplied,
    )
    report("scoring rally confidence")
    confidence = score_rally_review(refined, models.confidence)
    return HybridPrediction(features, contacts, refined, confidence)
