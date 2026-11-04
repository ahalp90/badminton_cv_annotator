"""Apply the contact tree and sequence refinements to one video's evidence."""

from dataclasses import dataclass
from functools import cache

import numpy as np

from annotator.contacts.features import ContactFeatures, build_contact_features
from annotator.contacts.model import ScoredContacts, score_contact_features
from annotator.courts.scenes import court_at_frame
from annotator.models import AnnotatorModels, SideGeometry
from annotator.outcomes.point_winner import Half, attribute_half
from annotator.preparation import AnnotationEvidence
from annotator.sequence.candidates import build_rally_candidates
from annotator.sequence.contacts import ContactEvent, initial_sequences
from annotator.sequence.rally_model import (
    RallyPrediction,
    RallyReview,
    predict_rally_options,
    score_rally_review,
)
from annotator.sequence.rally_options import RallyContext, build_rally_options
from shared.progress import report


@dataclass(frozen=True)
class HybridPrediction:
    """Final sequences and the already-computed inputs needed for a retune."""

    features: ContactFeatures
    contacts: ScoredContacts
    refined: RallyPrediction
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


def build_rally_context(
    evidence: AnnotationEvidence, contacts: ScoredContacts, search_intervals: list[tuple[int, int]],
    side_geometry: SideGeometry,
) -> RallyContext:
    """Prepare the same initial rallies and candidate contacts for fitting and inference."""
    @cache
    def side_for_frame(frame: int) -> Half | None:
        net_band = evidence.net_band
        if side_geometry is SideGeometry.SCENE:
            # Serve look-back frames can precede the accepted scene and its tracker.
            if evidence.track[frame, 2] != 1 or not np.isfinite(evidence.sticky.distances_per_slot[frame]).any():
                return None
            net_band = court_at_frame(evidence.scene_courts, frame).net_band
        return attribute_half(frame, evidence.track, evidence.sticky, evidence.bboxes, net_band)

    events = tuple(
        ContactEvent(int(frame), float(contacts.probabilities[row]), side_for_frame(int(frame)))
        for row, frame in zip(contacts.kept_indices, contacts.frames, strict=True)
    )
    initial = initial_sequences(evidence.heuristic_spans, events)
    candidates = build_rally_candidates(
        initial, events, contacts, search_intervals, fps=evidence.fps, side_for_frame=side_for_frame,
    )
    return RallyContext(evidence.fps, initial, events, candidates, len(evidence.track), evidence.bounds_supplied)


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
    )
    report("refining rally sequences")
    context = build_rally_context(evidence, contacts, features.search_intervals, models.side_geometry)
    pool = build_rally_options(context, models.sequences.max_edits, models.sequences.deletion_candidates)
    refined = predict_rally_options(models.sequences, pool)
    report("scoring rally confidence")
    confidence = score_rally_review(refined, models.confidence)
    return HybridPrediction(features, contacts, refined, confidence)
