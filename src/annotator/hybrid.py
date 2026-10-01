"""Apply the contact tree and sequence refinements to one video's evidence."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache

import numpy as np

from annotator.contacts.features import ContactFeatures, build_contact_features
from annotator.contacts.model import ScoredContacts, score_contact_features
from annotator.courts.scenes import SceneCourt, court_at_frame
from annotator.models import AnnotatorModels, SideGeometry
from annotator.outcomes.point_winner import Half, attribute_half
from annotator.sequence import (
    CandidateScores,
    ContactEvent,
    RefinedSequences,
    refine_contact_sequences,
)
from annotator.sequence.confidence import RallyConfidence, score_rally_confidence
from annotator.types import StickyResult


@dataclass(frozen=True)
class ContactEvidence:
    """Video arrays and heuristic results shared by the learned contact stages."""

    identity: str
    fps: float
    resolution: tuple[float, float]
    track: np.ndarray
    pose_kps: np.ndarray
    bboxes: np.ndarray
    sticky: StickyResult
    tracker_intervals: Sequence[tuple[int, int]]
    exclusion_mask: np.ndarray
    heuristic_spans: Sequence[tuple[int, int]]
    raw_contact_frames: Sequence[int]
    scene_courts: Sequence[SceneCourt]
    net_band: tuple[float, float]


@dataclass(frozen=True)
class HybridPrediction:
    """Final sequences and the already-computed inputs needed for a retune."""

    features: ContactFeatures
    contacts: ScoredContacts
    refined: RefinedSequences
    confidence: RallyConfidence


def features_from_evidence(evidence: ContactEvidence) -> ContactFeatures:
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
    evidence: ContactEvidence, contacts: ScoredContacts, side_geometry: SideGeometry,
) -> tuple[CandidateScores, tuple[ContactEvent, ...], Callable[[int], Half | None]]:
    """Share scored candidates and cached raw side guesses between fitting and inference."""
    @cache
    def side_for_frame(frame: int) -> Half | None:
        net_band = evidence.net_band
        if side_geometry is SideGeometry.SCENE:
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


def predict_contacts(evidence: ContactEvidence, models: AnnotatorModels) -> HybridPrediction:
    """Build features once, score contacts and refine their rally sequences."""
    features = features_from_evidence(evidence)
    contacts = score_contact_features(features.rows, models.contact, evidence.fps, models.contact_settings)
    scores, events, side_for_frame = sequence_inputs(evidence, contacts, models.side_geometry)
    refined = refine_contact_sequences(
        evidence.heuristic_spans, events, scores, features.rows, features.search_intervals,
        fps=evidence.fps, side_for_frame=side_for_frame, models=models.sequences, frame_count=len(evidence.track),
    )
    confidence = score_rally_confidence(refined, models.confidence, models.sequences.insertion, evidence.fps)
    return HybridPrediction(features, contacts, refined, confidence)
