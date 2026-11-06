"""Build the inputs shared by heuristic and learned contact inference.

Prepare scene geometry, track players, measure wrist gaps, exclude unusable
frames and find initial rallies and heuristic contact candidates. Training
uses the same prepared evidence before building learned-model features.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from annotator import player_tracking
from annotator.config import ResolvedAnnotatorConfig
from annotator.contacts.player_attribution import measure_wrist_distances
from annotator.courts.frame_geometry import SceneCourt, build_scene_courts
from annotator.excluded_frames.replay import (
    combine_mask,
    filter_exclusion_mask,
)
from annotator.rallies.find_rallies import find_rally_spans, segment_video
from annotator.types import ContactCandidate, StickyResult


@dataclass(frozen=True)
class AnnotationEvidence:
    """Original vision arrays, tracked players, masks and initial rally/contact detections."""

    identity: str
    resolution: tuple[float, float]
    track: np.ndarray
    pose_kps: np.ndarray
    bboxes: np.ndarray
    sticky: StickyResult
    tracker_intervals: Sequence[tuple[int, int]]
    scene_courts: tuple[SceneCourt, ...]
    resolved: ResolvedAnnotatorConfig
    raw_exclusion_mask: np.ndarray
    exclusion_mask: np.ndarray
    heuristic_spans: list[tuple[int, int]]
    heuristic_contacts: list[ContactCandidate]
    shuttle_hallucination_mask: np.ndarray
    source_codes: np.ndarray | None
    bounds_supplied: bool = False

    @property
    def fps(self) -> float:
        """Frame rate used to resolve preparation settings."""
        return self.resolved.fps

    @property
    def raw_contact_frames(self) -> list[int]:
        """Every candidate frame, preserving injected and detected contact order."""
        return [contact.contact_frame for contact in self.heuristic_contacts]


def prepare_contact_inference(
    track: np.ndarray,
    bboxes: np.ndarray,
    scores: np.ndarray,
    kps: np.ndarray,
    ndet: np.ndarray,
    *,
    resolved: ResolvedAnnotatorConfig,
    resolution: tuple[float, float],
    video_id: int | str,
    court_present: np.ndarray,
    homography_rows: pd.DataFrame | list[dict[str, object]],
    ref_err_px: float = 3.5,
    raw_exclusion_mask: np.ndarray | None = None,
    spans: list[tuple[int, int]] | None = None,
    contacts: dict[int, list[int]] | None = None,
    inpaint_codes: np.ndarray | None = None,
    court_invalid_is_excluded: bool = False,
) -> AnnotationEvidence:
    """Prepare player/court evidence and initial rallies for contact inference.

    The result feeds either inference route and supplies feature inputs during
    training. Supplied contacts bypass candidate detection and retain their order.
    Supplied spans bypass span finding. Neither changes the original shuttle track.
    """
    if isinstance(homography_rows, pd.DataFrame):
        homography_rows = homography_rows.to_dict('records')
    shuttle_hallucination_mask, source_codes = build_shuttle_hallucination_mask(
        len(track), resolved.rejected_grades, inpaint_codes,
    )
    scene_courts = build_scene_courts(homography_rows, resolution, ref_err_px)
    # Player tracking needs the original track before dead-time masking.
    segments = player_tracking.tracker_segments(scene_courts, court_present, len(track))
    sticky = build_player_evidence(
        track, segments, bboxes, scores, kps, ndet,
        resolution, resolved.thresholds.body_unit_half_window,
        scene_courts=scene_courts,
    )
    original_spans = spans
    if raw_exclusion_mask is None:
        # Replay-mask estimation needs bounds from the original track.
        if original_spans is None:
            original_spans = find_rally_spans(track, resolved)
        raw_exclusion_mask = combine_mask(
            court_present, track, original_spans, len(track), resolved.fps,
            non_evidence=shuttle_hallucination_mask,
        )
    exclusion_mask = filter_exclusion_mask(raw_exclusion_mask, len(track), resolved.thresholds.replay_mask_min_frames)
    usable_court_frames = np.zeros(len(track), dtype=bool)
    for scene in scene_courts:
        start = max(0, scene.start_frame)
        end = max(0, min(len(track), scene.end_frame))
        usable_court_frames[start:end] = True
    if court_invalid_is_excluded:
        usable_court_frames = usable_court_frames & court_present
    exclusion_mask = exclusion_mask | ~usable_court_frames
    if exclusion_mask.all():
        raise ValueError('mask is all True: no live frame to anchor a frozen position to')
    if contacts is not None:
        # Supplied contacts belong to the original track, before dead-time masking.
        if original_spans is None:
            original_spans = find_rally_spans(track, resolved)
        rally_spans = original_spans
        raw_contacts = [ContactCandidate(rally_id, frame, None, None)
                        for rally_id, frames in contacts.items() for frame in frames]
    else:
        rally_spans, raw_contacts = segment_video(
            track, resolved=resolved, exclusion_mask=exclusion_mask,
            sticky_distances=sticky.distances, spans=spans,
        )
    return AnnotationEvidence(
        identity=str(video_id), resolution=resolution, track=track, pose_kps=kps, bboxes=bboxes,
        sticky=sticky, tracker_intervals=segments, scene_courts=scene_courts,
        resolved=resolved, raw_exclusion_mask=raw_exclusion_mask, exclusion_mask=exclusion_mask,
        heuristic_spans=rally_spans, heuristic_contacts=raw_contacts,
        shuttle_hallucination_mask=shuttle_hallucination_mask, source_codes=source_codes,
        bounds_supplied=spans is not None,
    )


def build_shuttle_hallucination_mask(
    n_frames: int, rejected_grades: frozenset[int], inpaint_codes: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Resolve one frame-aligned event mask from the supplied shuttle grades."""
    if inpaint_codes is None:
        return np.zeros(n_frames, dtype=bool), None
    if inpaint_codes.ndim != 1 or len(inpaint_codes) != n_frames:
        raise ValueError('inpaint_codes must be a frame-aligned one-dimensional array')
    return np.isin(inpaint_codes, tuple(rejected_grades)), inpaint_codes


def build_player_evidence(
    track: np.ndarray, segments: list[tuple[int, int]],
    pose_bboxes: np.ndarray, pose_scores: np.ndarray, pose_kps: np.ndarray,
    pose_ndet: np.ndarray,
    resolution: tuple[float, float], half_window: int,
    *, scene_courts: tuple[SceneCourt, ...],
) -> StickyResult:
    """Track the top/bottom players, then measure their shuttle-to-wrist gaps."""
    evidence = player_tracking.track_sticky_players(
        len(track),
        segments,
        pose_bboxes,
        pose_scores,
        pose_kps,
        pose_ndet,
        resolution,
        scene_courts,
    )
    measure_wrist_distances(track, segments, pose_kps, resolution, half_window, evidence)
    return evidence
