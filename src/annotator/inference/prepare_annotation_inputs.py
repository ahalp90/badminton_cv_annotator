"""Prepare courts, players and initial contacts for either inference route.

Track players and measure wrist distances before masking footage. Then find
initial rallies and heuristic contacts. Training reuses these prepared inputs.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from annotator import player_tracking
from annotator.config import FpsScaledAnnotatorConfig
from annotator.contacts.player_attribution import measure_wrist_distances
from annotator.courts.frame_geometry import SceneCourt, build_scene_courts
from annotator.excluded_frames.replay import (
    combine_mask,
    filter_exclusion_mask,
)
from annotator.rallies.find_rallies import find_rally_spans, segment_video
from annotator.types import ContactCandidate, TrackedPlayers, WristGaps


@dataclass(frozen=True)
class PreparedAnnotationInputs:
    """Prepared video inputs shared by contact inference and outcome estimation.

    Arrays use original video frame indices. ``track`` is unchanged; only
    heuristic segmentation uses a copy with excluded footage frozen in place.
    ``tracker_intervals`` are accepted court-view ranges, incl start, excl end.

    ``raw_exclusion_mask`` flags footage before short-run filtering.
    ``exclusion_mask`` keeps sustained exclusions and adds frames without usable
    court geometry or court presence.
    ``rejected_shuttle_frames`` marks rejected shuttle grades, used when
    measuring speed evidence and interpreting final flights and landings.

    ``heuristic_contacts`` includes failed and unchecked candidates, not just
    accepted hits. ``bounds_supplied`` prevents learned refinement from moving
    supplied rally bounds.
    """

    resolution: tuple[float, float]
    track: np.ndarray
    pose_kps: np.ndarray
    bboxes: np.ndarray
    players: TrackedPlayers
    wrist_gaps: WristGaps
    tracker_intervals: Sequence[tuple[int, int]]
    scene_courts: tuple[SceneCourt, ...]
    video_config: FpsScaledAnnotatorConfig
    raw_exclusion_mask: np.ndarray
    exclusion_mask: np.ndarray
    heuristic_spans: list[tuple[int, int]]
    heuristic_contacts: list[ContactCandidate]
    rejected_shuttle_frames: np.ndarray
    bounds_supplied: bool = False

    @property
    def fps(self) -> float:
        """Frame rate used to scale the preparation settings."""
        return self.video_config.fps

    @property
    def raw_contact_frames(self) -> list[int]:
        """All initial candidate frames, in supplied or detection order."""
        return [contact.contact_frame for contact in self.heuristic_contacts]


def prepare_annotation_inputs(
    track: np.ndarray,
    bboxes: np.ndarray,
    scores: np.ndarray,
    kps: np.ndarray,
    ndet: np.ndarray,
    *,
    video_config: FpsScaledAnnotatorConfig,
    resolution: tuple[float, float],
    court_present: np.ndarray,
    homography_rows: pd.DataFrame | list[dict[str, object]],
    ref_err_px: float = 3.5,
    raw_exclusion_mask: np.ndarray | None = None,
    spans: list[tuple[int, int]] | None = None,
    contacts: dict[int, list[int]] | None = None,
    shuttle_grades: np.ndarray | None = None,
) -> PreparedAnnotationInputs:
    """Prepare courts, players, masks and initial rallies for inference or training.

    Supplied spans skip rally detection. Supplied contacts skip contact
    detection and keep their order. The original shuttle track is unchanged.
    """
    if isinstance(homography_rows, pd.DataFrame):
        homography_rows = homography_rows.to_dict('records')
    rejected_shuttle_frames = build_rejected_shuttle_mask(
        len(track), video_config.rejected_grades, shuttle_grades,
    )
    scene_courts = build_scene_courts(homography_rows, resolution, ref_err_px)
    # Player tracking needs the original track before dead-time masking.
    segments = player_tracking.tracker_segments(scene_courts, court_present, len(track))
    players, wrist_gaps = build_player_evidence(
        track, segments, bboxes, scores, kps, ndet,
        resolution, video_config.thresholds.body_unit_half_window,
        scene_courts=scene_courts,
    )
    original_spans = spans
    if raw_exclusion_mask is None:
        # Replay-mask estimation needs bounds from the original track.
        if original_spans is None:
            original_spans = find_rally_spans(track, video_config)
        raw_exclusion_mask = combine_mask(
            court_present, track, original_spans, len(track), video_config.fps,
            rejected_shuttle_frames=rejected_shuttle_frames,
        )
    exclusion_mask = filter_exclusion_mask(raw_exclusion_mask, len(track), video_config.thresholds.replay_mask_min_frames)
    usable_court_frames = np.zeros(len(track), dtype=bool)
    for scene in scene_courts:
        start = max(0, scene.start_frame)
        end = max(0, min(len(track), scene.end_frame))
        usable_court_frames[start:end] = True
    exclusion_mask = exclusion_mask | ~usable_court_frames | ~court_present
    if exclusion_mask.all():
        raise ValueError('mask is all True: no live frame to anchor a frozen position to')
    if contacts is not None:
        # Supplied contacts belong to the original track, before dead-time masking.
        if original_spans is None:
            original_spans = find_rally_spans(track, video_config)
        rally_spans = original_spans
        raw_contacts = [ContactCandidate(rally_id, frame, None, None)
                        for rally_id, frames in contacts.items() for frame in frames]
    else:
        rally_spans, raw_contacts = segment_video(
            track, video_config=video_config, exclusion_mask=exclusion_mask,
            nearest_wrist_distances=wrist_gaps.nearest, spans=spans,
        )
    return PreparedAnnotationInputs(
        resolution=resolution, track=track, pose_kps=kps, bboxes=bboxes,
        players=players, wrist_gaps=wrist_gaps, tracker_intervals=segments, scene_courts=scene_courts,
        video_config=video_config, raw_exclusion_mask=raw_exclusion_mask, exclusion_mask=exclusion_mask,
        heuristic_spans=rally_spans, heuristic_contacts=raw_contacts,
        rejected_shuttle_frames=rejected_shuttle_frames,
        bounds_supplied=spans is not None,
    )


def build_rejected_shuttle_mask(
    n_frames: int, rejected_grades: frozenset[int], shuttle_grades: np.ndarray | None,
) -> np.ndarray:
    """Mark frames whose shuttle grade is rejected."""
    if shuttle_grades is None:
        return np.zeros(n_frames, dtype=bool)
    if shuttle_grades.ndim != 1 or len(shuttle_grades) != n_frames:
        raise ValueError('shuttle_grades must be a frame-aligned one-dimensional array')
    return np.isin(shuttle_grades, tuple(rejected_grades))


def build_player_evidence(
    track: np.ndarray, segments: list[tuple[int, int]],
    pose_bboxes: np.ndarray, pose_scores: np.ndarray, pose_kps: np.ndarray,
    pose_ndet: np.ndarray,
    resolution: tuple[float, float], half_window: int,
    *, scene_courts: tuple[SceneCourt, ...],
) -> tuple[TrackedPlayers, WristGaps]:
    """Track the top/bottom players, then measure their shuttle-to-wrist gaps."""
    players = player_tracking.track_sticky_players(
        len(track),
        segments,
        pose_bboxes,
        pose_scores,
        pose_kps,
        pose_ndet,
        resolution,
        scene_courts,
    )
    wrist_gaps = measure_wrist_distances(track, segments, pose_kps, resolution, half_window, players)
    return players, wrist_gaps
