"""Resolve rally landings and winners, then calculate contact heights."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from annotator.courts.scenes import SceneCourt, court_at_frame
from annotator.outcomes import point_winner
from annotator.preparation import AnnotationEvidence
from annotator.sequence.contacts import FinishedRally
from shared.progress import progress_iter


def build_outcomes(
    evidence: AnnotationEvidence,
    rallies: Sequence[FinishedRally],
    *,
    landing_options: point_winner.LandingFilterOptions,
    rejection_diagnostics: list[dict[str, object]] | None,
) -> dict[int, point_winner.RallyOutcome]:
    """Resolve each final flight using landing options already scaled for this video."""
    video_kinematics = point_winner.build_landing_kinematics(
        evidence.track, evidence.sticky, evidence.pose_kps, evidence.resolution,
    )
    outcomes = {}
    for rally_id, rally in enumerate(progress_iter(rallies, "resolving outcomes", unit="rallies")):
        striker = rally.final_half
        if striker is None:
            continue
        next_rally = rallies[rally_id + 1] if rally_id + 1 < len(rallies) else None
        next_server = next_rally.first_half if next_rally is not None else None
        next_start = next_rally.sequence.start_frame if next_rally is not None else len(evidence.track)
        final_contact = rally.sequence.events[-1].frame
        flight = None
        if evidence.shuttle_hallucination_mask[final_contact]:
            # An earlier hit may belong to the other player; it cannot replace this hit.
            if rejection_diagnostics is not None:
                record_rejected_final_contact(rejection_diagnostics, rally_id, rally, evidence)
        else:
            flight = final_flight_range(final_contact, next_start, rally.sequence.end_frame, evidence)

        landing = None
        error_band = None
        shortened_by_exclusion = False
        kinematics = video_kinematics
        if flight is not None:
            scene, search_start, search_limit = flight
            if search_start > final_contact:
                # A camera cut is not shuttle motion in the returning view.
                speed = video_kinematics.speed.copy()
                speed[search_start] = np.nan
                kinematics = video_kinematics._replace(speed=speed)
            window = point_winner.landing_window(
                search_start, search_limit, evidence.track, evidence.exclusion_mask,
                evidence.resolved.thresholds.sustained_loss_frames, evidence.shuttle_hallucination_mask,
            )
            rejected_intervals = []
            landing = point_winner.pick_landing_to_end(
                search_start, window.end_frame, evidence.track, kinematics, landing_options,
                striker, scene.net_band, evidence.resolution, scene.court_info,
                evidence.resolved.thresholds,
                shuttle_hallucination_mask=evidence.shuttle_hallucination_mask,
                rejected_intervals=rejected_intervals, infer_net_fault=search_start == final_contact,
            )
            error_band = scene.landing_error_band_m
            shortened_by_exclusion = window.search_shortened_by_exclusion
            if rejection_diagnostics is not None:
                record_landing_rejections(
                    rejection_diagnostics, rally_id, evidence, search_start, search_limit,
                    window.end_frame, rejected_intervals,
                )

        outcomes[rally_id] = point_winner.resolve_rally_outcome(
            striker, next_server, landing, error_band, search_shortened_by_exclusion=shortened_by_exclusion,
        )
    return outcomes


def final_flight_range(
    final_contact: int, next_start: int, rally_end: int, evidence: AnnotationEvidence,
) -> tuple[SceneCourt, int, int] | None:
    """Choose one camera view and its half-open search range for the final flight."""
    scene = next((scene for scene in evidence.scene_courts
                  if scene.end_frame > final_contact and scene.start_frame < rally_end), None)
    if scene is None:
        return None
    if scene.start_frame <= final_contact:
        return scene, final_contact, min(next_start, scene.end_frame)
    # Recover a flight from a returning view only within this rally's live time.
    if evidence.exclusion_mask[scene.start_frame]:
        return None
    return scene, scene.start_frame, min(next_start, scene.end_frame, rally_end)


def build_hit_heights(
    evidence: AnnotationEvidence, rallies: Sequence[FinishedRally],
) -> tuple[dict[int, int], dict[int, str]]:
    """Measure hit heights and record why unavailable heights could not be measured."""
    hit_height_by_frame: dict[int, int] = {}
    unavailable_hit_height_reasons: dict[int, str] = {}
    for rally in rallies:
        for event in rally.sequence.events:
            contact_frame = event.frame
            try:
                frame_net_band = court_at_frame(evidence.scene_courts, contact_frame).net_band
                hit_height = point_winner.hit_height(
                    evidence.track, contact_frame, frame_net_band, evidence.resolution,
                )
            except ValueError as exc:
                unavailable_hit_height_reasons[contact_frame] = str(exc)
                continue
            hit_height_by_frame[contact_frame] = hit_height
    return hit_height_by_frame, unavailable_hit_height_reasons


def record_rejected_final_contact(
    rows: list[dict[str, object]], rally_id: int,
    rally: FinishedRally, evidence: AnnotationEvidence,
) -> None:
    """Record the rejected contact tail when its last hit is masked."""
    final_contact = rally.sequence.events[-1].frame
    start_frame = final_contact
    for event in reversed(rally.sequence.events):
        if not evidence.shuttle_hallucination_mask[event.frame]:
            break
        start_frame = event.frame
    record_rejection(
        rows, 'final_contact', rally_id, start_frame, final_contact + 1,
        evidence.shuttle_hallucination_mask, evidence.source_codes,
    )


def record_landing_rejections(
    rows: list[dict[str, object]], rally_id: int, evidence: AnnotationEvidence,
    search_start: int, search_limit: int, search_end: int,
    rejected_intervals: list[tuple[int, int]],
) -> None:
    """Record how rejected shuttle observations changed this landing search."""
    end_without_events = point_winner.landing_window(
        search_start, search_limit, evidence.track, evidence.exclusion_mask,
        evidence.resolved.thresholds.sustained_loss_frames,
        np.zeros(len(evidence.track), dtype=bool),
    ).end_frame
    if search_end < end_without_events:
        record_rejection(
            rows, 'lost_shuttle_guard', rally_id, search_start + 1, end_without_events,
            evidence.shuttle_hallucination_mask, evidence.source_codes,
        )
    for start_frame, end_frame in rejected_intervals:
        record_rejection(
            rows, 'landing_descent', rally_id, start_frame, end_frame,
            evidence.shuttle_hallucination_mask, evidence.source_codes,
        )


def record_rejection(
    rows: list[dict[str, object]],
    rule: str,
    rally_id: int,
    start_frame: int,
    end_frame: int,
    shuttle_hallucination_mask: np.ndarray,
    codes: np.ndarray | None,
) -> None:
    """Record one event interval when it contains an event-mask frame."""
    masked_frames = np.flatnonzero(shuttle_hallucination_mask[start_frame:end_frame]) + start_frame
    if len(masked_frames) == 0:
        return
    first_rejected_frame = int(masked_frames[0])
    rows.append({
        'rule': rule,
        'rally_id': rally_id,
        'start_frame': start_frame,
        'end_frame': end_frame,
        'first_rejected_frame': first_rejected_frame,
        'rejection_grade': int(codes[first_rejected_frame]) if codes is not None else '',
    })
