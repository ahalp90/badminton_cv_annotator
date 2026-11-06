"""Apply landing, winner and hit-height estimates to every rally in a video."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from annotator.courts.frame_geometry import SceneCourt, court_at_frame
from annotator.inference.prepare_contact_inference import AnnotationEvidence
from annotator.rally_results import landing_and_winner
from annotator.types import FinishedRally
from shared.progress import progress_iter


def build_outcomes(
    evidence: AnnotationEvidence,
    rallies: Sequence[FinishedRally],
    *,
    landing_options: landing_and_winner.LandingFilterOptions,
    rejection_diagnostics: list[dict[str, object]] | None,
) -> dict[int, landing_and_winner.RallyOutcome]:
    """Estimate landings and winners for rallies whose final hitter is known.

    Measure video-wide motion once, then search each final flight. Reject a
    final contact on a flagged shuttle frame; the next server can still give
    a winner. Landing options must already use this video's frame rate.

    Pass a list as ``rejection_diagnostics`` to collect rejection reasons for
    reports. Leave it as None to skip that bookkeeping; outcomes are the same.
    """
    video_kinematics = landing_and_winner.build_landing_kinematics(
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

        if flight is None:
            outcomes[rally_id] = landing_and_winner.resolve_rally_outcome(striker, next_server, None, None)
            continue

        scene, search_start, search_limit = flight
        starts_in_returning_view = search_start > final_contact
        kinematics = video_kinematics
        if starts_in_returning_view:
            # A camera cut is not shuttle motion in the returning view.
            speed = video_kinematics.speed.copy()
            speed[search_start] = np.nan
            kinematics = video_kinematics._replace(speed=speed)
        window = landing_and_winner.landing_window(
            search_start, search_limit, evidence.track, evidence.exclusion_mask,
            evidence.video_config.thresholds.sustained_loss_frames, evidence.shuttle_hallucination_mask,
        )
        rejected_intervals = []
        landing = landing_and_winner.pick_landing_to_end(
            search_start, window.end_frame, evidence.track, kinematics, landing_options,
            striker, scene.net_band, evidence.resolution, scene.court_info,
            evidence.video_config.thresholds,
            shuttle_hallucination_mask=evidence.shuttle_hallucination_mask,
            rejected_intervals=rejected_intervals, infer_net_fault=not starts_in_returning_view,
        )
        if rejection_diagnostics is not None:
            record_landing_rejections(
                rejection_diagnostics, rally_id, evidence, search_start, search_limit,
                window.end_frame, rejected_intervals,
            )
        outcomes[rally_id] = landing_and_winner.resolve_rally_outcome(
            striker, next_server, landing, scene.landing_error_band_m,
            search_shortened_by_exclusion=window.search_shortened_by_exclusion,
        )
    return outcomes


def final_flight_range(
    final_contact: int, next_start: int, rally_end: int, evidence: AnnotationEvidence,
) -> tuple[SceneCourt, int, int] | None:
    """Choose a camera view and landing search range, incl start, excl end.

    Prefer the view containing the final contact. If the view returns later,
    use it only within the rally's live time and when its first frame is usable.
    """
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
    """Measure each contact height; record a reason when its position/court is unavailable."""
    hit_height_by_frame: dict[int, int] = {}
    unavailable_hit_height_reasons: dict[int, str] = {}
    for rally in rallies:
        for event in rally.sequence.events:
            contact_frame = event.frame
            try:
                frame_net_band = court_at_frame(evidence.scene_courts, contact_frame).net_band
                hit_height = landing_and_winner.hit_height(
                    evidence.track, contact_frame, frame_net_band, evidence.resolution,
                )
            except ValueError as exc:
                unavailable_hit_height_reasons[contact_frame] = str(exc)
                continue
            hit_height_by_frame[contact_frame] = hit_height
    return hit_height_by_frame, unavailable_hit_height_reasons


# =============================================================================
# Optional rejection diagnostics
# Used by evaluation reports; enabled by passing rejection_diagnostics.
# These helpers append report rows without changing annotation decisions.
# =============================================================================

def record_rejected_final_contact(
    rows: list[dict[str, object]], rally_id: int,
    rally: FinishedRally, evidence: AnnotationEvidence,
) -> None:
    """Log the run of rejected hits at the end of the rally as one frame range."""
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
    """Log rejected descents and any search shortened by rejected shuttle detections.

    Recalculate the search end with no shuttle-grade exclusions, then compare
    with search_end. This comparison is only for reporting.
    """
    end_without_events = landing_and_winner.landing_window(
        search_start, search_limit, evidence.track, evidence.exclusion_mask,
        evidence.video_config.thresholds.sustained_loss_frames,
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
    """Append a diagnostic row for a range containing rejected shuttle frames.

    Include the first rejected frame and its grade, if the grades are available.
    """
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
