"""Resolve rally landings and winners, then calculate contact heights."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from annotator.courts.scenes import SceneCourt, court_at_frame
from annotator.outcomes import point_winner
from annotator.preparation import AnnotationEvidence
from annotator.sequence.contacts import FinishedRally
from shared.progress import progress_iter


class RallyOutcome(NamedTuple):
    """One rally's winner estimate, landing and geometric comparison."""

    verdict: point_winner.VerdictRow
    landing: point_winner.Landing | None
    geometric: point_winner.GeometricVerdictRow


def build_verdict_data(
    evidence: AnnotationEvidence,
    rallies: Sequence[FinishedRally],
    *,
    landing_options: point_winner.LandingFilterOptions,
    rejection_diagnostics: list[dict[str, object]] | None,
    landing_horizons_s: tuple[float, ...],
    horizon_rows: list[LandingHorizonRow] | None,
) -> dict[int, RallyOutcome]:
    """Find each final landing, then resolve the winner from the next serve or landing."""
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
        closed_by_mask = False
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
                evidence.resolved.thresholds, evidence.fps,
                shuttle_hallucination_mask=evidence.shuttle_hallucination_mask,
                rejected_intervals=rejected_intervals, infer_net_fault=search_start == final_contact,
            )
            error_band = scene.landing_error_band_m
            closed_by_mask = window.window_closed_by_mask
            if rejection_diagnostics is not None:
                record_landing_rejections(
                    rejection_diagnostics, rally_id, evidence, search_start, search_limit,
                    window.end_frame, rejected_intervals,
                )

        verdict = point_winner.rally_verdict(rally_id, striker, next_server, landing, error_band)
        geometric, geometric_winner, _source = point_winner.geometric_verdict(striker, landing)
        winner = winner_from_verdict(verdict, striker)
        agreement = None
        if winner is not None and geometric_winner is not None:
            agreement = winner == geometric_winner
        outcomes[rally_id] = RallyOutcome(verdict, landing, point_winner.GeometricVerdictRow(
            rally_id, geometric, geometric_winner, agreement, closed_by_mask,
        ))
        if landing_horizons_s and flight is not None:
            assert horizon_rows is not None
            horizon_rows.extend(build_horizon_rows(
                evidence, kinematics, landing_options, scene,
                final_contact=final_contact, search_start=search_start, window=window,
                strict_verdict=verdict, strict_landing=landing, strict_winner=winner,
                horizons_s=landing_horizons_s, next_server=next_server,
            ))
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
    """Build hit-height outputs without coupling them to landing verdicts."""
    hit_height_by_frame: dict[int, int] = {}
    hit_height_failures: dict[int, str] = {}
    for rally in rallies:
        for event in rally.sequence.events:
            contact_frame = event.frame
            try:
                frame_net_band = court_at_frame(evidence.scene_courts, contact_frame).net_band
                hit_height = point_winner.hit_height(
                    evidence.track, contact_frame, frame_net_band, evidence.resolution,
                )
            except ValueError as exc:
                hit_height_failures[contact_frame] = str(exc)
                continue
            hit_height_by_frame[contact_frame] = hit_height
    return hit_height_by_frame, hit_height_failures


def winner_from_verdict(
    verdict: point_winner.VerdictRow, striker: point_winner.Half,
) -> point_winner.Half | None:
    """Return the winner half encoded by a verdict row."""
    if verdict.verdict == point_winner.Verdict.WON:
        return striker
    if verdict.verdict == point_winner.Verdict.LOST:
        return point_winner.OTHER_HALF[striker]
    return None


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
    search_start: int, search_limit: int, safe_end: int,
    rejected_intervals: list[tuple[int, int]],
) -> None:
    """Record how rejected shuttle observations changed this landing search."""
    end_without_events = point_winner.window_end(
        search_start, search_limit, evidence.track, evidence.exclusion_mask,
        evidence.resolved.thresholds.sustained_loss_frames,
    )
    if safe_end < end_without_events:
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
    trigger_frame = int(masked_frames[0])
    rows.append({
        'rule': rule,
        'rally_id': rally_id,
        'start_frame': start_frame,
        'end_frame': end_frame,
        'trigger_frame': trigger_frame,
        'trigger_code': int(codes[trigger_frame]) if codes is not None else '',
    })


@dataclass(frozen=True)
class LandingHorizonRow:
    """One GT-free comparison between the safe and a shorter landing endpoint."""

    rally_id: int
    horizon_seconds: float
    horizon_frames: int
    final_contact_frame: int
    requested_end_frame: int
    safe_end_frame: int
    effective_end_frame: int
    closure_reasons: tuple[str, ...]
    strict_landing: point_winner.Landing | None
    capped_landing: point_winner.Landing | None
    strict_verdict: point_winner.VerdictRow
    capped_verdict: point_winner.VerdictRow
    strict_winner: point_winner.Half | None
    capped_winner: point_winner.Half | None
    landing_changed: bool
    winner_changed: bool


def build_horizon_rows(
    evidence: AnnotationEvidence,
    kinematics: point_winner.LandingKinematics,
    landing_options: point_winner.LandingFilterOptions,
    scene: SceneCourt,
    *,
    final_contact: int,
    search_start: int,
    window: point_winner.LandingWindow,
    strict_verdict: point_winner.VerdictRow,
    strict_landing: point_winner.Landing | None,
    strict_winner: point_winner.Half | None,
    horizons_s: tuple[float, ...],
    next_server: point_winner.Half | None,
) -> list[LandingHorizonRow]:
    """Compare shorter landing searches with the ordinary outcome, in horizon order."""
    rally_id = strict_verdict.rally_id
    striker = strict_verdict.striker_half
    rows = []
    for horizon_seconds in horizons_s:
        horizon_frames = max(1, math.floor(horizon_seconds * evidence.fps + 0.5))
        requested_end_frame = final_contact + horizon_frames
        effective_end_frame = max(final_contact + 1, min(requested_end_frame, window.end_frame))
        closure_reasons = []
        if requested_end_frame == effective_end_frame:
            closure_reasons.append('horizon_cap')
        if effective_end_frame == window.end_frame:
            closure_reasons.extend(window.closure_reasons)
            capped_landing, capped_verdict, capped_winner = strict_landing, strict_verdict, strict_winner
        else:
            capped_landing = None
            if effective_end_frame > search_start:
                capped_landing = point_winner.pick_landing_to_end(
                    search_start, effective_end_frame, evidence.track, kinematics, landing_options,
                    striker, scene.net_band, evidence.resolution, scene.court_info,
                    evidence.resolved.thresholds, evidence.fps,
                    shuttle_hallucination_mask=evidence.shuttle_hallucination_mask,
                    infer_net_fault=search_start == final_contact,
                )
            capped_verdict = point_winner.rally_verdict(
                rally_id, striker, next_server, capped_landing, scene.landing_error_band_m,
            )
            capped_winner = winner_from_verdict(capped_verdict, striker)
        rows.append(LandingHorizonRow(
            rally_id=rally_id,
            horizon_seconds=horizon_seconds,
            horizon_frames=horizon_frames,
            final_contact_frame=final_contact,
            requested_end_frame=requested_end_frame,
            safe_end_frame=window.end_frame,
            effective_end_frame=effective_end_frame,
            closure_reasons=tuple(closure_reasons),
            strict_landing=strict_landing,
            capped_landing=capped_landing,
            strict_verdict=strict_verdict,
            capped_verdict=capped_verdict,
            strict_winner=strict_winner,
            capped_winner=capped_winner,
            landing_changed=strict_landing != capped_landing,
            winner_changed=strict_winner != capped_winner,
        ))
    return rows
