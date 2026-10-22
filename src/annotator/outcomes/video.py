"""Landing, verdict, horizon and hit-height calculations after contact selection."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from annotator.config import ResolvedAnnotatorConfig
from annotator.contacts.selection import ContactData
from annotator.courts.scenes import SceneCourt, court_at_frame
from annotator.outcomes import point_winner
from annotator.preparation import AnnotationEvidence
from shared.progress import progress_iter

OTHER_HALF = point_winner.OTHER_HALF


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


@dataclass(frozen=True)
class VerdictData:
    """Rally-indexed landing and winner outputs."""

    verdict_rows: dict[int, point_winner.VerdictRow]
    landings: dict[int, point_winner.Landing | None]
    geometric_verdict_rows: dict[int, point_winner.GeometricVerdictRow]


@dataclass(frozen=True)
class LandingContext:
    """Evidence shared by landing and horizon evaluation for every rally."""

    track: np.ndarray
    fps: float
    kinematics: point_winner.LandingKinematics
    landing_options: point_winner.LandingFilterOptions
    net_band: tuple[float, float]
    resolution: tuple[float, float]
    court_info: dict
    resolved: ResolvedAnnotatorConfig
    definitive_exclusion_mask: np.ndarray
    shuttle_hallucination_mask: np.ndarray
    source_codes: np.ndarray | None
    rejection_diagnostics: list[dict[str, object]] | None
    band_m: float
    landing_horizons_s: tuple[float, ...]
    horizon_rows: list[LandingHorizonRow] | None
    scene_courts: tuple[SceneCourt, ...]


@dataclass(frozen=True)
class LandingSelection:
    """Landing evidence derived from one rally's final usable contact."""

    final_contact: int
    search_start: int
    safe_window: point_winner.LandingWindow
    landing: point_winner.Landing | None
    window_closed_by_mask: bool


@dataclass(frozen=True)
class RallyOutcome:
    """Verdict outputs for one rally."""

    verdict: point_winner.VerdictRow
    landing: point_winner.Landing | None
    geometric_verdict: point_winner.GeometricVerdictRow


def record_rejection(
    rows: list[dict[str, object]] | None,
    rule: str,
    rally_id: int,
    start_frame: int,
    end_frame: int,
    shuttle_hallucination_mask: np.ndarray,
    codes: np.ndarray | None,
    candidate_frames: list[int] | None = None,
) -> None:
    """Record one event interval when it contains an event-mask frame."""
    if rows is None:
        return
    if candidate_frames is None:
        masked_frames = (
            np.flatnonzero(shuttle_hallucination_mask[start_frame:end_frame]) + start_frame
        )
    else:
        masked_frames = np.array(
            [frame for frame in candidate_frames if shuttle_hallucination_mask[frame]], dtype=int,
        )
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


def record_trusted_mask_contact_rejection(
    rows: list[dict[str, object]] | None,
    rally_id: int,
    span: tuple[int, int],
    contact_frames: list[int],
) -> None:
    """Record a rally whose scoring contacts all fell on trusted-dead frames."""
    if rows is None:
        return
    rows.append({
        'rule': 'all_contacts_on_believed_mask',
        'rally_id': rally_id,
        'start_frame': span[0],
        'end_frame': span[1],
        'trigger_frame': contact_frames[0],
        'trigger_code': '',
    })


def winner_from_verdict(
    verdict: point_winner.VerdictRow, striker: point_winner.Half,
) -> point_winner.Half | None:
    """Return the winner half encoded by a verdict row."""
    if verdict.verdict == point_winner.Verdict.WON:
        return striker
    if verdict.verdict == point_winner.Verdict.LOST:
        return OTHER_HALF[striker]
    return None


def final_usable_contact(
    rally_id: int,
    frames: list[int],
    context: LandingContext,
) -> int | None:
    """Return the last event-usable contact and record rejected trailing contacts."""
    usable_final_contacts = [
        frame for frame in frames if not context.shuttle_hallucination_mask[frame]
    ]
    skipped_trailing: list[int] = []
    for frame in reversed(frames):
        if not context.shuttle_hallucination_mask[frame]:
            break
        skipped_trailing.append(frame)
    if skipped_trailing:
        record_rejection(
            context.rejection_diagnostics,
            'final_contact',
            rally_id,
            skipped_trailing[-1],
            frames[-1] + 1,
            context.shuttle_hallucination_mask,
            context.source_codes,
            candidate_frames=skipped_trailing[::-1],
        )
    return usable_final_contacts[-1] if usable_final_contacts else None


def select_landing(
    rally_id: int,
    final_contact: int,
    striker: point_winner.Half,
    next_start: int,
    context: LandingContext,
    *, search_start: int,
) -> LandingSelection:
    """Select one landing and record event-mask rejections around its window."""
    safe_window = point_winner.landing_window(
        search_start,
        next_start,
        context.track,
        context.definitive_exclusion_mask,
        context.resolved.thresholds.sustained_loss_frames,
        context.shuttle_hallucination_mask,
    )
    if context.rejection_diagnostics is not None:
        window_end_without_events = point_winner.window_end(
            search_start,
            next_start,
            context.track,
            context.definitive_exclusion_mask,
            context.resolved.thresholds.sustained_loss_frames,
        )
        if safe_window.end_frame < window_end_without_events:
            record_rejection(
                context.rejection_diagnostics,
                'lost_shuttle_guard',
                rally_id,
                search_start + 1,
                window_end_without_events,
                context.shuttle_hallucination_mask,
                context.source_codes,
            )
    all_false_exclusion_mask = np.zeros_like(context.definitive_exclusion_mask)
    window_end_without_exclusion_mask = point_winner.window_end(
        search_start,
        next_start,
        context.track,
        all_false_exclusion_mask,
        context.resolved.thresholds.sustained_loss_frames,
        context.shuttle_hallucination_mask,
    )
    landing_rejections: list[tuple[int, int]] = []
    landing = point_winner.pick_landing_to_end(
        search_start,
        safe_window.end_frame,
        context.track,
        context.kinematics,
        context.landing_options,
        striker,
        context.net_band,
        context.resolution,
        context.court_info,
        context.resolved.thresholds,
        context.fps,
        shuttle_hallucination_mask=context.shuttle_hallucination_mask,
        rejected_intervals=landing_rejections,
        infer_net_fault=search_start == final_contact,
    )
    for start_frame, end_frame in landing_rejections:
        record_rejection(
            context.rejection_diagnostics,
            'landing_descent',
            rally_id,
            start_frame,
            end_frame,
            context.shuttle_hallucination_mask,
            context.source_codes,
        )
    return LandingSelection(
        final_contact=final_contact,
        search_start=search_start,
        safe_window=safe_window,
        landing=landing,
        window_closed_by_mask=(
            safe_window.end_frame < window_end_without_exclusion_mask
        ),
    )


def build_horizon_rows(
    rally_id: int,
    striker: point_winner.Half,
    next_server: point_winner.Half | None,
    selection: LandingSelection,
    strict_verdict: point_winner.VerdictRow,
    strict_winner: point_winner.Half | None,
    context: LandingContext,
) -> None:
    """Append capped-landing comparisons in requested horizon order."""
    if context.landing_horizons_s:
        assert context.horizon_rows is not None
    for horizon_seconds in context.landing_horizons_s:
        horizon_frames = max(1, math.floor(horizon_seconds * context.fps + 0.5))
        requested_end_frame = selection.final_contact + horizon_frames
        effective_end_frame = max(
            selection.final_contact + 1,
            min(requested_end_frame, selection.safe_window.end_frame),
        )
        closure_reasons: list[str] = []
        if requested_end_frame == effective_end_frame:
            closure_reasons.append('horizon_cap')
        if effective_end_frame == selection.safe_window.end_frame:
            closure_reasons.extend(selection.safe_window.closure_reasons)
        capped_landing = None
        if effective_end_frame > selection.search_start:
            capped_landing = point_winner.pick_landing_to_end(
                selection.search_start,
                effective_end_frame,
                context.track,
                context.kinematics,
                context.landing_options,
                striker,
                context.net_band,
                context.resolution,
                context.court_info,
                context.resolved.thresholds,
                context.fps,
                shuttle_hallucination_mask=context.shuttle_hallucination_mask,
                infer_net_fault=selection.search_start == selection.final_contact,
            )
        capped_verdict = point_winner.rally_verdict(
            rally_id, striker, next_server, capped_landing, context.band_m,
        )
        capped_winner = winner_from_verdict(capped_verdict, striker)
        context.horizon_rows.append(LandingHorizonRow(
            rally_id=rally_id,
            horizon_seconds=horizon_seconds,
            horizon_frames=horizon_frames,
            final_contact_frame=selection.final_contact,
            requested_end_frame=requested_end_frame,
            safe_end_frame=selection.safe_window.end_frame,
            effective_end_frame=effective_end_frame,
            closure_reasons=tuple(closure_reasons),
            strict_landing=selection.landing,
            capped_landing=capped_landing,
            strict_verdict=strict_verdict,
            capped_verdict=capped_verdict,
            strict_winner=strict_winner,
            capped_winner=capped_winner,
            landing_changed=selection.landing != capped_landing,
            winner_changed=strict_winner != capped_winner,
        ))


def build_rally_outcome(
    rally_id: int,
    striker: point_winner.Half,
    next_server: point_winner.Half | None,
    next_start: int,
    frames: list[int],
    context: LandingContext,
    *, rally_end: int,
) -> RallyOutcome:
    """Build strict and horizon outputs for one rally with a fitted striker."""
    final_contact = final_usable_contact(rally_id, frames, context)
    # A rejected last hit leaves the final flight unknown. An earlier hit may
    # belong to the other player, so it cannot stand in for the final one.
    if final_contact != frames[-1]:
        final_contact = None
    scene = None
    if final_contact is not None:
        scene = next((scene for scene in context.scene_courts
                      if scene.end_frame > final_contact and scene.start_frame < rally_end), None)
        if scene is None or (
            scene.start_frame > final_contact and context.definitive_exclusion_mask[scene.start_frame]
        ):
            final_contact = None
    if final_contact is None:
        landing = None
        verdict = point_winner.rally_verdict(
            rally_id, striker, next_server, landing, context.band_m,
        )
        geometric, geometric_winner, _source = point_winner.geometric_verdict(striker, landing)
        geometric_verdict = point_winner.GeometricVerdictRow(
            rally_id, geometric, geometric_winner, None, False,
        )
        return RallyOutcome(verdict, landing, geometric_verdict)

    search_start = final_contact
    assert scene is not None
    context = replace(
        context, court_info=scene.court_info, net_band=scene.net_band,
        band_m=scene.landing_error_band_m,
    )
    # A landing search cannot carry one camera's geometry across a scene cut.
    next_start = min(next_start, scene.end_frame)
    if scene.start_frame > final_contact:
        search_start = scene.start_frame
        next_start = min(next_start, rally_end)
        # The camera cut is not shuttle motion in the returning view.
        speed = context.kinematics.speed.copy()
        speed[search_start] = np.nan
        context = replace(context, kinematics=context.kinematics._replace(speed=speed))
    selection = select_landing(
        rally_id, final_contact, striker, next_start, context, search_start=search_start,
    )
    verdict = point_winner.rally_verdict(
        rally_id, striker, next_server, selection.landing, context.band_m,
    )
    geometric, geometric_winner, _source = point_winner.geometric_verdict(
        striker, selection.landing,
    )
    strict_winner = winner_from_verdict(verdict, striker)
    agreement = None
    if strict_winner is not None and geometric_winner is not None:
        agreement = strict_winner == geometric_winner
    geometric_verdict = point_winner.GeometricVerdictRow(
        rally_id,
        geometric,
        geometric_winner,
        agreement,
        selection.window_closed_by_mask,
    )
    build_horizon_rows(
        rally_id, striker, next_server, selection, verdict, strict_winner, context,
    )
    return RallyOutcome(verdict, selection.landing, geometric_verdict)


def build_verdict_data(
    evidence: AnnotationEvidence,
    contact_data: ContactData,
    *,
    court_info: dict,
    landing_options: point_winner.LandingFilterOptions,
    band_m: float,
    rejection_diagnostics: list[dict[str, object]] | None,
    landing_horizons_s: tuple[float, ...],
    horizon_rows: list[LandingHorizonRow] | None,
) -> VerdictData:
    """Build landing, verdict, diagnostic and horizon outputs for every rally."""
    kinematics = point_winner.build_landing_kinematics(
        evidence.track, evidence.sticky, evidence.pose_kps, evidence.resolution,
    )
    context = LandingContext(
        track=evidence.track,
        fps=evidence.fps,
        kinematics=kinematics,
        landing_options=landing_options,
        net_band=evidence.net_band,
        resolution=evidence.resolution,
        court_info=court_info,
        resolved=evidence.resolved,
        definitive_exclusion_mask=evidence.exclusion_mask,
        shuttle_hallucination_mask=evidence.shuttle_hallucination_mask,
        source_codes=evidence.source_codes,
        rejection_diagnostics=rejection_diagnostics,
        band_m=band_m,
        landing_horizons_s=landing_horizons_s,
        horizon_rows=horizon_rows,
        scene_courts=evidence.scene_courts,
    )
    verdict_rows: dict[int, point_winner.VerdictRow] = {}
    landings: dict[int, point_winner.Landing | None] = {}
    geometric_verdict_rows: dict[int, point_winner.GeometricVerdictRow] = {}
    for rally_id, span in enumerate(progress_iter(contact_data.spans, "resolving outcomes", unit="rallies")):
        striker = contact_data.striker_halves[rally_id]
        if striker is None:
            scored_frames = contact_data.scored_by_rally.get(rally_id, [])
            if scored_frames and not contact_data.filtered_by_rally.get(rally_id):
                record_trusted_mask_contact_rejection(
                    rejection_diagnostics, rally_id, span, scored_frames,
                )
            continue
        next_start = (
            contact_data.spans[rally_id + 1][0]
            if rally_id + 1 < len(contact_data.spans) else len(evidence.track)
        )
        outcome = build_rally_outcome(
            rally_id,
            striker,
            contact_data.next_servers[rally_id],
            next_start,
            contact_data.filtered_by_rally[rally_id],
            context,
            rally_end=span[1],
        )
        verdict_rows[rally_id] = outcome.verdict
        landings[rally_id] = outcome.landing
        geometric_verdict_rows[rally_id] = outcome.geometric_verdict
    return VerdictData(verdict_rows, landings, geometric_verdict_rows)


def build_hit_heights(
    evidence: AnnotationEvidence, contact_data: ContactData,
) -> tuple[dict[int, int], list[tuple[int, int, int, str]]]:
    """Build hit-height outputs without coupling them to landing verdicts."""
    hit_height_by_frame: dict[int, int] = {}
    hit_height_failures: list[tuple[int, int, int, str]] = []
    for rally_id in range(len(contact_data.spans)):
        for stroke_idx, contact_frame in enumerate(contact_data.filtered_by_rally.get(rally_id, [])):
            try:
                frame_net_band = court_at_frame(evidence.scene_courts, contact_frame).net_band
                rows = point_winner.build_hit_height_rows(
                    [(rally_id, stroke_idx, contact_frame)], evidence.track, frame_net_band, evidence.resolution,
                )
            except ValueError as exc:
                hit_height_failures.append((rally_id, stroke_idx, contact_frame, str(exc)))
                continue
            hit_height_by_frame[contact_frame] = rows[0].hit_height
    return hit_height_by_frame, hit_height_failures
