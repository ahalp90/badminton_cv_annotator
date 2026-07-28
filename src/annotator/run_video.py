"""GT-free annotation-chain composition for one video."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

import annotator.point_winner as point_winner
import annotator.rally_segmentation as stage8_seg
from annotator.config import BaseAnnotatorConfig
from annotator.dead_mask import build_dead_mask
from annotator.replay_mask import believe_raw_mask
from annotator.resolve import resolve
from annotator.types import ContactCandidate, ServeStartConfig


OTHER_HALF = point_winner.OTHER_HALF


def scoring_filter(contacts):
    """Rows the scorer reads: wrist gate not failed, not suppressed."""
    return [c for c in contacts
            if c.wrist_near is not False and c.suppressed is not True]


def _build_event_non_evidence_mask(
    n_frames: int, rejected_grades: frozenset[int],
    inpaint_codes: np.ndarray | None, supplied_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Adapt source grades into the one boolean mask read by event rules."""
    if inpaint_codes is not None and supplied_mask is not None:
        raise ValueError('inpaint_codes and event_non_evidence_mask are mutually exclusive')
    if inpaint_codes is not None:
        if inpaint_codes.ndim != 1 or len(inpaint_codes) != n_frames:
            raise ValueError('inpaint_codes must be a frame-aligned one-dimensional array')
        return np.isin(inpaint_codes, tuple(rejected_grades)), inpaint_codes
    if supplied_mask is not None:
        if supplied_mask.ndim != 1 or len(supplied_mask) != n_frames or supplied_mask.dtype != np.bool_:
            raise ValueError('event_non_evidence_mask must be a frame-aligned boolean array')
        return supplied_mask, None
    return np.zeros(n_frames, dtype=bool), None


def _record_rejection(
    rows: list[dict[str, object]] | None, rule: str, rally_id: int,
    start_frame: int, end_frame: int, event_mask: np.ndarray,
    codes: np.ndarray | None, candidate_frames: list[int] | None = None,
) -> None:
    """Record one event interval when it contains an event-mask frame."""
    if rows is None:
        return
    if candidate_frames is None:
        masked_frames = np.flatnonzero(event_mask[start_frame:end_frame]) + start_frame
    else:
        masked_frames = np.array(
            [frame for frame in candidate_frames if event_mask[frame]], dtype=int,
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


def _record_trusted_mask_contact_rejection(
    rows: list[dict[str, object]] | None, rally_id: int,
    span: tuple[int, int], contact_frames: list[int],
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


def build_serve_options(
    config, sticky, constants, resolution, span_open=stage8_seg.SpanOpen.BACK_FILL,
) -> stage8_seg.ServeStartOptions:
    """Build sticky-sourced serve-start evidence from the unmasked cache.

    Serve evidence deliberately comes from the sticky cache built before any masking; the
    committed mask demonstrably eats live serves on sset21, and masking policy belongs to the
    decontamination commit and parked redesign, not this lane.
    """
    if config.close is not None and span_open is not None:
        raise ValueError('serve_start.close is unsupported with BACK_FILL')
    return stage8_seg.ServeStartOptions(
        # The sticky path supplies body-height-normalised serve-setup evidence.
        dist=None, threshold=config.threshold_bh, mode=config.mode, close=config.close,
        setup=stage8_seg.build_serve_setup_inputs(sticky, resolution),
        stillness_threshold_bh=config.stillness_threshold_bh,
        lookback_frames=constants.serve_start_lookback_frames,
        stillness_window_frames=constants.serve_stillness_window_frames,
    )


class AnnotatorResult(NamedTuple):
    """Everything the chain produces for one video, before any GT is read.

    :param spans: detected rally spans, `[(start, end), ...]`; rally_id is the list index.
    :param contacts: RAW `ContactCandidate` rows. `wrist_near` is the wrist gate verdict and
        `suppressed` records a gate-passing candidate that lost suppression.
    :param filtered_contacts: rows where `wrist_near is not False and suppressed is not True`.
        This keeps suppression winners and the unmeasured no-gate path — the set
        `scripts.stage8_score.score_contacts` scores the ball_round column against.
    :param filtered_by_rally: rally_id -> ascending contact frames from `filtered_contacts`.
    :param striker_halves: fitted final-contact half per rally_id (None: no contacts, or a tied
        fit); index-aligned to `spans`.
    :param n_strokes_list: `len(filtered_by_rally[rally_id])` per rally_id (0 when contact-less).
    :param next_servers: winner-serves-next half per rally_id (point_winner.next_server_half).
    :param fitted_first_all: each rally's OWN fitted first-stroke half (the server prediction),
        index-aligned to `spans`; None where `striker_halves[rally_id]` is None.
    :param verdict_rows: rally_id -> VerdictRow, only for rallies with a resolved striker.
    :param landings: rally_id -> Landing or None, same keys as `verdict_rows`.
    :param geometric_verdict_rows: rally_id -> geometric diagnostic, only for rallies with a
        resolved striker.
    :param hit_height_by_frame: contact_frame -> ShuttleSet-coded hit_height (1/2), one entry per
        filtered contact that scored successfully.
    :param hit_height_failures: `(rally_id, stroke_idx, contact_frame, error)` for filtered
        contacts where hit_height raised (shuttle not visible at that exact frame).
    """

    spans: list[tuple[int, int]]
    contacts: list[ContactCandidate]
    filtered_contacts: list[ContactCandidate]
    filtered_by_rally: dict[int, list[int]]
    striker_halves: list
    n_strokes_list: list[int]
    next_servers: list
    fitted_first_all: list
    verdict_rows: dict[int, object]
    landings: dict[int, object | None]
    geometric_verdict_rows: dict[int, object]
    hit_height_by_frame: dict[int, int]
    hit_height_failures: list[tuple[int, int, int, str]]


def _first_stroke_half(final_half, n_strokes: int):
    """The rally's own fitted first-stroke half, from its fitted final-contact half.

    Same parity formula as point_winner's private `_phase_assignment` at index 0 (last =
    n_strokes - 1; step back from the last stroke, flipping each step): duplicated here as a
    one-line arithmetic fact rather than reaching into that module-private helper, since
    `next_server_half` only ever exposes rally n+1's fitted first stroke (as rally n's winner),
    never a rally's own — Brief H's "server" column needs the latter for every rally.
    """
    return final_half if (n_strokes - 1) % 2 == 0 else OTHER_HALF[final_half]


def run_video(
    track, bboxes=None, scores=None, kps=None, ndet=None,
    *,
    fps: float,
    base: BaseAnnotatorConfig = BaseAnnotatorConfig(),
    landing_options=None,
    court_box=None,
    net_band: tuple[float, float] | None = None,
    resolution: tuple[float, float] | None = None,
    video_id: int | None = None,
    court_info: dict | None = None,
    homo_df=None,
    gate_court_info: dict | None = None,
    gate_resolution_table=None,
    ref_err_px: float = 3.5,
    dead_mask: np.ndarray | None = None,
    positions: np.ndarray | None = None,
    court_present=None,
    homography_rows=None,
    cut_frames=None,
    keep_vote=None,
    serve_start: ServeStartConfig | None = None,
    spans: list[tuple[int, int]] | None = None,
    contacts: dict[int, list[int]] | None = None,
    inpaint_codes: np.ndarray | None = None,
    event_non_evidence_mask: np.ndarray | None = None,
    rejection_diagnostics: list[dict[str, object]] | None = None,
    court_optional: bool = False,
    stop_after_segmentation: bool = False,
) -> AnnotatorResult:
    """Run segmentation, attribution, verdict, landing, and hit-height for one video.

    `court_optional` is only valid with `stop_after_segmentation`. That mode does not build
    sticky or enter any downstream court-dependent stage, so it accepts only the track, fps,
    positions, and optional dead mask.
    """
    resolved = resolve(base, fps)
    # Injected spans skip span finding, so serve-start would otherwise silently never run.
    if serve_start is not None and spans is not None:
        raise ValueError('serve_start cannot be combined with injected spans')
    if serve_start is not None and resolved.quiet_start_window is not None:
        raise ValueError('quiet_start_window cannot be combined with serve_start')
    if court_optional and not stop_after_segmentation:
        raise ValueError('court_optional requires stop_after_segmentation')
    if court_optional:
        supplied_optional_inputs = {
            'homography_rows': homography_rows,
            'court_present': court_present,
            'bboxes': bboxes,
            'scores': scores,
            'kps': kps,
            'ndet': ndet,
            'resolution': resolution,
            'video_id': video_id,
            'gate_court_info': gate_court_info,
            'gate_resolution_table': gate_resolution_table,
            'serve_start': serve_start,
        }
        supplied = [name for name, value in supplied_optional_inputs.items() if value is not None]
        if supplied:
            raise ValueError(f'court_optional rejects supplied inputs: {", ".join(supplied)}')
    else:
        if homography_rows is None or court_present is None:
            raise ValueError('scene-gated sticky needs homography_rows and court_present')
        required_sticky_inputs = {
            'bboxes': bboxes,
            'scores': scores,
            'kps': kps,
            'ndet': ndet,
            'resolution': resolution,
            'video_id': video_id,
            'gate_court_info': gate_court_info,
            'gate_resolution_table': gate_resolution_table,
        }
        missing_sticky_inputs = [name for name, value in required_sticky_inputs.items() if value is None]
        if missing_sticky_inputs:
            raise ValueError(f'normal mode requires {", ".join(missing_sticky_inputs)}')
        if not stop_after_segmentation:
            required_downstream_inputs = {
                'landing_options': landing_options,
                'court_box': court_box,
                'net_band': net_band,
                'court_info': court_info,
                'homo_df': homo_df,
            }
            missing_downstream_inputs = [
                name for name, value in required_downstream_inputs.items() if value is None
            ]
            if missing_downstream_inputs:
                raise ValueError(f'full-chain mode requires {", ".join(missing_downstream_inputs)}')

    event_mask, source_codes = _build_event_non_evidence_mask(
        len(track), resolved.rejected_grades, inpaint_codes, event_non_evidence_mask,
    )
    sticky = None
    if not court_optional:
        segments = stage8_seg.tracker_segments(homography_rows, court_present, len(track))
        sticky = stage8_seg.build_sticky_result(
            track, segments, bboxes, scores, kps, ndet, str(video_id), gate_court_info,
            gate_resolution_table, court_box, resolution, resolved.constants.body_unit_half_window,
        )
    serve_options = None
    if court_optional:
        mask = dead_mask if dead_mask is not None else np.zeros(len(track), dtype=bool)
        if len(mask) != len(track):
            raise ValueError(f'mask length {len(mask)} != track length {len(track)}')
        if mask.all():
            raise ValueError('mask is all True: no live frame to anchor a frozen position to')
        mask = believe_raw_mask(mask, resolved.constants.replay_mask_min_frames)
        if contacts is None:
            final_spans, raw_contacts = stage8_seg.segment_video(
                track, positions=positions, thresholds=resolved.thresholds,
                span_open=resolved.span_open, replay_mask=mask, sticky_distances=None,
                spans=spans, smoothing_mode=resolved.smoothing_mode,
                constants=resolved.constants, gap_state_demotion_bound=resolved.gap_state_demotion_bound,
                reentry_guard_variant=resolved.reentry_guard_variant,
                reentry_guard_buffer=resolved.reentry_guard_buffer,
                quiet_start_window=resolved.quiet_start_window,
            )
        else:
            final_spans = spans if spans is not None else stage8_seg.find_rally_spans(
                track, resolved.thresholds, span_open=resolved.span_open,
                constants=resolved.constants, gap_state_demotion_bound=resolved.gap_state_demotion_bound,
                reentry_guard_variant=resolved.reentry_guard_variant,
                reentry_guard_buffer=resolved.reentry_guard_buffer,
                quiet_start_window=resolved.quiet_start_window,
            )
            raw_contacts = [
                ContactCandidate(rally_id, frame, None, None, None)
                for rally_id, frames in contacts.items() for frame in frames
            ]
        return AnnotatorResult(
            spans=final_spans, contacts=raw_contacts, filtered_contacts=[], filtered_by_rally={},
            striker_halves=[], n_strokes_list=[], next_servers=[], fitted_first_all=[],
            verdict_rows={}, landings={}, geometric_verdict_rows={}, hit_height_by_frame={},
            hit_height_failures=[],
        )

    assert sticky is not None
    if contacts is not None:
        # Injected contacts already carry the selected rally IDs. Span finding still runs when
        # callers did not inject spans; sticky already ran over tracker segments above.
        final_spans = spans if spans is not None else stage8_seg.find_rally_spans(
            track, resolved.thresholds, span_open=resolved.span_open,
            constants=resolved.constants, gap_state_demotion_bound=resolved.gap_state_demotion_bound,
            reentry_guard_variant=resolved.reentry_guard_variant,
            reentry_guard_buffer=resolved.reentry_guard_buffer,
            quiet_start_window=resolved.quiet_start_window,
        )
        raw_contacts = [ContactCandidate(rally_id, frame, None, None, None)
                        for rally_id, frames in contacts.items() for frame in frames]
        # Replay mask baselines slow-motion detection against each rally's normal speed.
        mask = dead_mask
        if mask is None:
            mask = build_dead_mask(
                resolved.dead_mask_mode, len(track), fps, court_present=court_present,
                homography_rows=homography_rows, track=track, rally_spans=final_spans,
                cut_frames=cut_frames, keep_vote=keep_vote, non_evidence=event_mask,
            )
    else:
        # The sticky cache above already runs over scene-gated tracker segments. This span-only,
        # unmasked pass supplies rally spans to the mask builder.
        if dead_mask is None:
            bootstrap_spans = spans if spans is not None else stage8_seg.find_rally_spans(
                track, resolved.thresholds, span_open=resolved.span_open,
                constants=resolved.constants, gap_state_demotion_bound=resolved.gap_state_demotion_bound,
                reentry_guard_variant=resolved.reentry_guard_variant,
                reentry_guard_buffer=resolved.reentry_guard_buffer,
                quiet_start_window=resolved.quiet_start_window,
            )
            mask = build_dead_mask(
                resolved.dead_mask_mode, len(track), fps, court_present=court_present,
                homography_rows=homography_rows, track=track, rally_spans=bootstrap_spans,
                cut_frames=cut_frames, keep_vote=keep_vote, non_evidence=event_mask,
            )
        else:
            mask = dead_mask
        final_spans = spans
        if serve_start is not None:
            serve_options = build_serve_options(
                serve_start, sticky, resolved.constants, resolution, resolved.span_open,
            )

    if len(mask) != len(track):
        raise ValueError(f'mask length {len(mask)} != track length {len(track)}')
    if mask.all():
        raise ValueError('mask is all True: no live frame to anchor a frozen position to')
    mask = believe_raw_mask(mask, resolved.constants.replay_mask_min_frames)

    if contacts is None:
        final_spans, raw_contacts = stage8_seg.segment_video(
            track, positions=positions, thresholds=resolved.thresholds,
            span_open=resolved.span_open,
            replay_mask=mask, sticky_distances=sticky.distances, serve_start=serve_options,
            spans=final_spans, smoothing_mode=resolved.smoothing_mode,
            constants=resolved.constants, gap_state_demotion_bound=resolved.gap_state_demotion_bound,
            reentry_guard_variant=resolved.reentry_guard_variant,
            reentry_guard_buffer=resolved.reentry_guard_buffer,
            quiet_start_window=resolved.quiet_start_window,
        )
    spans, contacts = final_spans, raw_contacts

    if stop_after_segmentation:
        return AnnotatorResult(
            spans=spans, contacts=contacts, filtered_contacts=[], filtered_by_rally={},
            striker_halves=[], n_strokes_list=[], next_servers=[], fitted_first_all=[],
            verdict_rows={}, landings={}, geometric_verdict_rows={}, hit_height_by_frame={},
            hit_height_failures=[],
        )

    scored_contacts = scoring_filter(contacts)
    filtered_contacts = [
        contact for contact in scored_contacts if not mask[contact.contact_frame]
    ]
    scored_by_rally: dict[int, list[int]] = {}
    for contact in scored_contacts:
        scored_by_rally.setdefault(contact.rally_id, []).append(contact.contact_frame)
    filtered_by_rally: dict[int, list[int]] = {}
    for contact in filtered_contacts:
        filtered_by_rally.setdefault(contact.rally_id, []).append(contact.contact_frame)

    striker_halves = []
    for rally_id in range(len(spans)):
        frames = filtered_by_rally.get(rally_id, [])
        guesses = [
            point_winner.attribute_half(
                frame, track, sticky, bboxes, net_band,
            )
            for frame in frames
        ]
        striker_halves.append(point_winner.fit_alternation(guesses))
    n_strokes_list = [len(filtered_by_rally.get(rally_id, [])) for rally_id in range(len(spans))]
    next_servers = point_winner.next_server_half(striker_halves, n_strokes_list)
    fitted_first_all = [
        _first_stroke_half(half, n) if half is not None else None
        for half, n in zip(striker_halves, n_strokes_list)
    ]

    kin = point_winner.build_landing_kinematics(
        track, sticky, kps, resolution,
    )
    band_m = point_winner.corner_error_band_m(video_id, homo_df, court_info, ref_err_px)

    verdict_rows: dict[int, object] = {}
    landings: dict[int, object | None] = {}
    geometric_verdict_rows: dict[int, object] = {}
    for rally_id in range(len(spans)):
        striker = striker_halves[rally_id]
        if striker is None:
            scored_frames = scored_by_rally.get(rally_id, [])
            if scored_frames and not filtered_by_rally.get(rally_id):
                _record_trusted_mask_contact_rejection(
                    rejection_diagnostics, rally_id, spans[rally_id], scored_frames,
                )
            continue
        frames = filtered_by_rally[rally_id]
        usable_final_contacts = [frame for frame in frames if not event_mask[frame]]
        skipped_trailing: list[int] = []
        for frame in reversed(frames):
            if not event_mask[frame]:
                break
            skipped_trailing.append(frame)
        if skipped_trailing:
            _record_rejection(
                rejection_diagnostics, 'final_contact', rally_id, skipped_trailing[-1], frames[-1] + 1,
                event_mask, source_codes, candidate_frames=skipped_trailing[::-1],
            )
        if not usable_final_contacts:
            landing = None
            verdict = point_winner.rally_verdict(
                rally_id, striker, next_servers[rally_id], landing, band_m,
            )
            verdict_rows[rally_id] = verdict
            geometric, geometric_winner, _source = point_winner.geometric_verdict(striker, landing)
            geometric_verdict_rows[rally_id] = point_winner.GeometricVerdictRow(
                rally_id, geometric, geometric_winner, None, False,
            )
            landings[rally_id] = landing
            continue
        final_contact = usable_final_contacts[-1]
        next_start = spans[rally_id + 1][0] if rally_id + 1 < len(spans) else len(track)
        event_aware_window_end = point_winner.window_end(
            final_contact, next_start, track, mask, resolved.constants.sustained_loss_frames,
            event_mask,
        )
        if rejection_diagnostics is not None:
            window_end_without_events = point_winner.window_end(
                final_contact, next_start, track, mask, resolved.constants.sustained_loss_frames,
            )
            if event_aware_window_end < window_end_without_events:
                _record_rejection(
                    rejection_diagnostics, 'lost_shuttle_guard', rally_id, final_contact + 1,
                    window_end_without_events, event_mask, source_codes,
                )
        all_false_dead_mask = np.zeros_like(mask)
        window_end_without_dead_mask = point_winner.window_end(
            final_contact, next_start, track, all_false_dead_mask, resolved.constants.sustained_loss_frames,
            event_mask,
        )
        window_closed_by_mask = event_aware_window_end < window_end_without_dead_mask
        landing_rejections: list[tuple[int, int]] = []
        landing = point_winner.pick_landing(
            final_contact, next_start, track, mask, kin, landing_options, striker, net_band,
            resolution, court_info, resolved.constants, fps,
            event_non_evidence_mask=event_mask, rejected_intervals=landing_rejections,
        )
        for start_frame, end_frame in landing_rejections:
            _record_rejection(
                rejection_diagnostics, 'landing_descent', rally_id, start_frame, end_frame,
                event_mask, source_codes,
            )
        verdict = point_winner.rally_verdict(
            rally_id, striker, next_servers[rally_id], landing, band_m,
        )
        verdict_rows[rally_id] = verdict
        geometric, geometric_winner, _source = point_winner.geometric_verdict(striker, landing)
        shipped_winner = None
        if verdict.verdict == point_winner.Verdict.WON:
            shipped_winner = striker
        elif verdict.verdict == point_winner.Verdict.LOST:
            shipped_winner = OTHER_HALF[striker]
        # Agreement is a consistency check, not an accuracy meter: both arms share the same
        # fitted hitting order.
        agreement = None
        if shipped_winner is not None and geometric_winner is not None:
            agreement = shipped_winner == geometric_winner
        geometric_verdict_rows[rally_id] = point_winner.GeometricVerdictRow(
            rally_id, geometric, geometric_winner, agreement, window_closed_by_mask,
        )
        landings[rally_id] = landing

    hit_height_by_frame: dict[int, int] = {}
    hit_height_failures: list[tuple[int, int, int, str]] = []
    for rally_id in range(len(spans)):
        for stroke_idx, contact_frame in enumerate(filtered_by_rally.get(rally_id, [])):
            try:
                rows = point_winner.build_hit_height_rows(
                    [(rally_id, stroke_idx, contact_frame)], track, net_band, resolution,
                )
            except ValueError as exc:
                hit_height_failures.append((rally_id, stroke_idx, contact_frame, str(exc)))
                continue
            hit_height_by_frame[contact_frame] = rows[0].hit_height

    return AnnotatorResult(
        spans=spans, contacts=contacts, filtered_contacts=filtered_contacts,
        filtered_by_rally=filtered_by_rally,
        striker_halves=striker_halves, n_strokes_list=n_strokes_list, next_servers=next_servers,
        fitted_first_all=fitted_first_all, verdict_rows=verdict_rows, landings=landings,
        geometric_verdict_rows=geometric_verdict_rows,
        hit_height_by_frame=hit_height_by_frame, hit_height_failures=hit_height_failures,
    )
