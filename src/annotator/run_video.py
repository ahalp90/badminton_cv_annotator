"""GT-free annotation-chain composition for one video."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

import annotator.point_winner as point_winner
import annotator.rally_segmentation as stage8_seg
from annotator.config import BaseAnnotatorConfig
from annotator.dead_mask import build_dead_mask
from annotator.resolve import resolve
from annotator.types import ContactCandidate, ServeStartConfig


OTHER_HALF = point_winner.OTHER_HALF


def scoring_filter(contacts):
    """Rows the scorer reads: wrist gate not failed, not suppressed."""
    return [c for c in contacts
            if c.wrist_near is not False and c.suppressed is not True]


def build_serve_options(
    config, sticky, constants, resolution, span_open=stage8_seg.SpanOpen.BACK_FILL,
) -> stage8_seg.ServeStartOptions:
    """Build sticky-sourced serve-start evidence from the unmasked cache."""
    if config.close is not None and span_open is not None:
        raise ValueError('serve_start.close is unsupported with BACK_FILL')
    return stage8_seg.ServeStartOptions(
        dist=None, threshold=config.threshold, mode=config.mode, close=config.close,
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
    track, bboxes, scores, kps, ndet,
    *,
    fps: float,
    base: BaseAnnotatorConfig = BaseAnnotatorConfig(),
    landing_options,
    court_box,
    net_band: tuple[float, float],
    resolution: tuple[float, float],
    video_id: int,
    court_info: dict,
    homo_df,
    gate_court_info: dict,
    gate_resolution_table,
    ref_err_px: float = 3.5,
    dead_mask: np.ndarray | None = None,
    court_present=None,
    homography_rows=None,
    cut_frames=None,
    keep_vote=None,
    serve_start: ServeStartConfig | None = None,
    spans: list[tuple[int, int]] | None = None,
    contacts: dict[int, list[int]] | None = None,
) -> AnnotatorResult:
    """Run segmentation, attribution, verdict, landing, and hit-height for one video.

    Caller preconditions (intentionally not validated here): arrays are frame-aligned;
    `fps` is positive and finite; `court_info` is
    semantically `gate_court_info[str(video_id)]`; `gate_resolution_table` contains this video
    under a string index with width and height columns; and `homo_df` contains the video's full
    corner-column row.
    """
    resolved = resolve(base, fps)
    # Injected spans skip span finding, so serve-start would otherwise silently never run.
    if serve_start is not None and spans is not None:
        raise ValueError('serve_start cannot be combined with injected spans')
    if serve_start is not None and resolved.quiet_start_window is not None:
        raise ValueError('quiet_start_window cannot be combined with serve_start')
    if contacts is not None:
        # Injected contacts already carry the selected rally IDs. Only the preliminary
        # span pass is needed when callers did not inject spans; all discarded evidence
        # builders are irrelevant because segmentation is bypassed below.
        final_spans = spans if spans is not None else stage8_seg.find_rally_spans(
            track, resolved.thresholds, span_open=resolved.span_open,
            constants=resolved.constants, gap_state_demotion_bound=resolved.gap_state_demotion_bound,
            reentry_guard_variant=resolved.reentry_guard_variant,
            reentry_guard_buffer=resolved.reentry_guard_buffer,
            quiet_start_window=resolved.quiet_start_window,
        )
        raw_contacts = [ContactCandidate(rally_id, frame, None, None, None)
                        for rally_id, frames in contacts.items() for frame in frames]
    else:
        # First pass is span-only and unmasked: it supplies the single sticky EMA pass.
        bootstrap_spans = spans if spans is not None else stage8_seg.find_rally_spans(
            track, resolved.thresholds, span_open=resolved.span_open,
            constants=resolved.constants, gap_state_demotion_bound=resolved.gap_state_demotion_bound,
            reentry_guard_variant=resolved.reentry_guard_variant,
            reentry_guard_buffer=resolved.reentry_guard_buffer,
            quiet_start_window=resolved.quiet_start_window,
        )
        sticky = stage8_seg.build_sticky_result(
            track, bootstrap_spans, bboxes, scores, kps, ndet, str(video_id), gate_court_info,
            gate_resolution_table, court_box, resolution, resolved.constants.body_unit_half_window,
        )
        mask = dead_mask if dead_mask is not None else build_dead_mask(
            resolved.dead_mask_mode, len(track), fps, court_present=court_present,
            homography_rows=homography_rows, track=track, rally_spans=bootstrap_spans,
            cut_frames=cut_frames, keep_vote=keep_vote,
        )
        serve_options = None
        if serve_start is not None:
            serve_options = build_serve_options(
                serve_start, sticky, resolved.constants, resolution, resolved.span_open,
            )
        final_spans = spans
        final_spans, raw_contacts = stage8_seg.segment_video(
            track, positions=None, thresholds=resolved.thresholds,
            body_unit_half_window=resolved.constants.body_unit_half_window,
            span_open=resolved.span_open,
            replay_mask=mask, sticky_distances=sticky.distances, serve_start=serve_options,
            spans=final_spans, resolution=resolution, smoothing_mode=resolved.smoothing_mode,
            constants=resolved.constants, gap_state_demotion_bound=resolved.gap_state_demotion_bound,
            reentry_guard_variant=resolved.reentry_guard_variant,
            reentry_guard_buffer=resolved.reentry_guard_buffer,
            quiet_start_window=resolved.quiet_start_window,
        )
    spans, contacts = final_spans, raw_contacts

    filtered_contacts = scoring_filter(contacts)
    filtered_by_rally: dict[int, list[int]] = {}
    for contact in filtered_contacts:
        filtered_by_rally.setdefault(contact.rally_id, []).append(contact.contact_frame)

    striker_halves = []
    for rally_id in range(len(spans)):
        frames = filtered_by_rally.get(rally_id, [])
        guesses = [
            point_winner.attribute_half(
                frame, track, bboxes, scores, kps, court_box, net_band, resolution,
                resolved.constants.body_unit_half_window,
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
        track, bboxes, scores, kps, court_box, resolution,
    )
    band_m = point_winner.corner_error_band_m(video_id, homo_df, court_info, ref_err_px)

    verdict_rows: dict[int, object] = {}
    landings: dict[int, object | None] = {}
    for rally_id in range(len(spans)):
        striker = striker_halves[rally_id]
        if striker is None:
            continue
        frames = filtered_by_rally[rally_id]
        final_contact = frames[-1]
        next_start = spans[rally_id + 1][0] if rally_id + 1 < len(spans) else len(track)
        landing = point_winner.pick_landing(
            final_contact, next_start, track, mask, kin, landing_options, striker, net_band,
            resolution, court_info, resolved.constants, fps,
        )
        verdict_rows[rally_id] = point_winner.rally_verdict(
            rally_id, striker, next_servers[rally_id], landing, band_m,
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
        hit_height_by_frame=hit_height_by_frame, hit_height_failures=hit_height_failures,
    )
