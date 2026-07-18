"""GT-free annotation-chain composition for one video."""
from __future__ import annotations

from typing import NamedTuple

import annotator.point_winner as point_winner
import annotator.rally_segmentation as stage8_seg
from annotator.config import BaseAnnotatorConfig
from annotator.resolve import resolve
from annotator.types import ContactCandidate


OTHER_HALF = point_winner.OTHER_HALF


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
    track, bboxes, scores, kps, ndet, dead,
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
) -> AnnotatorResult:
    """Run segmentation, attribution, verdict, landing, and hit-height for one video.

    Caller preconditions (intentionally not validated here): arrays are frame-aligned; `dead` is
    a one-dimensional bool array of `len(track)`, where True means dead, and is not all True;
    `fps` is positive and finite; `court_info` is
    semantically `gate_court_info[str(video_id)]`; `gate_resolution_table` contains this video
    under a string index with width and height columns; and `homo_df` contains the video's full
    corner-column row.
    """
    resolved = resolve(base, fps)
    spans, contacts = stage8_seg.segment_video(
        track, positions=None, thresholds=resolved.thresholds,
        body_unit_half_window=resolved.constants.body_unit_half_window,
        span_open=stage8_seg.SpanOpen.BACK_FILL,
        replay_mask=dead, pose_bboxes=bboxes, pose_scores=scores, pose_kps=kps,
        pose_ndet=ndet, court_box=court_box, gate_video_id=str(video_id),
        gate_court_info=gate_court_info, gate_resolution_table=gate_resolution_table,
        resolution=resolution,
    )

    filtered_contacts = [
        contact for contact in contacts
        if contact.wrist_near is not False and contact.suppressed is not True
    ]
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
            final_contact, next_start, track, dead, kin, landing_options, striker, net_band,
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
