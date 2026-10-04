"""Run the mixed heuristic and tree annotator for one video."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from annotator import rally_segmentation
from annotator.config import BaseAnnotatorConfig, ResolvedAnnotatorConfig
from annotator.courts.scenes import SceneCourt, build_scene_courts
from annotator.hybrid import ContactEvidence, HybridPrediction, predict_contacts
from annotator.masks.dead import build_dead_mask
from annotator.masks.replay import filter_short_exclusion_runs
from annotator.models import AnnotatorModels, load_models
from annotator.outcomes import point_winner
from annotator.outcomes.video import (
    LandingHorizonRow,
    build_contact_data,
    build_hit_heights,
    build_refined_contact_data,
    build_verdict_data,
)
from annotator.resolve import resolve
from annotator.sequence import ContactEvent
from annotator.types import ContactCandidate, ServeStartConfig, StickyResult
from shared.progress import report

DEFAULT_PREPROCESSING = BaseAnnotatorConfig()
DEFAULT_MODEL_DIRECTORY = Path('models/annotator')


def build_shuttle_hallucination_mask(
    n_frames: int, rejected_grades: frozenset[int],
    inpaint_codes: np.ndarray | None, shuttle_hallucination_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Adapt source grades into the one boolean mask read by event rules."""
    if inpaint_codes is not None and shuttle_hallucination_mask is not None:
        raise ValueError('inpaint_codes and shuttle_hallucination_mask are mutually exclusive')
    if inpaint_codes is not None:
        if inpaint_codes.ndim != 1 or len(inpaint_codes) != n_frames:
            raise ValueError('inpaint_codes must be a frame-aligned one-dimensional array')
        return np.isin(inpaint_codes, tuple(rejected_grades)), inpaint_codes
    if shuttle_hallucination_mask is not None:
        if (shuttle_hallucination_mask.ndim != 1
                or len(shuttle_hallucination_mask) != n_frames
                or shuttle_hallucination_mask.dtype != np.bool_):
            raise ValueError('shuttle_hallucination_mask must be a frame-aligned boolean array')
        return shuttle_hallucination_mask, None
    return np.zeros(n_frames, dtype=bool), None


def build_serve_options(
    config, sticky, constants, resolution, span_open=rally_segmentation.SpanOpen.BACK_FILL,
) -> rally_segmentation.ServeStartOptions:
    """Build sticky-sourced serve-start evidence from the unmasked cache.

    Build serve evidence before replay masking, which can otherwise remove
    the quiet setup frames needed to recognise a serve.
    """
    if config.close is not None and span_open is not None:
        raise ValueError('serve_start.close is unsupported with BACK_FILL')
    return rally_segmentation.ServeStartOptions(
        # The sticky path supplies body-height-normalised serve-setup evidence.
        dist=None, threshold=config.threshold_bh, mode=config.mode, close=config.close,
        setup=rally_segmentation.build_serve_setup_inputs(sticky, resolution),
        stillness_threshold_bh=config.stillness_threshold_bh,
        lookback_frames=constants.serve_start_lookback_frames,
        stillness_window_frames=constants.serve_stillness_window_frames,
    )


class AnnotatorResult(NamedTuple):
    """Final rallies, contacts, player sides and outcomes for one video.

    :param spans: detected rally spans, `[(start, end), ...]`; rally_id is the list index.
    :param contacts: Contact rows owned by rallies. Tree-selected rows have no
        heuristic wrist or suppression verdict. In heuristic-only comparisons,
        these fields retain the original candidate gate results.
    :param filtered_contacts: rows where `wrist_near is not False and suppressed is not True`.
        This keeps suppression winners and the unmeasured no-gate path — the set
        `annotator.evaluation.scoring.score_contacts` scores the ball_round column against.
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
    :param contact_events: Full final tree-model stream, including contacts outside
        every rally. Each event retains its score and individual player side.
    :param rally_confidence: Review-ranking score per rally, in span order.
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
    contact_events: tuple[ContactEvent, ...] = ()
    rally_confidence: tuple[float, ...] = ()


@dataclass
class RunCapture:
    """Caller-owned copies of the masks and landing diagnostics from one run."""

    raw_exclusion_mask: np.ndarray | None = None
    definitive_exclusion_mask: np.ndarray | None = None
    landing_horizon_rows: list[LandingHorizonRow] = field(default_factory=list)
    hybrid: HybridPrediction | None = None
    contact_evidence: ContactEvidence | None = None


@dataclass(frozen=True)
class CourtInputs:
    """Court and player evidence used only by court-dependent modes."""

    bboxes: np.ndarray | None
    scores: np.ndarray | None
    kps: np.ndarray | None
    ndet: np.ndarray | None
    resolution: tuple[float, float] | None
    video_id: int | str | None
    court_info: dict[str, object] | None
    homo_df: pd.DataFrame | None
    gate_court_info: dict[str, dict[str, object]] | None
    gate_resolution_table: pd.DataFrame | None
    court_present: np.ndarray | None
    homography_rows: pd.DataFrame | list[dict[str, object]] | None
    cut_frames: Sequence[int] | np.ndarray | None
    keep_vote: np.ndarray | None


@dataclass(frozen=True)
class SegmentationData:
    """Final spans, contacts, and exclusion mask from the segmentation phase."""

    spans: list[tuple[int, int]]
    contacts: list[ContactCandidate]
    definitive_exclusion_mask: np.ndarray
    sticky: StickyResult | None = None
    tracker_intervals: Sequence[tuple[int, int]] = ()


def validate_landing_horizons(
    capture: RunCapture | None, landing_horizons_s: tuple[float, ...],
) -> None:
    """Validate optional horizon capture without changing its rounding policy."""
    if not landing_horizons_s:
        return
    if capture is None:
        raise ValueError('landing_horizons_s requires capture')
    for horizon in landing_horizons_s:
        if not math.isfinite(horizon) or horizon <= 0:
            raise ValueError('landing_horizons_s must contain finite positive values')
    for earlier, later in pairwise(landing_horizons_s):
        if later <= earlier:
            raise ValueError('landing_horizons_s must be strictly increasing')


def build_span_options(resolved: ResolvedAnnotatorConfig) -> dict[str, Any]:
    """Return the shared options for every span-finding path."""
    return {
        'thresholds': resolved.thresholds,
        'span_open': resolved.span_open,
        'constants': resolved.constants,
        'gap_state_demotion_bound': resolved.gap_state_demotion_bound,
        'reentry_guard_variant': resolved.reentry_guard_variant,
        'reentry_guard_buffer': resolved.reentry_guard_buffer,
        'quiet_start_window': resolved.quiet_start_window,
    }


def validate_run_inputs(
    court: CourtInputs,
    *,
    serve_start: ServeStartConfig | None,
    spans: list[tuple[int, int]] | None,
    resolved: ResolvedAnnotatorConfig,
    court_optional: bool,
    stop_after_segmentation: bool,
    landing_options: Any,
    net_band: tuple[float, float] | None,
    landing_error_band_m: float | None,
) -> object | None:
    """Validate mode-specific inputs and normalise scene rows."""
    if serve_start is not None and spans is not None:
        raise ValueError('serve_start cannot be combined with injected spans')
    if resolved.quiet_start_window is not None and resolved.span_open is not None:
        raise ValueError('quiet_start_window cannot be combined with span_open')
    if serve_start is not None and resolved.quiet_start_window is not None:
        raise ValueError('quiet_start_window cannot be combined with serve_start')
    if court_optional and not stop_after_segmentation:
        raise ValueError('court_optional requires stop_after_segmentation')
    if court_optional:
        supplied_optional_inputs = {
            'homography_rows': court.homography_rows,
            'court_present': court.court_present,
            'bboxes': court.bboxes,
            'scores': court.scores,
            'kps': court.kps,
            'ndet': court.ndet,
            'resolution': court.resolution,
            'video_id': court.video_id,
            'gate_court_info': court.gate_court_info,
            'gate_resolution_table': court.gate_resolution_table,
            'serve_start': serve_start,
        }
        supplied = [name for name, value in supplied_optional_inputs.items() if value is not None]
        if supplied:
            raise ValueError(f'court_optional rejects supplied inputs: {", ".join(supplied)}')
        return court.homography_rows

    if court.homography_rows is None or court.court_present is None:
        raise ValueError('scene-gated sticky needs homography_rows and court_present')
    required_sticky_inputs = {
        'bboxes': court.bboxes,
        'scores': court.scores,
        'kps': court.kps,
        'ndet': court.ndet,
        'resolution': court.resolution,
        'video_id': court.video_id,
        'gate_court_info': court.gate_court_info,
        'gate_resolution_table': court.gate_resolution_table,
    }
    missing_sticky_inputs = [name for name, value in required_sticky_inputs.items() if value is None]
    if missing_sticky_inputs:
        raise ValueError(f'normal mode requires {", ".join(missing_sticky_inputs)}')
    if not stop_after_segmentation:
        required_downstream_inputs = {
            'landing_options': landing_options,
            'net_band': net_band,
            'court_info': court.court_info,
        }
        missing_downstream_inputs = [
            name for name, value in required_downstream_inputs.items() if value is None
        ]
        if landing_error_band_m is None and court.homo_df is None:
            missing_downstream_inputs.append('homo_df or landing_error_band_m')
        if missing_downstream_inputs:
            raise ValueError(f'full-chain mode requires {", ".join(missing_downstream_inputs)}')

    if hasattr(court.homography_rows, 'to_dict'):
        return court.homography_rows.to_dict('records')
    return court.homography_rows


def empty_result(
    spans: list[tuple[int, int]], contacts: list[ContactCandidate],
) -> AnnotatorResult:
    """Return the existing segmentation-only result shape."""
    return AnnotatorResult(
        spans=spans, contacts=contacts, filtered_contacts=[], filtered_by_rally={},
        striker_halves=[], n_strokes_list=[], next_servers=[], fitted_first_all=[],
        verdict_rows={}, landings={}, geometric_verdict_rows={}, hit_height_by_frame={},
        hit_height_failures=[],
    )


def injected_contact_rows(contacts: dict[int, list[int]]) -> list[ContactCandidate]:
    """Adapt injected rally-indexed frames to unmeasured contact rows."""
    return [
        ContactCandidate(rally_id, frame, None, None, None)
        for rally_id, frames in contacts.items()
        for frame in frames
    ]


def finalise_exclusion_mask(
    raw_exclusion_mask: np.ndarray,
    *,
    n_frames: int,
    replay_mask_min_frames: int,
    capture: RunCapture | None,
    court_present: np.ndarray | None = None,
    include_court_invalid: bool = False,
    check_definitive_mask: bool = True,
) -> np.ndarray:
    """Validate, filter, optionally union, and capture an exclusion mask."""
    if len(raw_exclusion_mask) != n_frames:
        raise ValueError(f'mask length {len(raw_exclusion_mask)} != track length {n_frames}')
    if capture is not None:
        capture.raw_exclusion_mask = raw_exclusion_mask.copy()
    if raw_exclusion_mask.all():
        raise ValueError('mask is all True: no live frame to anchor a frozen position to')
    definitive_exclusion_mask = filter_short_exclusion_runs(
        raw_exclusion_mask, replay_mask_min_frames,
    )
    if include_court_invalid:
        assert court_present is not None
        definitive_exclusion_mask = definitive_exclusion_mask | ~court_present
    if capture is not None:
        capture.definitive_exclusion_mask = definitive_exclusion_mask.copy()
    if check_definitive_mask and definitive_exclusion_mask.all():
        raise ValueError('mask is all True: no live frame to anchor a frozen position to')
    return definitive_exclusion_mask


def run_court_optional_segmentation(
    track: np.ndarray,
    positions: np.ndarray | None,
    spans: list[tuple[int, int]] | None,
    contacts: dict[int, list[int]] | None,
    raw_exclusion_mask: np.ndarray | None,
    capture: RunCapture | None,
    resolved: ResolvedAnnotatorConfig,
    span_options: dict[str, Any],
) -> SegmentationData:
    """Run the court-free segmentation-only mode."""
    raw_mask = (
        raw_exclusion_mask
        if raw_exclusion_mask is not None
        else np.zeros(len(track), dtype=bool)
    )
    definitive_exclusion_mask = finalise_exclusion_mask(
        raw_mask,
        n_frames=len(track),
        replay_mask_min_frames=resolved.constants.replay_mask_min_frames,
        capture=capture,
        check_definitive_mask=False,
    )
    if contacts is None:
        final_spans, raw_contacts = rally_segmentation.segment_video(
            track,
            positions=positions,
            exclusion_mask=definitive_exclusion_mask,
            sticky_distances=None,
            spans=spans,
            smoothing_mode=resolved.smoothing_mode,
            **span_options,
        )
    else:
        final_spans = spans if spans is not None else rally_segmentation.find_rally_spans(
            track, **span_options,
        )
        raw_contacts = injected_contact_rows(contacts)
    return SegmentationData(final_spans, raw_contacts, definitive_exclusion_mask)


def run_court_segmentation(
    track: np.ndarray,
    *,
    fps: float,
    court: CourtInputs,
    homography_rows: object,
    scene_courts: tuple[SceneCourt, ...],
    raw_exclusion_mask: np.ndarray | None,
    positions: np.ndarray | None,
    serve_start: ServeStartConfig | None,
    spans: list[tuple[int, int]] | None,
    contacts: dict[int, list[int]] | None,
    shuttle_hallucination_mask: np.ndarray,
    capture: RunCapture | None,
    court_invalid_is_excluded: bool,
    stop_after_segmentation: bool,
    resolved: ResolvedAnnotatorConfig,
    span_options: dict[str, Any],
) -> SegmentationData:
    """Build sticky evidence, exclusion masks, spans, and contacts."""
    # Sticky evidence must see the original track. Build it once before any replay masking.
    segments = rally_segmentation.tracker_segments(
        homography_rows, court.court_present, len(track),
    )
    sticky = rally_segmentation.build_sticky_result(
        track, segments, court.bboxes, court.scores, court.kps, court.ndet,
        str(court.video_id), court.gate_court_info, court.gate_resolution_table,
        court.resolution, resolved.constants.body_unit_half_window,
        scene_courts=scene_courts,
    )

    serve_options = None
    if contacts is not None:
        final_spans = spans if spans is not None else rally_segmentation.find_rally_spans(
            track, **span_options,
        )
        raw_contacts = injected_contact_rows(contacts)
        if raw_exclusion_mask is None:
            raw_exclusion_mask = build_dead_mask(
                resolved.dead_mask_mode, len(track), fps, court_present=court.court_present,
                homography_rows=homography_rows, track=track, rally_spans=final_spans,
                cut_frames=court.cut_frames, keep_vote=court.keep_vote,
                shuttle_hallucination_mask=shuttle_hallucination_mask,
            )
    else:
        if raw_exclusion_mask is None:
            # The mask builder needs unmasked rally spans as its normal-speed baseline.
            bootstrap_spans = (
                spans
                if spans is not None
                else rally_segmentation.find_rally_spans(track, **span_options)
            )
            raw_exclusion_mask = build_dead_mask(
                resolved.dead_mask_mode, len(track), fps, court_present=court.court_present,
                homography_rows=homography_rows, track=track, rally_spans=bootstrap_spans,
                cut_frames=court.cut_frames, keep_vote=court.keep_vote,
                shuttle_hallucination_mask=shuttle_hallucination_mask,
            )
        final_spans = spans
        if serve_start is not None:
            assert court.resolution is not None
            serve_options = build_serve_options(
                serve_start, sticky, resolved.constants, court.resolution, resolved.span_open,
            )

    assert raw_exclusion_mask is not None
    # Full-chain contact attribution needs the geometry of its own scene.
    usable_court_frames = np.zeros(len(track), dtype=bool)
    for scene in scene_courts:
        start = max(0, scene.start_frame)
        end = max(0, min(len(track), scene.end_frame))
        usable_court_frames[start:end] = True
    if court_invalid_is_excluded:
        usable_court_frames = usable_court_frames & court.court_present
    definitive_exclusion_mask = finalise_exclusion_mask(
        raw_exclusion_mask,
        n_frames=len(track),
        replay_mask_min_frames=resolved.constants.replay_mask_min_frames,
        capture=capture,
        court_present=usable_court_frames,
        include_court_invalid=not stop_after_segmentation,
    )
    if contacts is None:
        final_spans, raw_contacts = rally_segmentation.segment_video(
            track,
            positions=positions,
            exclusion_mask=definitive_exclusion_mask,
            sticky_distances=sticky.distances,
            serve_start=serve_options,
            spans=final_spans,
            smoothing_mode=resolved.smoothing_mode,
            **span_options,
        )
    return SegmentationData(final_spans, raw_contacts, definitive_exclusion_mask, sticky, segments)


def run_video(
    track: np.ndarray,
    bboxes: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    kps: np.ndarray | None = None,
    ndet: np.ndarray | None = None,
    *,
    fps: float,
    base: BaseAnnotatorConfig | None = None,
    models: AnnotatorModels | None = None,
    heuristic_only: bool = False,
    landing_options: point_winner.LandingFilterOptions | None = None,
    net_band: tuple[float, float] | None = None,
    resolution: tuple[float, float] | None = None,
    video_id: int | str | None = None,
    court_info: dict[str, object] | None = None,
    homo_df: pd.DataFrame | None = None,
    gate_court_info: dict[str, dict[str, object]] | None = None,
    gate_resolution_table: pd.DataFrame | None = None,
    ref_err_px: float = 3.5,
    raw_exclusion_mask: np.ndarray | None = None,
    positions: np.ndarray | None = None,
    court_present: np.ndarray | None = None,
    homography_rows: pd.DataFrame | list[dict[str, object]] | None = None,
    cut_frames: Sequence[int] | np.ndarray | None = None,
    keep_vote: np.ndarray | None = None,
    serve_start: ServeStartConfig | None = None,
    spans: list[tuple[int, int]] | None = None,
    contacts: dict[int, list[int]] | None = None,
    inpaint_codes: np.ndarray | None = None,
    shuttle_hallucination_mask: np.ndarray | None = None,
    rejection_diagnostics: list[dict[str, object]] | None = None,
    court_optional: bool = False,
    stop_after_segmentation: bool = False,
    capture: RunCapture | None = None,
    court_invalid_is_excluded: bool = False,
    landing_error_band_m: float | None = None,
    landing_horizons_s: tuple[float, ...] = (),
) -> AnnotatorResult:
    """Run segmentation, attribution, verdict, landing, and hit-height for one video.

    Full annotation uses ``models`` or loads ``models/annotator`` from the
    working directory. Its preprocessing settings must match the fitted bundle.
    ``heuristic_only`` supports feature preparation and historical comparisons;
    normal annotation runs the complete mixed heuristic/tree system.

    Core arrays are ``track`` as ``(t, 3)`` normalised shuttle ``[x, y,
    visibility]``; ``bboxes`` as ``(t, n_max, 4)`` xyxy pose boxes; ``scores``
    as ``(t, n_max)`` detection scores; ``kps`` as ``(t, n_max, 17, 2)`` pose
    coordinates; and ``ndet`` as ``(t,)`` detection counts. ``fps`` and ``base``
    select the resolved frame-rate policy.

    Court-dependent mode also requires ``resolution``, ``video_id``,
    ``gate_court_info``, ``gate_resolution_table``, ``court_present`` as
    ``(t,)`` booleans, and scene ``homography_rows``. The full chain additionally
    requires ``landing_options``, ``net_band``, ``court_info``, and either the
    static ``homo_df`` or a parent-supplied ``landing_error_band_m``.

    ``raw_exclusion_mask`` injects the ``(t,)`` dead-time producer output.
    Otherwise the resolved dead-mask mode consumes replay inputs plus
    composition ``cut_frames`` and ``keep_vote`` as needed. ``inpaint_codes``
    and ``shuttle_hallucination_mask`` are mutually exclusive frame-aligned
    sources for event rejection. ``positions`` is the optional ``(t, 2, 2)``
    player-position evidence used by contact diagnostics.

    ``spans`` and ``contacts`` inject prior stage results. ``serve_start`` cannot
    accompany injected spans. ``court_optional`` is valid only with
    ``stop_after_segmentation``. It rejects scene and court-presence inputs, all
    four pose arrays, ``resolution``, ``video_id``, both sticky-gate inputs, and
    ``serve_start``; other full-chain arguments are unused in that mode.

    ``capture`` is caller-owned, cleared at entry, and receives mask copies.
    Full-chain mode also appends to caller-owned ``rejection_diagnostics`` and
    records requested ``landing_horizons_s`` in ``capture``. Horizons require a
    capture and must be finite, positive, and strictly increasing.
    Full-chain mode always excludes frames outside the supplied scene geometry.
    ``court_invalid_is_excluded`` also excludes frames where ``court_present``
    is false. Neither exclusion is added when stopping after segmentation.
    """
    if heuristic_only and models is not None:
        raise ValueError('heuristic_only cannot be combined with fitted models')
    if not stop_after_segmentation and not heuristic_only:
        models = load_models(DEFAULT_MODEL_DIRECTORY) if models is None else models
        if base is not None and base != models.preprocessing:
            raise ValueError('Preprocessing differs from the fitted model bundle; use its settings or refit the models')
        base = models.preprocessing
    if base is None:
        base = DEFAULT_PREPROCESSING

    court = CourtInputs(
        bboxes=bboxes, scores=scores, kps=kps, ndet=ndet, resolution=resolution,
        video_id=video_id, court_info=court_info, homo_df=homo_df,
        gate_court_info=gate_court_info, gate_resolution_table=gate_resolution_table,
        court_present=court_present, homography_rows=homography_rows, cut_frames=cut_frames,
        keep_vote=keep_vote,
    )
    if capture is not None:
        capture.raw_exclusion_mask = None
        capture.definitive_exclusion_mask = None
        capture.landing_horizon_rows.clear()
        capture.hybrid = None
        capture.contact_evidence = None
    validate_landing_horizons(capture, landing_horizons_s)
    if capture is not None and raw_exclusion_mask is not None:
        capture.raw_exclusion_mask = raw_exclusion_mask.copy()

    report("finding rallies")
    resolved = resolve(base, fps)
    span_options = build_span_options(resolved)
    homography_rows = validate_run_inputs(
        court,
        serve_start=serve_start,
        spans=spans,
        resolved=resolved,
        court_optional=court_optional,
        stop_after_segmentation=stop_after_segmentation,
        landing_options=landing_options,
        net_band=net_band,
        landing_error_band_m=landing_error_band_m,
    )
    shuttle_hallucination_mask, source_codes = build_shuttle_hallucination_mask(
        len(track), resolved.rejected_grades, inpaint_codes, shuttle_hallucination_mask,
    )

    if court_optional:
        segmentation = run_court_optional_segmentation(
            track, positions=positions, spans=spans, contacts=contacts,
            raw_exclusion_mask=raw_exclusion_mask, capture=capture,
            resolved=resolved, span_options=span_options,
        )
        return empty_result(segmentation.spans, segmentation.contacts)

    assert homography_rows is not None
    scene_courts = build_scene_courts(homography_rows, court.resolution, ref_err_px)
    segmentation = run_court_segmentation(
        track,
        fps=fps,
        court=court,
        homography_rows=homography_rows,
        scene_courts=scene_courts,
        raw_exclusion_mask=raw_exclusion_mask,
        positions=positions,
        serve_start=serve_start,
        spans=spans,
        contacts=contacts,
        shuttle_hallucination_mask=shuttle_hallucination_mask,
        capture=capture,
        court_invalid_is_excluded=court_invalid_is_excluded,
        stop_after_segmentation=stop_after_segmentation,
        resolved=resolved,
        span_options=span_options,
    )
    if stop_after_segmentation:
        return empty_result(segmentation.spans, segmentation.contacts)

    assert segmentation.sticky is not None
    assert court.bboxes is not None
    assert court.kps is not None
    assert court.resolution is not None
    assert court.court_info is not None
    assert net_band is not None
    final_spans = segmentation.spans
    final_contacts = segmentation.contacts
    contact_events = ()
    rally_confidence = ()
    evidence = ContactEvidence(
        identity=str(video_id), fps=fps, resolution=court.resolution,
        track=track, pose_kps=court.kps, bboxes=court.bboxes, sticky=segmentation.sticky,
        tracker_intervals=segmentation.tracker_intervals,
        exclusion_mask=segmentation.definitive_exclusion_mask,
        heuristic_spans=segmentation.spans,
        raw_contact_frames=[contact.contact_frame for contact in segmentation.contacts],
        scene_courts=scene_courts, net_band=net_band,
        shuttle_hallucination_mask=shuttle_hallucination_mask,
    )
    if capture is not None:
        capture.contact_evidence = evidence
    if models is None:
        contact_data = build_contact_data(
            spans=final_spans, contacts=final_contacts,
            definitive_exclusion_mask=segmentation.definitive_exclusion_mask,
            track=track, sticky=segmentation.sticky, bboxes=court.bboxes, net_band=net_band,
            scene_courts=scene_courts,
        )
    else:
        prediction = predict_contacts(evidence, models)
        if capture is not None:
            capture.hybrid = prediction
        final_spans = [(sequence.start_frame, sequence.end_frame) for sequence in prediction.refined.sequences]
        contact_data = build_refined_contact_data(prediction.refined.sequences)
        final_contacts = contact_data.filtered_contacts
        contact_events = prediction.refined.events
        rally_confidence = tuple(float(score) for score in prediction.confidence.scores)
    report("resolving outcomes")
    verdict_data = build_verdict_data(
        track,
        fps=fps, spans=final_spans,
        definitive_exclusion_mask=segmentation.definitive_exclusion_mask,
        sticky=segmentation.sticky, contact_data=contact_data, resolved=resolved,
        kps=court.kps, resolution=court.resolution,
        video_id=court.video_id, homo_df=court.homo_df, court_info=court.court_info,
        landing_options=landing_options, net_band=net_band, ref_err_px=ref_err_px,
        landing_error_band_m=landing_error_band_m,
        shuttle_hallucination_mask=shuttle_hallucination_mask,
        source_codes=source_codes, rejection_diagnostics=rejection_diagnostics,
        landing_horizons_s=landing_horizons_s,
        horizon_rows=capture.landing_horizon_rows if capture is not None else None,
        scene_courts=scene_courts,
    )
    report("calculating hit heights")
    hit_height_by_frame, hit_height_failures = build_hit_heights(
        spans=final_spans, filtered_by_rally=contact_data.filtered_by_rally,
        track=track, net_band=net_band, resolution=court.resolution,
        scene_courts=scene_courts,
    )
    return AnnotatorResult(
        spans=final_spans,
        contacts=final_contacts,
        filtered_contacts=contact_data.filtered_contacts,
        filtered_by_rally=contact_data.filtered_by_rally,
        striker_halves=contact_data.striker_halves,
        n_strokes_list=contact_data.n_strokes_list,
        next_servers=contact_data.next_servers,
        fitted_first_all=contact_data.fitted_first_all,
        verdict_rows=verdict_data.verdict_rows,
        landings=verdict_data.landings,
        geometric_verdict_rows=verdict_data.geometric_verdict_rows,
        hit_height_by_frame=hit_height_by_frame,
        hit_height_failures=hit_height_failures,
        contact_events=contact_events,
        rally_confidence=rally_confidence,
    )
