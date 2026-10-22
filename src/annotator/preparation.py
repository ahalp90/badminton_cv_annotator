"""Prepare shuttle, player and court evidence for annotation and model fitting."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from annotator import rally_segmentation
from annotator.config import ResolvedAnnotatorConfig, Thresholds
from annotator.courts.scenes import SceneCourt, build_scene_courts
from annotator.masks.dead import build_dead_mask
from annotator.masks.replay import filter_exclusion_mask
from annotator.rally import evidence as rally_evidence
from annotator.rally.serve import ServeStartOptions, build_serve_setup_inputs
from annotator.rally.spans import find_rally_spans
from annotator.types import ContactCandidate, ServeStartConfig, SpanOpen, StickyResult


@dataclass(frozen=True)
class AnnotationEvidence:
    """Original video arrays and completed rally/contact preparation."""

    identity: str
    resolution: tuple[float, float]
    track: np.ndarray
    pose_kps: np.ndarray
    bboxes: np.ndarray
    sticky: StickyResult
    tracker_intervals: Sequence[tuple[int, int]]
    scene_courts: tuple[SceneCourt, ...]
    net_band: tuple[float, float]
    resolved: ResolvedAnnotatorConfig
    raw_exclusion_mask: np.ndarray
    exclusion_mask: np.ndarray
    heuristic_spans: list[tuple[int, int]]
    heuristic_contacts: list[ContactCandidate]
    shuttle_hallucination_mask: np.ndarray
    source_codes: np.ndarray | None

    @property
    def fps(self) -> float:
        """Frame rate used to resolve preparation settings."""
        return self.resolved.fps

    @property
    def raw_contact_frames(self) -> list[int]:
        """Every candidate frame, preserving injected and detected contact order."""
        return [contact.contact_frame for contact in self.heuristic_contacts]


def prepare_annotation(
    track: np.ndarray,
    bboxes: np.ndarray | None,
    scores: np.ndarray | None,
    kps: np.ndarray | None,
    ndet: np.ndarray | None,
    *,
    resolved: ResolvedAnnotatorConfig,
    resolution: tuple[float, float] | None,
    video_id: int | str | None,
    net_band: tuple[float, float] | None,
    gate_court_info: dict[str, dict[str, object]] | None,
    gate_resolution_table: pd.DataFrame | None,
    court_present: np.ndarray | None,
    homography_rows: pd.DataFrame | list[dict[str, object]] | None,
    ref_err_px: float = 3.5,
    raw_exclusion_mask: np.ndarray | None = None,
    cut_frames: Sequence[int] | np.ndarray | None = None,
    keep_vote: np.ndarray | None = None,
    serve_start: ServeStartConfig | None = None,
    spans: list[tuple[int, int]] | None = None,
    contacts: dict[int, list[int]] | None = None,
    inpaint_codes: np.ndarray | None = None,
    shuttle_hallucination_mask: np.ndarray | None = None,
    court_invalid_is_excluded: bool = False,
) -> AnnotationEvidence:
    """Build masks, sticky players and initial rallies from the original arrays.

    Supplied contacts bypass candidate detection and retain their input order.
    Supplied spans bypass span finding. Neither changes the original shuttle track.
    """
    if serve_start is not None and spans is not None:
        raise ValueError('serve_start cannot be combined with injected spans')
    if serve_start is not None and resolved.quiet_start_window is not None:
        raise ValueError('quiet_start_window cannot be combined with serve_start')
    if homography_rows is None or court_present is None:
        raise ValueError('scene-gated sticky needs homography_rows and court_present')
    required_inputs = {
        'bboxes': bboxes, 'scores': scores, 'kps': kps, 'ndet': ndet,
        'resolution': resolution, 'video_id': video_id, 'net_band': net_band,
        'gate_court_info': gate_court_info, 'gate_resolution_table': gate_resolution_table,
    }
    missing = [name for name, value in required_inputs.items() if value is None]
    if missing:
        raise ValueError(f'annotation preparation requires {", ".join(missing)}')
    assert bboxes is not None
    assert scores is not None
    assert kps is not None
    assert ndet is not None
    assert resolution is not None
    assert video_id is not None
    assert net_band is not None
    assert gate_court_info is not None
    assert gate_resolution_table is not None
    if isinstance(homography_rows, pd.DataFrame):
        homography_rows = homography_rows.to_dict('records')
    shuttle_hallucination_mask, source_codes = build_shuttle_hallucination_mask(
        len(track), resolved.rejected_grades, inpaint_codes, shuttle_hallucination_mask,
    )
    scene_courts = build_scene_courts(homography_rows, resolution, ref_err_px)
    # Tracker and serve evidence need the original track before dead-time masking.
    segments = rally_evidence.tracker_segments(homography_rows, court_present, len(track))
    sticky = rally_evidence.build_sticky_result(
        track, segments, bboxes, scores, kps, ndet,
        str(video_id), gate_court_info, gate_resolution_table,
        resolution, resolved.thresholds.body_unit_half_window,
        scene_courts=scene_courts,
    )
    final_spans = spans
    if contacts is not None:
        final_spans = spans if spans is not None else find_rally_spans(track, resolved=resolved)
        raw_contacts = [
            ContactCandidate(rally_id, frame, None, None, None)
            for rally_id, frames in contacts.items()
            for frame in frames
        ]
    if raw_exclusion_mask is None:
        # The mask producer compares speed against unmasked rally spans.
        unmasked_spans = final_spans if final_spans is not None else find_rally_spans(track, resolved=resolved)
        raw_exclusion_mask = build_dead_mask(
            resolved.dead_mask_mode, len(track), resolved.fps, court_present=court_present,
            homography_rows=homography_rows, track=track, rally_spans=unmasked_spans,
            cut_frames=cut_frames, keep_vote=keep_vote,
            shuttle_hallucination_mask=shuttle_hallucination_mask,
        )
    serve_options = None
    if contacts is None and serve_start is not None:
        serve_options = build_serve_options(
            serve_start, sticky, resolved.thresholds, resolution, resolved.span_open,
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
    if contacts is None:
        final_spans, raw_contacts = rally_segmentation.segment_video(
            track, exclusion_mask=exclusion_mask, sticky_distances=sticky.distances,
            serve_start=serve_options, spans=final_spans, resolved=resolved,
        )
    return AnnotationEvidence(
        identity=str(video_id), resolution=resolution, track=track, pose_kps=kps, bboxes=bboxes,
        sticky=sticky, tracker_intervals=segments, scene_courts=scene_courts, net_band=net_band,
        resolved=resolved, raw_exclusion_mask=raw_exclusion_mask, exclusion_mask=exclusion_mask,
        heuristic_spans=final_spans, heuristic_contacts=raw_contacts,
        shuttle_hallucination_mask=shuttle_hallucination_mask, source_codes=source_codes,
    )


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
    config: ServeStartConfig,
    sticky: StickyResult,
    constants: Thresholds,
    resolution: tuple[float, float],
    span_open: SpanOpen | None = SpanOpen.BACK_FILL,
) -> ServeStartOptions:
    """Build sticky-sourced serve-start evidence from the unmasked cache.

    Build serve evidence before replay masking, which can otherwise remove
    the quiet setup frames needed to recognise a serve.
    """
    if config.close is not None and span_open is not None:
        raise ValueError('serve_start.close is unsupported with BACK_FILL')
    return ServeStartOptions(
        # The sticky path supplies body-height-normalised serve-setup evidence.
        threshold=config.threshold_bh, mode=config.mode, close=config.close,
        setup=build_serve_setup_inputs(sticky, resolution),
        stillness_threshold_bh=config.stillness_threshold_bh,
        lookback_frames=constants.serve_start_lookback_frames,
        stillness_window_frames=constants.serve_stillness_window_frames,
    )
