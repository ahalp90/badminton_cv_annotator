"""Prepare video evidence, select contacts and calculate rally outcomes."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import NamedTuple

import numpy as np
import pandas as pd

from annotator.config import BaseAnnotatorConfig, resolve
from annotator.contacts.selection import (
    contacts_from_sequences,
    select_heuristic_contacts,
)
from annotator.hybrid import HybridPrediction, predict_contacts
from annotator.models import DEFAULT_MODEL_DIRECTORY, AnnotatorModels, load_models
from annotator.outcomes import point_winner
from annotator.outcomes.video import (
    LandingHorizonRow,
    build_hit_heights,
    build_verdict_data,
)
from annotator.preparation import prepare_annotation
from annotator.sequence import ContactEvent
from annotator.types import ContactCandidate, ServeStartConfig
from shared.progress import report

DEFAULT_PREPROCESSING = BaseAnnotatorConfig()


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
    capture: RunCapture | None = None,
    court_invalid_is_excluded: bool = False,
    landing_error_band_m: float | None = None,
    landing_horizons_s: tuple[float, ...] = (),
) -> AnnotatorResult:
    """Find rallies and contacts, assign player halves, then estimate outcomes.

    Uses the supplied fitted models or the selected bundle. ``base`` must match
    its fitted preprocessing. ``heuristic_only=True`` selects rule-based contacts.

    Core arrays are normalised shuttle ``track`` (frames, 3), pose ``bboxes``
    (frames, detections, 4), ``scores`` (frames, detections), ``kps`` (frames,
    detections, 17, 2) and ``ndet`` counts (frames,). Preparation uses the original
    arrays and scene geometry. ``spans`` and ``contacts`` can supply prior research results.
    ``serve_start`` cannot be combined with supplied spans.

    Preparation requires all five arrays, ``resolution``, ``video_id``,
    ``net_band``, both ``gate_*`` inputs, ``court_present`` booleans (frames,) and scene
    ``homography_rows`` as a DataFrame or a list of row dictionaries. Outcomes
    also require ``landing_options``, ``court_info`` and either ``homo_df`` or
    ``landing_error_band_m``.

    ``raw_exclusion_mask`` supplies the dead-time mask before short-run filtering.
    ``inpaint_codes`` and ``shuttle_hallucination_mask`` are mutually exclusive
    event-rejection inputs. Frames outside accepted scene geometry are excluded;
    ``court_invalid_is_excluded`` also excludes frames without court presence.

    ``capture`` is reset at entry and receives owned mask copies after preparation
    succeeds. It also receives model predictions and requested landing horizons.
    ``rejection_diagnostics`` receives outcome rejection rows when supplied.
    Landing horizons require capture; their seconds must be finite, positive
    and strictly increasing.
    """
    if capture is not None:
        capture.raw_exclusion_mask = None
        capture.definitive_exclusion_mask = None
        capture.landing_horizon_rows.clear()
        capture.hybrid = None
    if heuristic_only and models is not None:
        raise ValueError('heuristic_only cannot be combined with fitted models')
    if not heuristic_only:
        models = load_models(DEFAULT_MODEL_DIRECTORY) if models is None else models
        if base is not None and base != models.preprocessing:
            raise ValueError('Preprocessing differs from the fitted model bundle; use its settings or refit the models')
        base = models.preprocessing
    if base is None:
        base = DEFAULT_PREPROCESSING
    validate_landing_horizons(capture, landing_horizons_s)
    required_outcome_inputs = {
        'landing_options': landing_options, 'court_info': court_info,
    }
    missing = [name for name, value in required_outcome_inputs.items() if value is None]
    if landing_error_band_m is None and homo_df is None:
        missing.append('homo_df or landing_error_band_m')
    if missing:
        raise ValueError(f'annotation outcomes require {", ".join(missing)}')
    assert landing_options is not None
    assert court_info is not None

    report("finding rallies")
    evidence = prepare_annotation(
        track, bboxes, scores, kps, ndet,
        resolved=resolve(base, fps), resolution=resolution, video_id=video_id, net_band=net_band,
        gate_court_info=gate_court_info, gate_resolution_table=gate_resolution_table,
        court_present=court_present, homography_rows=homography_rows, ref_err_px=ref_err_px,
        raw_exclusion_mask=raw_exclusion_mask, cut_frames=cut_frames, keep_vote=keep_vote,
        serve_start=serve_start, spans=spans, contacts=contacts, inpaint_codes=inpaint_codes,
        shuttle_hallucination_mask=shuttle_hallucination_mask,
        court_invalid_is_excluded=court_invalid_is_excluded,
    )
    if capture is not None:
        capture.raw_exclusion_mask = evidence.raw_exclusion_mask.copy()
        capture.definitive_exclusion_mask = evidence.exclusion_mask.copy()
    contact_events = ()
    rally_confidence = ()
    if models is None:
        contact_data = select_heuristic_contacts(evidence)
        final_contacts = evidence.heuristic_contacts
    else:
        prediction = predict_contacts(evidence, models)
        if capture is not None:
            capture.hybrid = prediction
        contact_data = contacts_from_sequences(prediction.refined.sequences)
        final_contacts = contact_data.filtered_contacts
        contact_events = prediction.refined.events
        rally_confidence = tuple(float(score) for score in prediction.confidence.scores)
    report("resolving outcomes")
    # The static homography table may use integer keys; preserve the caller's ID.
    band_m = (
        landing_error_band_m if landing_error_band_m is not None
        else point_winner.corner_error_band_m(video_id, homo_df, court_info, ref_err_px)
    )
    verdict_data = build_verdict_data(
        evidence, contact_data, court_info=court_info, landing_options=landing_options, band_m=band_m,
        rejection_diagnostics=rejection_diagnostics, landing_horizons_s=landing_horizons_s,
        horizon_rows=capture.landing_horizon_rows if capture is not None else None,
    )
    report("calculating hit heights")
    hit_height_by_frame, hit_height_failures = build_hit_heights(evidence, contact_data)
    return AnnotatorResult(
        spans=contact_data.spans,
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
        contacts where hit-height evidence is unavailable, including missing scene geometry
        or an invisible shuttle at that frame.
    :param contact_events: Full final tree-model stream, including contacts outside
        every rally. Each event retains its score and final court-half assignment.
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
