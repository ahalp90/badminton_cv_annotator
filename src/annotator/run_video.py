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
from annotator.contacts.selection import select_heuristic_contacts
from annotator.hybrid import HybridPrediction, predict_contacts
from annotator.models import DEFAULT_MODEL_DIRECTORY, AnnotatorModels, load_models
from annotator.outcomes import point_winner
from annotator.outcomes.video import (
    LandingHorizonRow,
    RallyOutcome,
    build_hit_heights,
    build_verdict_data,
)
from annotator.preparation import prepare_annotation
from annotator.sequence.contacts import ContactEvent, FinishedRally
from annotator.types import ContactCandidate
from shared.progress import report

DEFAULT_PREPROCESSING = BaseAnnotatorConfig()


def run_video(
    track: np.ndarray,
    bboxes: np.ndarray,
    scores: np.ndarray,
    kps: np.ndarray,
    ndet: np.ndarray,
    *,
    fps: float,
    landing_options: point_winner.LandingFilterOptions,
    net_band: tuple[float, float],
    resolution: tuple[float, float],
    video_id: int | str,
    gate_court_info: dict[str, dict[str, object]],
    gate_resolution_table: pd.DataFrame,
    court_present: np.ndarray,
    homography_rows: pd.DataFrame | list[dict[str, object]],
    base: BaseAnnotatorConfig | None = None,
    models: AnnotatorModels | None = None,
    heuristic_only: bool = False,
    ref_err_px: float = 3.5,
    raw_exclusion_mask: np.ndarray | None = None,
    cut_frames: Sequence[int] | np.ndarray | None = None,
    keep_vote: np.ndarray | None = None,
    spans: list[tuple[int, int]] | None = None,
    contacts: dict[int, list[int]] | None = None,
    inpaint_codes: np.ndarray | None = None,
    shuttle_hallucination_mask: np.ndarray | None = None,
    rejection_diagnostics: list[dict[str, object]] | None = None,
    capture: RunCapture | None = None,
    court_invalid_is_excluded: bool = False,
    landing_horizons_s: tuple[float, ...] = (),
) -> AnnotatorResult:
    """Find rallies and contacts, assign player halves, then estimate outcomes.

    Uses the supplied fitted models or the selected bundle. ``base`` must match
    its fitted preprocessing. ``heuristic_only=True`` selects rule-based contacts.

    Core arrays are normalised shuttle ``track`` (frames, 3), pose ``bboxes``
    (frames, detections, 4), ``scores`` (frames, detections), ``kps`` (frames,
    detections, 17, 2) and ``ndet`` counts (frames,). Preparation uses the original
    arrays and scene geometry. ``spans`` and ``contacts`` can supply prior research results.

    Preparation requires all five arrays, ``resolution``, ``video_id``,
    ``net_band``, both ``gate_*`` inputs, ``court_present`` booleans (frames,) and scene
    ``homography_rows`` as a DataFrame or a list of row dictionaries. Outcomes
    use ``landing_options`` and the selected scene's court geometry.

    ``raw_exclusion_mask`` supplies the dead-time mask before short-run filtering.
    ``inpaint_codes`` and ``shuttle_hallucination_mask`` are mutually exclusive
    event-rejection inputs. Frames outside accepted scene geometry are excluded;
    ``court_invalid_is_excluded`` also excludes frames without court presence.

    ``capture`` is reset at entry and receives owned mask copies after preparation
    succeeds. It also receives model predictions and requested landing horizons.
    ``rejection_diagnostics`` receives rejection rows when supplied, grouped by
    filtering and outcome stages rather than rally order.
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
    report("finding rallies")
    evidence = prepare_annotation(
        track, bboxes, scores, kps, ndet,
        resolved=resolve(base, fps), resolution=resolution, video_id=video_id, net_band=net_band,
        gate_court_info=gate_court_info, gate_resolution_table=gate_resolution_table,
        court_present=court_present, homography_rows=homography_rows, ref_err_px=ref_err_px,
        raw_exclusion_mask=raw_exclusion_mask, cut_frames=cut_frames, keep_vote=keep_vote,
        spans=spans, contacts=contacts, inpaint_codes=inpaint_codes,
        shuttle_hallucination_mask=shuttle_hallucination_mask,
        court_invalid_is_excluded=court_invalid_is_excluded,
    )
    if capture is not None:
        capture.raw_exclusion_mask = evidence.raw_exclusion_mask.copy()
        capture.definitive_exclusion_mask = evidence.exclusion_mask.copy()
    contact_events = ()
    rally_confidence = ()
    if models is None:
        rallies = select_heuristic_contacts(evidence, rejection_diagnostics)
        heuristic_candidates = evidence.heuristic_contacts
    else:
        prediction = predict_contacts(evidence, models)
        if capture is not None:
            capture.hybrid = prediction
        rallies = prediction.refined.rallies
        heuristic_candidates = []
        contact_events = prediction.refined.events
        rally_confidence = tuple(float(score) for score in prediction.confidence.scores)
    report("resolving outcomes")
    outcomes = build_verdict_data(
        evidence, rallies, landing_options=landing_options,
        rejection_diagnostics=rejection_diagnostics, landing_horizons_s=landing_horizons_s,
        horizon_rows=capture.landing_horizon_rows if capture is not None else None,
    )
    report("calculating hit heights")
    hit_heights, hit_height_failures = build_hit_heights(evidence, rallies)
    return AnnotatorResult(
        rallies=rallies, outcomes=outcomes, hit_heights=hit_heights, hit_height_failures=hit_height_failures,
        contact_events=contact_events, heuristic_candidates=heuristic_candidates, rally_confidence=rally_confidence,
    )


class AnnotatorResult(NamedTuple):
    """Finished rallies and their outcomes for one source video.

    Rally IDs are positions in ``rallies``. Each rally retains its selected contacts,
    fitted final striker and bounds. ``outcomes`` contains the rallies with a resolved
    final striker. Hit heights and unavailable-height reasons are keyed by contact frame.
    ``contact_events`` retains the learned model's full stream, including contacts
    outside rallies. ``heuristic_candidates`` retains raw gate flags in heuristic mode.
    Review scores follow rally order and are empty in heuristic mode.
    """

    rallies: tuple[FinishedRally, ...]
    outcomes: dict[int, RallyOutcome]
    hit_heights: dict[int, int]
    hit_height_failures: dict[int, str]
    contact_events: tuple[ContactEvent, ...]
    heuristic_candidates: list[ContactCandidate]
    rally_confidence: tuple[float, ...]


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
