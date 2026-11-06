"""Annotate a video from saved shuttle, player-pose and court arrays."""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd

from annotator.config import BaseAnnotatorConfig, resolve
from annotator.inference.heuristic import select_heuristic_contacts
from annotator.inference.learned import LearnedModelPrediction, predict_contacts
from annotator.inference.prepare_contact_inference import (
    AnnotationEvidence,
    prepare_contact_inference,
)
from annotator.models import DEFAULT_MODEL_DIRECTORY, AnnotatorModels, load_models
from annotator.rally_results import landing_and_winner
from annotator.rally_results.estimate_results import build_hit_heights, build_outcomes
from annotator.rally_results.landing_and_winner import (
    RallyOutcome,
    convert_landing_options,
)
from annotator.types import ContactCandidate, ContactEvent, FinishedRally
from shared.progress import report

DEFAULT_PREPROCESSING = BaseAnnotatorConfig()


def annotate_video(
    track: np.ndarray,
    bboxes: np.ndarray,
    scores: np.ndarray,
    kps: np.ndarray,
    ndet: np.ndarray,
    *,
    fps: float,
    landing_options: landing_and_winner.LandingFilterOptions,
    resolution: tuple[float, float],
    video_id: int | str,
    court_present: np.ndarray,
    homography_rows: pd.DataFrame | list[dict[str, object]],
    base: BaseAnnotatorConfig | None = None,
    models: AnnotatorModels | None = None,
    heuristic_only: bool = False,
    ref_err_px: float = 3.5,
    raw_exclusion_mask: np.ndarray | None = None,
    spans: list[tuple[int, int]] | None = None,
    contacts: dict[int, list[int]] | None = None,
    inpaint_codes: np.ndarray | None = None,
    rejection_diagnostics: list[dict[str, object]] | None = None,
    capture: RunCapture | None = None,
    court_invalid_is_excluded: bool = False,
) -> AnnotatorResult:
    """Find rallies and contacts, assign hitters, and estimate landings and winners.

    The inputs are frame-aligned vision arrays, not an RGB video to decode.
    The default route uses the supplied fitted models or loads the selected bundle.
    ``base`` must match the bundle's preprocessing. ``heuristic_only=True`` uses
    rule-based contacts with the same preparation and rally-result estimates.

    Core arrays are normalised shuttle ``track`` (frames, 3), pose ``bboxes``
    (frames, detections, 4), ``scores`` (frames, detections), ``kps`` (frames,
    detections, 17, 2) and ``ndet`` counts (frames,). ``homography_rows`` supplies
    court corners and frame intervals as records or a DataFrame.

    ``spans`` and ``contacts`` optionally supply initial rally bounds and contact
    frames. ``raw_exclusion_mask`` supplies excluded footage before short-run
    filtering; ``inpaint_codes`` supplies per-frame shuttle-hallucination grades.
    Frames outside accepted court geometry are excluded. Set
    ``court_invalid_is_excluded`` to also exclude frames without court presence.

    ``capture`` is reset at entry and retains prepared inputs and learned predictions
    after each stage succeeds. ``rejection_diagnostics`` receives rejection rows
    in processing-stage order rather than rally order.
    """
    if capture is not None:
        capture.evidence = None
        capture.learned_model_prediction = None
    if heuristic_only and models is not None:
        raise ValueError('heuristic_only cannot be combined with fitted models')
    if not heuristic_only:
        models = load_models(DEFAULT_MODEL_DIRECTORY) if models is None else models
        if base is not None and base != models.preprocessing:
            raise ValueError('Preprocessing differs from the fitted model bundle; use its settings or refit the models')
        base = models.preprocessing
    if base is None:
        base = DEFAULT_PREPROCESSING
    report("finding rallies")
    evidence = prepare_contact_inference(
        track, bboxes, scores, kps, ndet,
        resolved=resolve(base, fps), resolution=resolution, video_id=video_id,
        court_present=court_present, homography_rows=homography_rows, ref_err_px=ref_err_px,
        raw_exclusion_mask=raw_exclusion_mask,
        spans=spans, contacts=contacts, inpaint_codes=inpaint_codes,
        court_invalid_is_excluded=court_invalid_is_excluded,
    )
    if capture is not None:
        capture.evidence = evidence
    contact_events = ()
    rally_confidence = ()
    if models is None:
        rallies = select_heuristic_contacts(evidence, rejection_diagnostics)
        heuristic_candidates = evidence.heuristic_contacts
    else:
        prediction = predict_contacts(evidence, models)
        if capture is not None:
            capture.learned_model_prediction = prediction
        rallies = prediction.refined.rallies
        heuristic_candidates = []
        contact_events = prediction.refined.events
        rally_confidence = tuple(float(score) for score in prediction.confidence.scores)
    report("resolving outcomes")
    outcomes = build_outcomes(
        evidence, rallies, landing_options=convert_landing_options(landing_options, evidence.fps),
        rejection_diagnostics=rejection_diagnostics,
    )
    report("calculating hit heights")
    hit_heights, unavailable_hit_height_reasons = build_hit_heights(evidence, rallies)
    return AnnotatorResult(
        rallies=rallies, outcomes=outcomes, hit_heights=hit_heights,
        unavailable_hit_height_reasons=unavailable_hit_height_reasons,
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
    unavailable_hit_height_reasons: dict[int, str]
    contact_events: tuple[ContactEvent, ...]
    heuristic_candidates: list[ContactCandidate]
    rally_confidence: tuple[float, ...]


@dataclass
class RunCapture:
    """Prepared evidence and learned-model predictions retained for an optional caller."""

    evidence: AnnotationEvidence | None = None
    learned_model_prediction: LearnedModelPrediction | None = None
