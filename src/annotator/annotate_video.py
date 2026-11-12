"""Annotate a video from saved shuttle, player-pose and court arrays."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from annotator.config import BaseAnnotatorConfig, scale_config_to_fps
from annotator.inference.heuristic import select_heuristic_contacts
from annotator.inference.learned import (
    LearnedModelPrediction,
    predict_rallies_and_contacts,
)
from annotator.inference.prepare_annotation_inputs import (
    PreparedAnnotationInputs,
    prepare_annotation_inputs,
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
    court_present: np.ndarray,
    homography_rows: pd.DataFrame | list[dict[str, object]],
    base: BaseAnnotatorConfig | None = None,
    models: AnnotatorModels | None = None,
    heuristic_only: bool = False,
    ref_err_px: float = 3.5,
    raw_exclusion_mask: np.ndarray | None = None,
    spans: list[tuple[int, int]] | None = None,
    contacts: dict[int, list[int]] | None = None,
    shuttle_grades: np.ndarray | None = None,
    keep_working_data: bool = False,
) -> VideoAnnotation:
    """Find rallies and contacts, assign hitters, then estimate landings and winners.

    Takes saved vision arrays with matching frame indices. The learned route
    uses ``models``, or loads the default bundle. Its saved preprocessing
    settings apply; an explicit ``base`` must match. ``heuristic_only=True``
    uses rule-based contacts with the same preparation and outcome estimates.

    Array shapes: normalised shuttle ``track`` (frames, 3), pose ``bboxes``
    (frames, detections, 4), ``scores`` (frames, detections), ``kps`` (frames,
    detections, 17, 2), and ``ndet`` (frames,). ``homography_rows`` gives court
    corners and frame ranges as records or a DataFrame.

    ``spans`` and ``contacts`` can supply initial bounds and contact frames.
    ``raw_exclusion_mask`` supplies footage flags before short-run filtering;
    ``shuttle_grades`` supplies shuttle-quality grades. Frames without accepted
    court geometry or court presence are excluded.

    Set ``keep_working_data=True`` to retain prepared inputs and model scores
    for debugging or experiments. Exclusion masks are always returned for
    dataset saving and commentary.
    """
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
    prepared_inputs = prepare_annotation_inputs(
        track, bboxes, scores, kps, ndet,
        video_config=scale_config_to_fps(base, fps), resolution=resolution,
        court_present=court_present, homography_rows=homography_rows, ref_err_px=ref_err_px,
        raw_exclusion_mask=raw_exclusion_mask,
        spans=spans, contacts=contacts, shuttle_grades=shuttle_grades,
    )
    prediction = None
    contact_events = ()
    rally_confidence = ()
    if models is None:
        rallies = select_heuristic_contacts(prepared_inputs)
        heuristic_candidates = prepared_inputs.heuristic_contacts
    else:
        prediction = predict_rallies_and_contacts(prepared_inputs, models)
        rallies = prediction.rallies
        heuristic_candidates = []
        contact_events = prediction.events
        rally_confidence = prediction.rally_confidence
    report("resolving outcomes")
    outcomes = build_outcomes(
        prepared_inputs, rallies, landing_options=convert_landing_options(landing_options, prepared_inputs.fps),
    )
    report("calculating hit heights")
    hit_heights, unavailable_hit_height_reasons = build_hit_heights(prepared_inputs, rallies)
    result = AnnotatorResult(
        rallies=rallies, outcomes=outcomes, hit_heights=hit_heights,
        unavailable_hit_height_reasons=unavailable_hit_height_reasons,
        contact_events=contact_events, heuristic_candidates=heuristic_candidates, rally_confidence=rally_confidence,
    )
    working_data = AnnotationWorkingData(prepared_inputs, prediction) if keep_working_data else None
    return VideoAnnotation(result, prepared_inputs.raw_exclusion_mask, prepared_inputs.exclusion_mask, working_data)


class AnnotatorResult(NamedTuple):
    """Finished annotations for one video.

    Rally IDs index ``rallies``. Each rally includes bounds, assigned contact
    sides and its final hitter. ``outcomes`` includes rallies whose final
    hitter is known. Heights and missing-height reasons are keyed by contact
    frame.

    ``contact_events`` is the full learned contact list, including contacts
    outside rallies. ``heuristic_candidates`` includes the wrist/spacing check
    results in heuristic mode. ``rally_confidence`` follows rally order and
    is empty in heuristic mode.
    """

    rallies: tuple[FinishedRally, ...]
    outcomes: dict[int, RallyOutcome]
    hit_heights: dict[int, int]
    unavailable_hit_height_reasons: dict[int, str]
    contact_events: tuple[ContactEvent, ...]
    heuristic_candidates: list[ContactCandidate]
    rally_confidence: tuple[float, ...]


class VideoAnnotation(NamedTuple):
    """Annotations and masks needed for saving; optionally retain intermediate data."""

    result: AnnotatorResult
    raw_exclusion_mask: np.ndarray
    exclusion_mask: np.ndarray
    working_data: AnnotationWorkingData | None


class AnnotationWorkingData(NamedTuple):
    """Prepared inputs and model scores for debugging and experiments."""

    prepared_inputs: PreparedAnnotationInputs
    learned_model_prediction: LearnedModelPrediction | None
