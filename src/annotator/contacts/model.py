"""Score the contact tree's fixed feature set and suppress nearby detections."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from annotator.contacts.features import (
    BASE_MISSINGNESS_SIGNALS,
    BASE_PHYSICS_SIGNALS,
    REGION_FIELDS,
    WINDOW_OFFSETS_BASE30,
)
from annotator.fps_constants import ScalingKind

CONTACT_FEATURE_NAMES = tuple(
    f"{signal}_t{offset:+d}"
    for signal in BASE_PHYSICS_SIGNALS + BASE_MISSINGNESS_SIGNALS
    for offset in WINDOW_OFFSETS_BASE30
)


NEARBY_CONTACT_DISTANCE_AT_30_FPS = 6


@dataclass(frozen=True)
class ContactModelConfig:
    """Prediction settings maintainers may retune for a new dataset."""

    score_cutoff: float = 0.9
    reject_masked_without_player: bool = False


DEFAULT_CONTACT_MODEL_CONFIG = ContactModelConfig()


class ScoredContacts(NamedTuple):
    """Candidate rows, their probabilities and selected row indices."""

    candidates: np.ndarray
    probabilities: np.ndarray
    kept_indices: np.ndarray

    @property
    def frames(self) -> np.ndarray:
        """Return selected contact frames in chronological order."""
        return self.candidates["frame"][self.kept_indices]


def contact_feature_matrix(
    rows: np.ndarray, names: Sequence[str] = CONTACT_FEATURE_NAMES,
) -> np.ndarray:
    """Return float32 model inputs in the specified order, preserving NaNs."""
    if rows.dtype.names is None or any(name not in rows.dtype.names for name in names):
        raise ValueError("contact feature rows lack a model input field")
    return np.column_stack([rows[name].astype(np.float32, copy=False) for name in names])


def remove_nearby_contacts(
    frames: np.ndarray,
    interval_ids: np.ndarray,
    scores: np.ndarray,
    cutoff: float,
    distance: int,
) -> np.ndarray:
    """Select strongest contacts per interval; earlier frames win score ties.

    Frames exactly ``distance`` apart compete. Greedy suppression compares
    each remaining candidate against accepted contacts, rather than merging
    chains of nearby candidates into a single group.
    """
    kept: list[int] = []
    for interval_id in np.unique(interval_ids):
        possible = np.flatnonzero((interval_ids == interval_id) & (scores >= cutoff))
        strongest_first = sorted(possible, key=lambda index: (-scores[index], frames[index]))
        interval_kept: list[int] = []
        for index in strongest_first:
            if all(abs(int(frames[index]) - int(frames[other])) > distance for other in interval_kept):
                interval_kept.append(int(index))
        kept.extend(interval_kept)
    return np.asarray(sorted(kept, key=lambda index: frames[index]), dtype=np.int32)


def score_contact_features(
    rows: np.ndarray,
    model: Any,
    fps: float,
    config: ContactModelConfig = DEFAULT_CONTACT_MODEL_CONFIG,
    *,
    shuttle_hallucination_mask: np.ndarray | None = None,
) -> ScoredContacts:
    """Score candidate-region rows for one video and select contact frames.

    :param rows: All rows returned by ``build_contact_features`` for one video.
    :param model: Fitted binary classifier with ``predict_proba``.
    :param fps: Source frame rate, used to scale nearby-contact suppression.
    :param config: Contact score cutoff and optional candidate rejection policy.
    :param shuttle_hallucination_mask: Frame-aligned boolean mask already derived
        from the fitted preprocessing's rejected shuttle grades.
    :return: Candidate rows, probabilities and accepted candidate indices.
    """
    model_names = getattr(model, "feature_names_in_", None)
    if model_names is not None and tuple(model_names) != CONTACT_FEATURE_NAMES:
        raise ValueError("model feature names do not match the contact feature order")
    feature_count = getattr(model, "n_features_in_", None)
    if feature_count is not None and feature_count != len(CONTACT_FEATURE_NAMES):
        raise ValueError("model input count does not match the contact feature order")
    selected = np.zeros(len(rows), dtype=bool)
    for name in REGION_FIELDS:
        selected |= rows[name].astype(bool)
    candidates = rows[selected]
    if config.reject_masked_without_player:
        if shuttle_hallucination_mask is None:
            raise ValueError('masked candidate rejection requires the shuttle hallucination mask')
        has_player = np.zeros(len(candidates), dtype=bool)
        # These features already sample the FPS-scaled offsets within each search
        # interval. NaNs outside the interval do not count as a nominated player.
        for side in ('top', 'bot'):
            for offset in WINDOW_OFFSETS_BASE30:
                has_player |= candidates[f'pose_valid_{side}_t{offset:+d}'] > 0.0
        candidates = candidates[~shuttle_hallucination_mask[candidates['frame']] | has_player]
    if not len(candidates):
        return ScoredContacts(candidates, np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int32))
    probabilities = np.asarray(model.predict_proba(contact_feature_matrix(candidates))[:, 1], dtype=np.float64)
    if probabilities.shape != (len(candidates),) or not np.isfinite(probabilities).all():
        raise ValueError("contact probabilities are incomplete or non-finite")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("contact probabilities lie outside zero and one")
    distance = int(ScalingKind.FRAME_COUNT.scale(NEARBY_CONTACT_DISTANCE_AT_30_FPS, fps))
    kept = remove_nearby_contacts(
        candidates["frame"], candidates["interval_id"], probabilities, config.score_cutoff, distance,
    )
    return ScoredContacts(candidates, probabilities, kept)
