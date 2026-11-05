"""Score the contact tree's fixed feature set and suppress nearby detections."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from annotator.config import frames_at_fps
from annotator.contacts.features import CONTACT_FEATURE_NAMES

NEARBY_CONTACT_DISTANCE_AT_30_FPS = 6


@dataclass(frozen=True)
class ContactModelConfig:
    """Prediction settings maintainers may retune for a new dataset."""

    score_cutoff: float = 0.9


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
        kept.extend(strongest_spaced_rows(possible, frames, scores, distance))
    return np.asarray(sorted(kept, key=lambda index: frames[index]), dtype=np.int32)


def strongest_spaced_rows(
    rows: Sequence[int] | np.ndarray, frames: np.ndarray, scores: np.ndarray, distance: int,
    *, limit: int | None = None, taken_frames: Sequence[int] = (),
) -> list[int]:
    """Keep strongest rows clear of earlier picks; earlier frames win score ties."""
    strongest_first = sorted(rows, key=lambda row: (-float(scores[row]), int(frames[row])))
    occupied = list(taken_frames)
    selected = []
    for row in strongest_first:
        frame = int(frames[row])
        if all(abs(frame - taken) > distance for taken in occupied):
            selected.append(int(row))
            occupied.append(frame)
            if limit is not None and len(selected) == limit:
                break
    return selected


def score_contact_features(
    rows: np.ndarray,
    model: Any,
    fps: float,
    config: ContactModelConfig = DEFAULT_CONTACT_MODEL_CONFIG,
) -> ScoredContacts:
    """Score candidate-region rows for one video and select contact frames.

    :param rows: All rows returned by ``build_contact_features`` for one video.
    :param model: Fitted binary classifier with ``predict_proba``.
    :param fps: Source frame rate, used to scale nearby-contact suppression.
    :param config: Contact score cutoff.
    :return: Candidate rows, probabilities and accepted candidate indices.
    """
    model_names = getattr(model, "feature_names_in_", None)
    if model_names is not None and tuple(model_names) != CONTACT_FEATURE_NAMES:
        raise ValueError("model feature names do not match the contact feature order")
    feature_count = getattr(model, "n_features_in_", None)
    if feature_count is not None and feature_count != len(CONTACT_FEATURE_NAMES):
        raise ValueError("model input count does not match the contact feature order")
    if not len(rows):
        return ScoredContacts(rows, np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int32))
    probabilities = np.asarray(model.predict_proba(contact_feature_matrix(rows))[:, 1], dtype=np.float64)
    if probabilities.shape != (len(rows),) or not np.isfinite(probabilities).all():
        raise ValueError("contact probabilities are incomplete or non-finite")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("contact probabilities lie outside zero and one")
    distance = frames_at_fps(NEARBY_CONTACT_DISTANCE_AT_30_FPS, fps)
    kept = remove_nearby_contacts(
        rows["frame"], rows["interval_id"], probabilities, config.score_cutoff, distance,
    )
    return ScoredContacts(rows, probabilities, kept)
