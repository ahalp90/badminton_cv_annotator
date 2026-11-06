"""Score binary trees and select contacts from the contact model's predictions."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from annotator.config import frames_at_fps
from annotator.contacts.build_features import CONTACT_FEATURE_NAMES

NEARBY_CONTACT_DISTANCE_AT_30_FPS = 6


@dataclass(frozen=True)
class ContactModelConfig:
    """Prediction settings maintainers may retune for a new dataset."""

    score_cutoff: float = 0.9


DEFAULT_CONTACT_MODEL_CONFIG = ContactModelConfig()


class ScoredContacts(NamedTuple):
    """Candidate rows, their model inputs, probabilities and selected row indices."""

    candidates: np.ndarray
    feature_matrix: np.ndarray
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


def positive_probabilities(model: Any, matrix: np.ndarray) -> np.ndarray:
    """Score an annotator tree trained on labels 0 and 1, keeping its dtype."""
    if not len(matrix):
        return np.empty(0, dtype=np.float64)
    probabilities = np.asarray(model.predict_proba(matrix))[:, 1]
    if probabilities.shape != (len(matrix),) or not np.isfinite(probabilities).all():
        raise ValueError("model probabilities are incomplete or non-finite")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("model probabilities lie outside zero and one")
    return probabilities


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
    :param model: Fitted annotator tree with class-0/class-1 probability columns.
    :param fps: Source frame rate, used to scale nearby-contact suppression.
    :param config: Contact score cutoff.
    :return: Candidate rows, model inputs, probabilities and accepted indices.
    """
    matrix = contact_feature_matrix(rows)
    # Float32 comparison would round the cutoff and can accept scores just below it.
    probabilities = positive_probabilities(model, matrix).astype(np.float64, copy=False)
    distance = frames_at_fps(NEARBY_CONTACT_DISTANCE_AT_30_FPS, fps)
    kept = remove_nearby_contacts(
        rows["frame"], rows["interval_id"], probabilities, config.score_cutoff, distance,
    )
    return ScoredContacts(rows, matrix, probabilities, kept)
