"""Run the contact classifier and retain strong, sufficiently separated hit candidates."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from annotator.contacts.build_features import CONTACT_FEATURE_NAMES
from shared.frame_timing import frames_at_fps

NEARBY_CONTACT_DISTANCE_AT_30_FPS = 6


@dataclass(frozen=True)
class ContactModelConfig:
    """Probability cutoff for retaining contact candidates."""

    score_cutoff: float = 0.9


DEFAULT_CONTACT_MODEL_CONFIG = ContactModelConfig()


class ScoredContacts(NamedTuple):
    """All scored candidates, including possible additions during rally refinement.

    ``candidates``, ``feature_matrix`` and ``probabilities`` share row indices.
    ``kept_indices`` selects the initial contacts after probability and spacing
    checks. Rejected rows remain available for recovering missed contacts.
    """

    candidates: np.ndarray
    feature_matrix: np.ndarray
    probabilities: np.ndarray
    kept_indices: np.ndarray

    @property
    def frames(self) -> np.ndarray:
        """Return selected contact frames in chronological order."""
        return self.candidates["frame"][self.kept_indices]


def score_contact_features(
    rows: np.ndarray,
    model: Any,
    fps: float,
    config: ContactModelConfig = DEFAULT_CONTACT_MODEL_CONFIG,
) -> ScoredContacts:
    """Score every candidate, then retain strong, sufficiently separated contacts.

    :param rows: One video's structured rows from ``build_contact_features``.
    :param model: Fitted classifier; probability columns are classes 0 and 1.
    :param fps: Video frame rate, used to scale the minimum contact spacing.
    :param config: Minimum probability to retain a contact.
    :return: All rows and scores, plus indices of the selected contacts.
    """
    matrix = contact_feature_matrix(rows)
    # Float32 comparison would round the cutoff and can accept scores just below it.
    probabilities = positive_probabilities(model, matrix).astype(np.float64, copy=False)
    distance = frames_at_fps(NEARBY_CONTACT_DISTANCE_AT_30_FPS, fps)
    kept = remove_nearby_contacts(
        rows["frame"], rows["interval_id"], probabilities, config.score_cutoff, distance,
    )
    return ScoredContacts(rows, matrix, probabilities, kept)


def contact_feature_matrix(
    rows: np.ndarray, names: Sequence[str] = CONTACT_FEATURE_NAMES,
) -> np.ndarray:
    """Return float32 model inputs in the specified order, preserving NaNs."""
    if rows.dtype.names is None or any(name not in rows.dtype.names for name in names):
        raise ValueError("contact feature rows lack a model input field")
    return np.column_stack([rows[name].astype(np.float32, copy=False) for name in names])


def positive_probabilities(model: Any, matrix: np.ndarray) -> np.ndarray:
    """Return class-1 probabilities from a classifier fitted on labels 0 and 1."""
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
    """Keep the strongest contacts in each search interval.

    Consider candidates at or above cutoff, highest score first; earlier frames
    win ties. Skip a candidate within ``distance`` frames of an accepted contact
    in the same interval, including exactly that distance.
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
    """Select highest-scoring rows first, keeping contacts more than distance frames apart.

    Earlier frames win ties. Also keep that distance from ``taken_frames``.
    Return selected row indices in selection order; stop at ``limit`` if supplied.
    """
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
