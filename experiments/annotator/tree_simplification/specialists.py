"""Fit legacy serve/insertion specialists and expose their scores to rally layers."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from annotator.sequence.features import positive_probabilities, serve_edit_key
from annotator.training.sequences import (
    SequenceFitConfig,
    SequenceTargets,
    SequenceTrainingVideo,
    fit_tree,
    sequence_targets,
)
from experiments.annotator.tree_simplification.options import ExpandedPool, sequence_key

SPECIALIST_ROLES = frozenset(('serve', 'insertion'))


@dataclass(frozen=True)
class SpecialistModels:
    """Optional legacy serve and insertion models used as layer features."""

    serve: Any | None = None
    insertion: Any | None = None


@dataclass(frozen=True)
class SpecialisedLayer:
    """A rally layer together with the specialists used to build its inputs."""

    estimator: Any
    specialists: SpecialistModels


def fit_specialists(
    videos: Sequence[SequenceTrainingVideo], config: SequenceFitConfig, roles: tuple[str, ...], *,
    targets: Mapping[str, SequenceTargets] | None = None,
) -> SpecialistModels:
    """Fit requested legacy specialists on the same labels as the production stack."""
    unknown_roles = set(roles) - SPECIALIST_ROLES
    if unknown_roles:
        raise ValueError(f'unknown specialist roles: {sorted(unknown_roles)}')
    if len(set(roles)) != len(roles):
        raise ValueError('specialist roles must be unique')
    if not roles:
        return SpecialistModels()

    serve_rows = []
    serve_targets = []
    insertion_rows = []
    insertion_targets = []
    wants_serve = 'serve' in roles
    wants_insertion = 'insertion' in roles
    for video in videos:
        video_targets = sequence_targets(video) if targets is None else targets[video.identity]
        if wants_serve:
            serve_rows.append(video.pool.serve_inputs.matrix)
            serve_targets.append(video_targets.serve)
        if wants_insertion:
            added = video.pool.has_added_contact
            insertion_rows.append(video.pool.insertion_features[added])
            insertion_targets.append(video_targets.insertion[added])

    serve_model = None
    if wants_serve:
        serve_model = fit_tree(np.concatenate(serve_rows), np.concatenate(serve_targets), config.serve)
    insertion_model = None
    if wants_insertion:
        insertion_model = fit_tree(
            np.concatenate(insertion_rows), np.concatenate(insertion_targets), config.sequence,
        )
    return SpecialistModels(serve=serve_model, insertion=insertion_model)


def _merge_finite_max(scores: dict[Any, float], key: Any, score: float) -> None:
    if np.isfinite(score):
        scores[key] = max(scores.get(key, -np.inf), score)


def _serve_scores_by_sequence(pool: ExpandedPool, model: Any) -> dict[tuple[int, tuple[int, int, tuple[int, ...]]], float]:
    legacy = pool.original.pool
    row_scores = positive_probabilities(model, legacy.serve_inputs.matrix)
    scores_by_key = {key: row_scores[row] for key, row in legacy.serve_inputs.rows_by_key.items()}
    scores_by_sequence: dict[tuple[int, tuple[int, int, tuple[int, ...]]], float] = {}
    for option in legacy.options:
        key = serve_edit_key(option)
        if key is not None:
            _merge_finite_max(
                scores_by_sequence,
                (option.span_id, sequence_key(option.sequence)),
                float(scores_by_key[key]),
            )
    return scores_by_sequence


def _insertion_scores_by_sequence(
    pool: ExpandedPool, model: Any,
) -> dict[tuple[int, tuple[int, int, tuple[int, ...]]], float]:
    legacy = pool.original.pool
    added = legacy.has_added_contact
    row_scores = np.full(len(legacy.options), np.nan, dtype=np.float64)
    row_scores[added] = positive_probabilities(model, legacy.insertion_features[added])
    scores_by_sequence: dict[tuple[int, tuple[int, int, tuple[int, ...]]], float] = {}
    for option, score in zip(legacy.options, row_scores, strict=True):
        if option.added_contact is not None:
            _merge_finite_max(
                scores_by_sequence,
                (option.span_id, sequence_key(option.sequence)),
                float(score),
            )
    return scores_by_sequence


def _expanded_scores(
    pool: ExpandedPool,
    scores_by_sequence: dict[tuple[int, tuple[int, int, tuple[int, ...]]], float],
) -> np.ndarray:
    return np.asarray([
        scores_by_sequence.get((sequence.span_id, sequence_key(sequence)), np.nan)
        for sequence in pool.sequences
    ], dtype=np.float32)


def _span_best(pool: ExpandedPool, scores: np.ndarray) -> np.ndarray:
    best = np.full(len(pool.sequences), np.nan, dtype=np.float32)
    for rows in pool.span_rows.values():
        finite = scores[rows][np.isfinite(scores[rows])]
        if len(finite):
            best[rows] = np.max(finite)
    return best


def augment_pool(pool: ExpandedPool, models: SpecialistModels) -> ExpandedPool:
    """Append legacy specialist scores without consulting expanded-pool labels."""
    empty = {}
    serve_by_sequence = empty if models.serve is None else _serve_scores_by_sequence(pool, models.serve)
    insertion_by_sequence = empty if models.insertion is None else _insertion_scores_by_sequence(
        pool, models.insertion,
    )
    serve = _expanded_scores(pool, serve_by_sequence)
    insertion = _expanded_scores(pool, insertion_by_sequence)
    specialist_columns = np.column_stack((serve, insertion, _span_best(pool, serve), _span_best(pool, insertion)))
    return replace(pool, matrix=np.column_stack((pool.matrix, specialist_columns)))
