"""Label rally options and fit the shared selector and confidence classifiers."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from annotator.rallies.build_contact_proposals import RallyOptions
from annotator.training.rally_targets import (
    option_targets_at_tolerances,
    preferred_targets,
)
from annotator.training.sequences import SequenceTrainingVideo
from annotator.training.trees import TreeFitConfig, fit_binary_tree


@dataclass(frozen=True)
class RallyFitConfig:
    """Rally-selector tree, proposal limits and label-matching distances.

    ``span_weight`` is the total weight for each judged rally, shared by its
    alternatives. Prefer the tighter matching tolerance when at least one
    alternative remains correct; otherwise use the fallback tolerance.
    """

    tree: TreeFitConfig = field(default_factory=lambda: TreeFitConfig(
        max_iter=400, max_leaf_nodes=31,
    ))
    max_edits: int = 3
    deletion_candidates: int = 6
    span_weight: float = 372.0
    preferred_tolerance_at_30_fps: int = 5
    fallback_tolerance_at_30_fps: int = 10
    review_tolerance_at_30_fps: int = 10


@dataclass(frozen=True)
class RallyTrainingPool:
    """One video's proposed contact lists and their judged correctness labels.

    ``targets`` aligns with option rows: 1 correct, 0 wrong, -1 unjudgeable.
    ``group`` identifies videos to withhold together during review training.
    """
    video_id: str
    group: str
    options: RallyOptions
    targets: np.ndarray


def label_rally_options(video: SequenceTrainingVideo, pool: RallyOptions, config: RallyFitConfig) -> RallyTrainingPool:
    """Apply the same correctness targets to ordinary rallies and split pieces."""
    wide, preferred = option_targets_at_tolerances(
        video, pool, (config.fallback_tolerance_at_30_fps, config.preferred_tolerance_at_30_fps),
    )
    targets = preferred_targets(pool, wide, preferred)
    return RallyTrainingPool(video.video_id, video.group, pool, targets)


def fit_selector(
    pools: Sequence[RallyOptions], targets: Sequence[np.ndarray],
    config: TreeFitConfig, *, span_weight: float,
) -> Any:
    """Fit judged proposal rows, giving each rally the same total weight.

    Exclude targets of -1. Split span_weight across the remaining alternatives
    for each rally so rallies with more proposals do not dominate training.
    """
    if not np.isfinite(span_weight) or span_weight <= 0:
        raise ValueError('span_weight must be finite and positive')
    matrices, labels, weights = [], [], []
    for pool, target in zip(pools, targets, strict=True):
        judged = target >= 0
        matrices.append(pool.matrix[judged])
        labels.append(target[judged])
        weights.append(judged_weights(pool, target)[judged])
    if not labels:
        raise ValueError('supply at least one training pool')
    labelled = np.concatenate(labels)
    if set(labelled.tolist()) != {0, 1}:
        raise ValueError('selector training needs both correct and incorrect judged options')
    return fit_binary_tree(
        np.concatenate(matrices), labelled, config,
        sample_weight=np.concatenate(weights) * span_weight, balanced=False,
    )


def judged_weights(pool: RallyOptions, targets: np.ndarray) -> np.ndarray:
    """Divide weight 1 equally among each rally's judged alternatives."""
    weights = np.zeros(len(targets), dtype=np.float64)
    for rows in pool.span_rows.values():
        judged = rows[targets[rows] >= 0]
        if len(judged):
            weights[judged] = 1.0 / len(judged)
    return weights


def fit_review_model(matrix: np.ndarray, correctness: np.ndarray, config: TreeFitConfig) -> Any:
    """Fit rally correctness from judged rows, excluding labels of -1."""
    labels = np.asarray(correctness)
    judged = labels >= 0
    if set(labels[judged].tolist()) != {0, 1}:
        raise ValueError('review training needs both correct and wrong judged sections')
    return fit_binary_tree(
        matrix[judged], labels[judged], config, balanced=False,
    )
