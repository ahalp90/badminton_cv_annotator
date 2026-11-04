"""Fit rally selectors and their review ranker.

Review training uses predictions from selectors fitted without each video's group.
These selector fits reuse the contact predictions supplied by the caller.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from annotator.sequence.rally_model import (
    RallyPrediction,
    RallyReviewFeatures,
    RallySelector,
    build_review_features,
    predict_rally_options,
)
from annotator.sequence.rally_options import (
    RallyContext,
    RallyOptions,
    build_rally_options,
)
from annotator.training.rally_targets import (
    option_targets_at_tolerances,
    preferred_targets,
)
from annotator.training.sequences import (
    SequenceTrainingVideo,
    whole_rally_correctness,
)
from annotator.training.trees import TreeFitConfig, fit_binary_tree


@dataclass(frozen=True)
class RallyFitConfig:
    """Selected development settings for one rally selector.

    ``span_weight`` is each judged rally's total weight, shared by its options.
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
    identity: str
    group: str
    options: RallyOptions
    targets: np.ndarray


@dataclass(frozen=True)
class RallyModelFit:
    selector: RallySelector
    held_group_predictions: dict[str, RallyPrediction]


def judged_weights(pool: RallyOptions, targets: np.ndarray) -> np.ndarray:
    """Give each judged span total weight one, regardless of its option count."""
    weights = np.zeros(len(targets), dtype=np.float64)
    for rows in pool.span_rows.values():
        judged = rows[targets[rows] >= 0]
        if len(judged):
            weights[judged] = 1.0 / len(judged)
    return weights


def fit_selector(
    pools: Sequence[RallyOptions], targets: Sequence[np.ndarray],
    config: TreeFitConfig, *, span_weight: float,
) -> Any:
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
        np.concatenate(matrices), labelled, sample_weight=np.concatenate(weights) * span_weight, balanced=False,
        max_iter=config.max_iter, max_leaf_nodes=config.max_leaf_nodes, learning_rate=config.learning_rate,
        l2_regularization=config.l2_regularization,
        random_seed=config.random_seed, device=config.device,
        min_child_weight=config.min_child_weight,
    )


def prepare_training_pools(
    videos: Sequence[SequenceTrainingVideo], config: RallyFitConfig,
) -> dict[str, RallyTrainingPool]:
    pools = {}
    for video in videos:
        context = RallyContext(video.fps, video.initial, video.events, video.candidates, video.frame_count, video.bounds_supplied)
        pool = build_rally_options(context, config.max_edits, config.deletion_candidates)
        wide, preferred = option_targets_at_tolerances(
            video, pool, (config.fallback_tolerance_at_30_fps, config.preferred_tolerance_at_30_fps),
        )
        targets = preferred_targets(pool, wide, preferred)
        pools[video.identity] = RallyTrainingPool(video.identity, video.group, pool, targets)
    return pools


def fit_rally_models(videos: Sequence[SequenceTrainingVideo], config: RallyFitConfig) -> RallyModelFit:
    """Fit the deployed selector on all videos and predictions for review training.

    Each group's review-training predictions use a selector fitted without that group.
    """
    groups = tuple(sorted({video.group for video in videos}))
    if len(groups) < 2:
        raise ValueError('rally training needs at least two groups for held-group review predictions')
    if len({video.identity for video in videos}) != len(videos):
        raise ValueError('rally training video identities must be unique')
    pools = prepare_training_pools(videos, config)

    def fit(selected: list[RallyTrainingPool]) -> RallySelector:
        selector = fit_selector(
            [pool.options for pool in selected], [pool.targets for pool in selected],
            config.tree, span_weight=config.span_weight,
        )
        return RallySelector(selector, config.max_edits, config.deletion_candidates)

    selector = fit(list(pools.values()))
    held_group_predictions = {}
    for held_group in groups:
        held_model = fit([pool for pool in pools.values() if pool.group != held_group])
        for identity, pool in pools.items():
            if pool.group == held_group:
                held_group_predictions[identity] = predict_rally_options(held_model, pool.options)
    return RallyModelFit(selector, held_group_predictions)


def fit_review_model(features: RallyReviewFeatures, correctness: np.ndarray, config: TreeFitConfig) -> Any:
    labels = np.asarray(correctness)
    if labels.shape != (len(features.span_ids),) or not np.isin(labels, (-1, 0, 1)).all():
        raise ValueError('review labels must be -1, 0 or 1 and align with feature rows')
    judged = labels >= 0
    if set(labels[judged].tolist()) != {0, 1}:
        raise ValueError('review training needs both correct and wrong judged sections')
    return fit_binary_tree(
        features.matrix[judged], labels[judged], balanced=False,
        max_iter=config.max_iter, max_leaf_nodes=config.max_leaf_nodes, learning_rate=config.learning_rate,
        l2_regularization=config.l2_regularization,
        random_seed=config.random_seed, device=config.device,
        min_child_weight=config.min_child_weight,
    )


def fit_rally_review_model(
    videos: Sequence[SequenceTrainingVideo], predictions: dict[str, RallyPrediction], config: TreeFitConfig,
    *, tolerance_at_30_fps: int = 10,
) -> Any:
    matrices, correctness = [], []
    for video in videos:
        prediction = predictions[video.identity]
        features = build_review_features(prediction)
        matrices.append(features.matrix)
        sequences = {sequence.span_id: sequence for sequence in prediction.sequences}
        correctness.extend(whole_rally_correctness(
            sequences[span_id], video.rallies, video.fps, tolerance_at_30_fps=tolerance_at_30_fps,
        ) for span_id in features.span_ids)
    matrix = np.concatenate(matrices)
    features = RallyReviewFeatures(tuple(range(len(matrix))), matrix)
    return fit_review_model(features, np.asarray(correctness, dtype=np.int8), config)
