"""Fit the rally selector, then the model that ranks rallies for review.

For review training, predict each video's rallies with a selector fitted
without its group. Contact predictions are supplied by the caller and
reused across these selector fits.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from annotator.rallies.build_contact_proposals import RallyOptions, build_rally_options
from annotator.rallies.refine_rallies import (
    RallyPrediction,
    RallySelector,
    build_review_features,
    predict_rally_options,
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


@dataclass(frozen=True)
class RallyModelFit:
    """Deployed selector and predictions needed to train the review model.

    ``held_group_predictions`` is keyed by video ID. Each prediction uses a
    selector fitted without that video's group. The deployed selector uses
    all supplied training videos.
    """
    selector: RallySelector
    held_group_predictions: dict[str, RallyPrediction]


def fit_rally_models(videos: Sequence[SequenceTrainingVideo], config: RallyFitConfig) -> RallyModelFit:
    """Fit the deployed selector and predict withheld groups for review training.

    Use all supplied videos for the deployed model. Then leave out one group
    at a time, fit a selector and predict that group's videos.
    """
    groups = tuple(sorted({video.group for video in videos}))
    if len(groups) < 2:
        raise ValueError('rally training needs at least two groups for held-group review predictions')
    if len({video.video_id for video in videos}) != len(videos):
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
        for video_id, pool in pools.items():
            if pool.group == held_group:
                held_group_predictions[video_id] = predict_rally_options(held_model, pool.options)
    return RallyModelFit(selector, held_group_predictions)


def fit_rally_review_model(
    videos: Sequence[SequenceTrainingVideo], predictions: dict[str, RallyPrediction], config: TreeFitConfig,
    *, tolerance_at_30_fps: int = 10,
) -> Any:
    """Fit review scores from predictions made while withholding each video's group.

    Judge the finished rallies against contact/player labels and use the same
    review features as inference.
    """
    matrices, correctness = [], []
    for video in videos:
        prediction = predictions[video.video_id]
        features = build_review_features(prediction)
        matrices.append(features.matrix)
        sequences = {sequence.span_id: sequence for sequence in prediction.sequences}
        correctness.extend(whole_rally_correctness(
            sequences[span_id], video.rallies, video.context.fps, tolerance_at_30_fps=tolerance_at_30_fps,
        ) for span_id in features.span_ids)
    matrix = np.concatenate(matrices)
    return fit_review_model(matrix, np.asarray(correctness, dtype=np.int8), config)


def prepare_training_pools(
    videos: Sequence[SequenceTrainingVideo], config: RallyFitConfig,
) -> dict[str, RallyTrainingPool]:
    """Build each video's proposals and judge them using the configured tolerances."""
    pools = {}
    for video in videos:
        pool = build_rally_options(video.context, config.max_edits, config.deletion_candidates)
        wide, preferred = option_targets_at_tolerances(
            video, pool, (config.fallback_tolerance_at_30_fps, config.preferred_tolerance_at_30_fps),
        )
        targets = preferred_targets(pool, wide, preferred)
        pools[video.video_id] = RallyTrainingPool(video.video_id, video.group, pool, targets)
    return pools


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
