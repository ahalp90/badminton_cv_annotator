"""Fit rally selectors and their review ranker with downstream groups held out.

Contact predictions are supplied by the caller. Their inherited indirect paths
between training groups remain; only separate validation data measures quality.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from annotator.sequence.rally_layers import layer_features
from annotator.sequence.rally_model import (
    LayerReviewFeatures,
    RallyLayerModels,
    RallyLayerPrediction,
    build_layer_review_features,
    predict_rally_options,
    score_layers,
)
from annotator.sequence.rally_options import (
    RallyContext,
    RallyOptions,
    build_rally_options,
)
from annotator.training.rally_targets import (
    five_frame_preferred_targets,
    option_targets,
    original_option_rows,
)
from annotator.training.sequences import (
    SequenceTrainingVideo,
    whole_rally_correctness,
)
from annotator.training.trees import TreeFitConfig, fit_binary_tree


@dataclass(frozen=True)
class RallyFitConfig:
    """Selected development settings; deeper stacks remain available for comparisons."""

    tree: TreeFitConfig = field(default_factory=lambda: TreeFitConfig(
        max_iter=400, max_leaf_nodes=31,
    ))
    depth: int = 1
    max_edits: int = 3
    deletion_candidates: int = 6


@dataclass(frozen=True)
class RallyTrainingPool:
    identity: str
    group: str
    options: RallyOptions
    targets: np.ndarray


@dataclass(frozen=True)
class RallyModelFit:
    models: RallyLayerModels
    cross_fitted: dict[str, RallyLayerPrediction]
    span_weight: float


def judged_weights(pool: RallyOptions, targets: np.ndarray) -> np.ndarray:
    """Give each judged span total weight one, regardless of its option count."""
    weights = np.zeros(len(targets), dtype=np.float64)
    for rows in pool.span_rows.values():
        judged = rows[targets[rows] >= 0]
        if len(judged):
            weights[judged] = 1.0 / len(judged)
    return weights


def fit_layer(
    pools: Sequence[RallyOptions], targets: Sequence[np.ndarray], histories: Sequence[Sequence[np.ndarray]],
    config: TreeFitConfig, *, span_weight: float,
) -> Any:
    if not np.isfinite(span_weight) or span_weight <= 0:
        raise ValueError('span_weight must be finite and positive')
    matrices, labels, weights = [], [], []
    for pool, target, history in zip(pools, targets, histories, strict=True):
        judged = target >= 0
        matrices.append(layer_features(pool, history)[judged])
        labels.append(target[judged])
        weights.append(judged_weights(pool, target)[judged])
    if not labels:
        raise ValueError('supply at least one training pool')
    labelled = np.concatenate(labels)
    if set(labelled.tolist()) != {0, 1}:
        raise ValueError('layer training needs both correct and incorrect judged options')
    return fit_binary_tree(
        np.concatenate(matrices), labelled, sample_weight=np.concatenate(weights) * span_weight, balanced=False,
        max_iter=config.max_iter, max_leaf_nodes=config.max_leaf_nodes, learning_rate=config.learning_rate,
        l2_regularization=config.l2_regularization,
        random_seed=config.random_seed, device=config.device,
        min_child_weight=config.min_child_weight,
    )


@dataclass
class RallyTrainer:
    """Reuse every fitted group subset and held-group score history."""

    pools: dict[str, RallyTrainingPool]
    config: TreeFitConfig
    span_weight: float
    stacks: dict[tuple[tuple[str, ...], int], tuple[Any, ...]] = field(default_factory=dict)
    scores: dict[tuple[tuple[str, ...], int, str], tuple[np.ndarray, ...]] = field(default_factory=dict)

    def held_scores(self, groups: tuple[str, ...], depth: int, identity: str) -> tuple[np.ndarray, ...]:
        key = groups, depth, identity
        if key not in self.scores:
            self.scores[key] = score_layers(self.fit(groups, depth), self.pools[identity].options)
        return self.scores[key]

    def fit(self, groups: tuple[str, ...], depth: int) -> tuple[Any, ...]:
        if depth == 0:
            return ()
        if depth > len(groups):
            raise ValueError('recursive rally training needs at least one group per layer')
        key = groups, depth
        if key not in self.stacks:
            prefix = self.fit(groups, depth - 1)
            selected = [pool for pool in self.pools.values() if pool.group in groups]
            histories = [self.held_scores(
                tuple(group for group in groups if group != pool.group), depth - 1, pool.identity,
            ) for pool in selected]
            model = fit_layer(
                [pool.options for pool in selected], [pool.targets for pool in selected], histories,
                self.config, span_weight=self.span_weight,
            )
            self.stacks[key] = (*prefix, model)
        return self.stacks[key]


def prepare_training_pools(
    videos: Sequence[SequenceTrainingVideo], config: RallyFitConfig,
) -> tuple[dict[str, RallyTrainingPool], float]:
    pools = {}
    judged_original_rows = judged_original_spans = 0
    for video in videos:
        context = RallyContext(video.fps, video.initial, video.events, video.repairs, video.frame_count)
        pool = build_rally_options(context, config.max_edits, config.deletion_candidates)
        wide = option_targets(video, pool)
        five = option_targets(video, pool, tolerance_at_30_fps=5)
        original = original_option_rows(pool)
        included = original[wide[original] >= 0]
        judged_original_rows += len(included)
        judged_original_spans += len({pool.sequences[row].span_id for row in included})
        targets = five_frame_preferred_targets(pool, wide, five)
        pools[video.identity] = RallyTrainingPool(video.identity, video.group, pool, targets)
    if not judged_original_spans:
        raise ValueError('rally training needs judged original-option spans to anchor option weights')
    return pools, judged_original_rows / judged_original_spans


def fit_rally_models(videos: Sequence[SequenceTrainingVideo], config: RallyFitConfig) -> RallyModelFit:
    groups = tuple(sorted({video.group for video in videos}))
    if not 1 <= config.depth < len(groups):
        raise ValueError('rally training needs more groups than layers for held-group review predictions')
    if len({video.identity for video in videos}) != len(videos):
        raise ValueError('rally training video identities must be unique')
    pools, span_weight = prepare_training_pools(videos, config)
    trainer = RallyTrainer(pools, config.tree, span_weight)
    models = RallyLayerModels(trainer.fit(groups, config.depth), config.max_edits, config.deletion_candidates)
    predictions = {}
    for identity, pool in pools.items():
        others = tuple(group for group in groups if group != pool.group)
        held = RallyLayerModels(trainer.fit(others, config.depth), config.max_edits, config.deletion_candidates)
        predictions[identity] = predict_rally_options(held, pool.options)
    return RallyModelFit(models, predictions, span_weight)


def fit_review_model(features: LayerReviewFeatures, correctness: np.ndarray, config: TreeFitConfig) -> Any:
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
    videos: Sequence[SequenceTrainingVideo], predictions: dict[str, RallyLayerPrediction], config: TreeFitConfig,
) -> Any:
    matrices, correctness = [], []
    for video in videos:
        prediction = predictions[video.identity]
        features = build_layer_review_features(prediction)
        matrices.append(features.matrix)
        sequences = {sequence.span_id: sequence for sequence in prediction.sequences}
        correctness.extend(whole_rally_correctness(
            sequences[span_id], video.rallies, video.fps,
        ) for span_id in features.span_ids)
    matrix = np.concatenate(matrices)
    features = LayerReviewFeatures(tuple(range(len(matrix))), matrix)
    return fit_review_model(features, np.asarray(correctness, dtype=np.int8), config)
