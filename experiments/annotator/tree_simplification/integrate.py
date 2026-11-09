"""Fit and inspect review ranking for the current rally selector."""

import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from annotator.rallies.refine_rallies import RallySelector, predict_rally_options
from annotator.training.rally import fit_rally_review_model
from annotator.training.trees import TreeFitConfig
from dataset_builder.vision import load_npy_xz
from experiments.annotator.development import (
    dump_atomically,
)
from experiments.annotator.heuristic_only_tune.run import read_json
from experiments.annotator.tree_simplification.options import LabelledOptions
from experiments.annotator.tree_simplification.targets import (
    preferred_targets,
)
from experiments.annotator.tree_simplification.train import SelectorTrainer


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    judged = labels >= 0
    correct, confidence = labels[judged], scores[judged]
    wrong = 1 - correct
    result: dict[str, Any] = {'judged': len(correct), 'unjudgeable': int((~judged).sum()), 'wrong': int(wrong.sum())}
    result['correct_auc'] = float(roc_auc_score(correct, confidence)) if len(np.unique(correct)) == 2 else None
    result['wrong_average_precision'] = float(average_precision_score(wrong, 1 - confidence)) if wrong.any() else None
    ordered = np.argsort(confidence, kind='stable')
    for fraction in (.1, .25, .5):
        budget = int(np.ceil(len(ordered) * fraction))
        result[f'wrong_in_first_{fraction:g}'] = int(wrong[ordered[:budget]].sum())
    return result


def load_training(settings: dict, targets_root: Path) -> dict[str, LabelledOptions]:
    root = Path(settings['pools'])
    if read_json(root / 'pools.json.gz') != settings['pool_settings']:
        raise ValueError('Saved option-pool manifest differs from the fitted run')
    target_manifest = read_json(targets_root / 'targets.json.gz')
    if settings['target_policy']['manifest'] != target_manifest:
        raise ValueError('Five-frame target manifest differs from the fitted run')
    pools = {}
    for path in sorted((root / 'videos').glob('*.joblib')):
        if read_json(path.with_suffix('.json.gz'))['split'] != 'train':
            continue
        pool = joblib.load(path)
        five = load_npy_xz(targets_root / 'videos' / f'{pool.video.video_id}.npy.xz')
        pools[pool.video.video_id] = replace(pool, targets=preferred_targets(pool, five))
    if len(pools) != 32:
        raise ValueError('Development export requires all 32 training videos')
    return pools


def fit_review(
    run: Path, targets: Path, output: Path, settings: dict, *,
    device: str = 'cpu',
) -> Any:
    pools = load_training(settings, targets)
    # Reuse completed selector fits; new held-group review fits stay in this export.
    fits = output / 'review-fits'
    shutil.copytree(run / 'fits', fits / 'fits', dirs_exist_ok=True)
    selector_config = replace(TreeFitConfig(**settings['tree']), device=device)
    trainer = SelectorTrainer(pools, selector_config, fits, settings['span_weight'])
    groups = tuple(sorted({pool.video.group for pool in pools.values()}))
    predictions = {}
    for group in groups:
        held = tuple(other for other in groups if other != group)
        models = RallySelector(trainer.fit(held))
        for identity, pool in pools.items():
            if pool.video.group == group:
                predictions[identity] = predict_rally_options(models, pool.runtime)
    config = replace(
        TreeFitConfig(**settings['pool_settings']['settings']['confidence']), device=device,
    )
    review = fit_rally_review_model(
        [pool.video for pool in pools.values()], predictions, config,
    )
    dump_atomically(review, output / 'review-model.joblib')
    return review
