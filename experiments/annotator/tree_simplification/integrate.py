"""Fit and inspect review ranking for the current rally selector."""

import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from annotator.sequence.rally_model import RallyLayerModels, predict_rally_options
from annotator.sequence.rally_options import RallyOptions, build_rally_options
from annotator.training.rally_layers import fit_rally_review_model
from annotator.training.rally_targets import option_targets
from annotator.training.trees import TreeFitConfig
from dataset_builder.vision import load_npy_xz
from experiments.annotator.development import (
    dump_atomically,
    progress,
    xgboost_settings,
)
from experiments.annotator.heuristic_only_tune.run import read_json
from experiments.annotator.tree_simplification.options import LabelledOptions
from experiments.annotator.tree_simplification.targets import (
    five_frame_preferred_targets,
)
from experiments.annotator.tree_simplification.train import LayerTrainer


def verify_pool(actual: RallyOptions, expected: LabelledOptions) -> None:
    if actual.sequences != expected.sequences or tuple(actual.span_rows) != tuple(expected.span_rows):
        raise ValueError('Production proposal sequences or span order differ from the saved experiment')
    np.testing.assert_array_equal(actual.matrix, expected.matrix)
    for span_id, rows in actual.span_rows.items():
        np.testing.assert_array_equal(rows, expected.span_rows[span_id])


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
        five = load_npy_xz(targets_root / 'videos' / f'{pool.video.identity}.npy.xz')
        pools[pool.video.identity] = replace(pool, targets=five_frame_preferred_targets(pool, five))
    if len(pools) != 32:
        raise ValueError('Development export requires all 32 training videos')
    return pools


def fit_review(run: Path, depth: int, targets: Path, output: Path, settings: dict, *, device: str = 'cpu') -> Any:
    pools = load_training(settings, targets)
    sample = next(iter(pools))
    saved = joblib.load(Path(settings['pools']) / 'videos' / f'{sample}.joblib')
    rebuilt = build_rally_options(saved.runtime.context, settings['pool_settings']['max_edits'],
                                 settings['pool_settings']['deletion_candidates'])
    verify_pool(rebuilt, saved)
    if asdict(saved.boundary_policy) != settings['boundary_policy']:
        raise ValueError('Review fitting and selector targets use different boundary policies')
    np.testing.assert_array_equal(
        option_targets(saved.video, rebuilt, boundary_policy=saved.boundary_policy), saved.targets,
    )
    progress(output, 'training-pool-parity', video=sample, rows=len(saved.targets))
    # Reuse completed selector fits; new held-group review fits stay in this export.
    fits = output / 'review-fits'
    shutil.copytree(run / 'fits', fits / 'fits', dirs_exist_ok=True)
    selector_config = replace(TreeFitConfig(**xgboost_settings(settings['tree'])), device=device)
    trainer = LayerTrainer(pools, selector_config, fits, settings['span_weight'])
    groups = tuple(sorted({pool.video.group for pool in pools.values()}))
    predictions = {}
    for identity, pool in pools.items():
        held = tuple(group for group in groups if group != pool.video.group)
        models = RallyLayerModels(trainer.fit(held, depth))
        predictions[identity] = predict_rally_options(models, pool.runtime)
    config = replace(
        TreeFitConfig(**xgboost_settings(settings['pool_settings']['settings']['confidence'])), device=device,
    )
    review = fit_rally_review_model(
        [pool.video for pool in pools.values()], predictions, config, boundary_policy=saved.boundary_policy,
    )
    dump_atomically(review, output / 'review-model.joblib')
    return review
