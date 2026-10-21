"""Export a completed rally stack and verify production replay and review ranking."""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(REPOSITORY := Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(REPOSITORY / 'src'))

from experiments.annotator.heuristic_only_tune.run import (
    read_json,
    source_record,
    write_json,
)

# isort: split

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from annotator.hybrid import predict_contact_features
from annotator.models import AnnotatorModels, fitted_trees, load_models, save_models
from annotator.sequence.confidence import score_rally_confidence
from annotator.sequence.rally_model import (
    RallyLayerModels,
    RallyLayerPrediction,
    predict_rally_options,
    score_layer_confidence,
)
from annotator.sequence.rally_options import RallyOptions, build_rally_options
from annotator.sequence.refine import SequenceModels, finish_sequences
from annotator.training.confidence import ConfidenceFitConfig
from annotator.training.rally_layers import fit_rally_review_model
from annotator.training.rally_targets import option_targets
from annotator.training.sequences import TreeFitConfig, whole_rally_correctness
from dataset_builder.vision import load_npy_xz
from experiments.annotator.heuristic_only_tune.data import score_stream
from experiments.annotator.old_court_regression.retrain import (
    dump_atomically,
    stream_payload,
)
from experiments.annotator.tree_hp_tune.run import pooled_scores
from experiments.annotator.tree_simplification.options import LabelledOptions
from experiments.annotator.tree_simplification.pilot import (
    load_prepared,
    progress,
    require_raw_features,
)
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
        five = load_npy_xz(targets_root / 'videos' / f'{pool.original.identity}.npy.xz')
        pools[pool.original.identity] = replace(pool, targets=five_frame_preferred_targets(pool, five))
    if len(pools) != 32:
        raise ValueError('Development export requires all 32 training videos')
    return pools


def fit_review(run: Path, depth: int, targets: Path, output: Path, settings: dict) -> Any:
    pools = load_training(settings, targets)
    sample = next(iter(pools))
    saved = joblib.load(Path(settings['pools']) / 'videos' / f'{sample}.joblib')
    rebuilt = build_rally_options(saved.runtime.original, settings['pool_settings']['max_edits'],
                                 settings['pool_settings']['deletion_candidates'])
    verify_pool(rebuilt, saved)
    np.testing.assert_array_equal(option_targets(saved.original, rebuilt), saved.targets)
    progress(output, 'training-pool-parity', video=sample, rows=len(saved.targets))
    # Reuse completed selector fits; new held-group review fits stay in this export.
    fits = output / 'review-fits'
    shutil.copytree(run / 'fits', fits / 'fits', dirs_exist_ok=True)
    trainer = LayerTrainer(pools, TreeFitConfig(**settings['tree']), fits, settings['span_weight'])
    groups = tuple(sorted({pool.original.group for pool in pools.values()}))
    predictions = {}
    for identity, pool in pools.items():
        held = tuple(group for group in groups if group != pool.original.group)
        models = RallyLayerModels(trainer.fit(held, depth))
        predictions[identity] = predict_rally_options(models, pool.runtime)
    config = ConfidenceFitConfig(**settings['pool_settings']['settings']['confidence'])
    review = fit_rally_review_model([pool.original for pool in pools.values()], predictions, config)
    dump_atomically(review, output / 'review-model.joblib')
    return review


def validate(
    run: Path, depth: int, settings: dict, old: AnnotatorModels, models: AnnotatorModels, output: Path,
) -> dict[str, Any]:
    prepared_root = Path(settings['pool_settings']['prepared'])
    pools_root = Path(settings['pools'])
    per_video, rankings, rows = {}, {}, []
    if not isinstance(old.sequences, SequenceModels):
        raise TypeError('The contact run must contain the old eight-model reference stack')
    (output / 'streams').mkdir(exist_ok=True)
    for path in sorted((prepared_root / 'videos').glob('*.joblib')):
        if read_json(path.with_suffix('.json.gz'))['split'] != 'validation':
            continue
        prepared = load_prepared(prepared_root, path.stem)
        video = prepared.video
        prediction = predict_contact_features(prepared.evidence, prepared.features, models)
        layer_prediction = cast(RallyLayerPrediction, prediction.refined)
        saved: LabelledOptions = joblib.load(pools_root / 'videos' / path.name)
        verify_pool(layer_prediction.pool, saved)
        stream = stream_payload(layer_prediction.sequences, layer_prediction.events)
        if stream != read_json(run / f'layers-{depth}' / 'streams' / f'{video.identity}.json.gz'):
            raise ValueError(f'{video.identity}: production stream differs from the saved experiment')
        write_json(output / 'streams' / f'{video.identity}.json.gz', stream)
        per_video[video.identity] = score_stream(video, stream)
        selector = score_layer_confidence(layer_prediction, None)
        context = layer_prediction.pool.original
        old_stack = finish_sequences(context.initial, context.events, context.pool, old.sequences,
                                     fps=video.fps, frame_count=context.frame_count)
        old_review = score_rally_confidence(old_stack, old.confidence, old.sequences.insertion, video.fps)
        old_stack_stream = stream_payload(old_stack.sequences, old_stack.events)
        contact_run = Path(settings['pool_settings']['contact_run'])
        if old_stack_stream != read_json(contact_run / 'validation_streams' / f'{video.identity}.json.gz'):
            raise ValueError(f'{video.identity}: old eight-model stack stream differs from the contact run')
        rankings[video.identity] = {}
        for tolerance in (5, 10):
            for name, sequences, review in (
                ('selector', layer_prediction.sequences, selector), ('fitted', layer_prediction.sequences, prediction.confidence),
                ('old-eight-model-stack', old_stack.sequences, old_review),
            ):
                by_span = {sequence.span_id: sequence for sequence in sequences}
                labels = np.asarray([whole_rally_correctness(
                    by_span[span], video.rallies, video.fps, tolerance_at_30_fps=tolerance,
                ) for span in review.features.span_ids], dtype=np.int8)
                rankings[video.identity][f'{name}-{tolerance}'] = ranking_metrics(labels, review.scores)
                for span, label, score in zip(review.features.span_ids, labels, review.scores, strict=True):
                    rows.append({'video': video.identity, 'span': span, 'tolerance': tolerance,
                                 'ranking': name, 'correct': int(label), 'score': float(score)})
        progress(output, 'production-parity', video=video.identity, complete=len(per_video), planned=8)
    if len(per_video) != 8:
        raise ValueError('Production replay requires all eight validation videos')
    pooled = {}
    for name in ('selector', 'fitted', 'old-eight-model-stack'):
        for tolerance in (5, 10):
            selected = [row for row in rows if row['ranking'] == name and row['tolerance'] == tolerance]
            pooled[f'{name}-{tolerance}'] = ranking_metrics(
                np.asarray([row['correct'] for row in selected]), np.asarray([row['score'] for row in selected]),
            )
    write_json(output / 'review-rows.json.gz', rows)
    return {'per_video': per_video, 'totals': pooled_scores(per_video), 'review_per_video': rankings, 'review': pooled}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--layers', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--five-frame-targets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.threads <= 28:
        parser.error('--threads must be between 1 and 28')
    result = read_json(args.run / 'result.json.gz')
    settings = read_json(args.run / 'settings.json.gz')
    require_raw_features(Path(settings['pool_settings']['prepared']))
    if not result['complete'] or args.layers > settings['layers']:
        parser.error('--run must contain the completed requested depth')
    if settings.get('specialists') or settings['original_options'] or 'target_policy' not in settings:
        parser.error('Export requires a general-layer stack on all options with five-frame-preferred targets')
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = {'run': str(args.run.resolve()), 'layers': args.layers, 'settings': settings}
    path = args.output / 'settings.json.gz'
    if path.exists() and read_json(path) != receipt:
        raise ValueError('Export inputs changed; use a fresh output directory')
    write_json(path, receipt)
    with threadpool_limits(limits=args.threads):
        old = load_models(Path(settings['pool_settings']['contact_run']) / 'bundle')
        for tree in fitted_trees(old):
            if type(tree).__module__.startswith('xgboost'):
                tree.set_params(device='cpu')
        layers = joblib.load(args.run / f'layers-{args.layers}' / 'rally-models.joblib')
        if len(layers) != args.layers:
            raise ValueError('Saved model count differs from the requested prefix')
        review = fit_review(args.run, args.layers, args.five_frame_targets, args.output, settings)
        pool_settings = settings['pool_settings']
        models = replace(old, sequences=RallyLayerModels(layers, pool_settings['max_edits'],
                                                       pool_settings['deletion_candidates']), confidence=review)
        save_models(models, args.output / 'bundle')
        models = load_models(args.output / 'bundle')
        result = validate(args.run, args.layers, settings, old, models, args.output)
    result.update(complete=True, source=source_record(), layers=args.layers, production_streams_match=True,
                  review_config=asdict(ConfidenceFitConfig(**pool_settings['settings']['confidence'])))
    write_json(args.output / 'result.json.gz', result)
    progress(args.output, 'complete', totals=result['totals'], review=result['review'])


if __name__ == '__main__':
    main()
