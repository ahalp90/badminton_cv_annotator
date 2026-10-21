"""Fit review ranking and replay the experimental FPS-normalised stack in memory.

No production bundle is written: its loader does not yet encode motion units.
The separate contact/rally/review checkpoints require the transformed preparation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from annotator.config import BaseAnnotatorConfig
from annotator.contacts.model import ContactModelConfig
from annotator.hybrid import predict_contact_features
from annotator.models import AnnotatorModels, SideGeometry, fitted_trees
from annotator.sequence.contacts import stream_payload
from annotator.sequence.rally_model import (
    RallyLayerModels,
    RallyLayerPrediction,
    score_rally_review,
)
from annotator.training.sequences import whole_rally_correctness
from experiments.annotator import development as experiment
from experiments.annotator.development import pooled_scores
from experiments.annotator.heuristic_only_tune.data import score_stream
from experiments.annotator.heuristic_only_tune.run import source_record
from experiments.annotator.tree_simplification import integrate
from experiments.annotator.tree_simplification.fps_normalisation import (
    preparation_receipt,
)


def evaluate(run: Path, targets: Path, output: Path) -> None:
    """Check rebuilt pools and streams before reporting fitted review rankings."""
    settings = experiment.read_json(run / 'settings.json.gz')
    result = experiment.read_json(run / 'result.json.gz')
    if not result['complete'] or settings['layers'] != 1:
        raise ValueError('FPS comparison requires a complete single-selector run')
    pool_settings = settings['pool_settings']
    prepared_root = Path(pool_settings['prepared'])
    current_units = preparation_receipt(prepared_root)
    preparation = current_units['preparation']
    contact_run_root = Path(pool_settings['contact_run'])
    units = experiment.read_json(contact_run_root / 'motion-units.json.gz')
    if units != current_units:
        raise ValueError('Contact fit and replay use different preparation semantics')
    output.mkdir(parents=True, exist_ok=True)
    receipt = {'run': str(run.resolve()), 'settings': settings, 'motion_units': units}
    receipt_path = output / 'settings.json.gz'
    if receipt_path.exists() and experiment.read_json(receipt_path) != receipt:
        raise ValueError('Review inputs changed; use a fresh output directory')
    experiment.write_json(receipt_path, receipt)
    review = integrate.fit_review(run, 1, targets, output, settings)
    layers = joblib.load(run / 'layers-1' / 'rally-models.joblib')
    if len(layers) != 1:
        raise ValueError('Expected exactly one rally selector')
    models = AnnotatorModels(
        joblib.load(contact_run_root / 'contacts' / 'v_32.joblib'),
        RallyLayerModels(layers, pool_settings['max_edits'], pool_settings['deletion_candidates']),
        review, ContactModelConfig(**pool_settings['settings']['contact_prediction']),
        BaseAnnotatorConfig(), SideGeometry.VIDEO,
    )
    for tree in fitted_trees(models):
        tree.set_params(device='cpu')
    per_video, rankings, rows = {}, {}, []
    (output / 'streams').mkdir(exist_ok=True)
    for identity, group in preparation['inputs']['groups'].items():
        if group != experiment.VALIDATION_GROUP:
            continue
        prepared = experiment.load_prepared(prepared_root, identity)
        if prepared.video.split != 'validation':
            raise ValueError('Replay video is not validation')
        prediction = predict_contact_features(prepared.evidence, prepared.features, models)
        layer_prediction = prediction.refined
        if not isinstance(layer_prediction, RallyLayerPrediction):
            raise TypeError('Expected a rally-layer prediction')
        saved = joblib.load(Path(settings['pools']) / 'videos' / f'{identity}.joblib')
        integrate.verify_pool(layer_prediction.pool, saved)
        stream = stream_payload(layer_prediction.sequences, layer_prediction.events)
        if stream != experiment.read_json(run / 'layers-1' / 'streams' / f'{identity}.json.gz'):
            raise ValueError(f'{identity}: replay stream differs')
        experiment.write_json(output / 'streams' / f'{identity}.json.gz', stream)
        per_video[identity] = score_stream(prepared.video, stream)
        selector = score_rally_review(layer_prediction, None)
        rankings[identity] = {}
        by_span = {sequence.span_id: sequence for sequence in layer_prediction.sequences}
        for tolerance in (5, 10):
            for name, confidence in (('selector', selector), ('fitted', prediction.confidence)):
                labels = np.asarray([
                    whole_rally_correctness(
                        by_span[span], prepared.video.rallies, prepared.video.fps, tolerance_at_30_fps=tolerance,
                    )
                    for span in confidence.features.span_ids
                ], dtype=np.int8)
                rankings[identity][f'{name}-{tolerance}'] = integrate.ranking_metrics(labels, confidence.scores)
                for span, label, score in zip(confidence.features.span_ids, labels, confidence.scores, strict=True):
                    rows.append({'video': identity, 'span': span, 'tolerance': tolerance,
                                 'ranking': name, 'correct': int(label), 'score': float(score)})
        experiment.progress(output, 'replay-parity', video=identity, complete=len(per_video), planned=8)
    if len(per_video) != 8:
        raise ValueError('Replay requires the eight validation videos')
    pooled = {}
    for name in ('selector', 'fitted'):
        for tolerance in (5, 10):
            selected = [row for row in rows if row['ranking'] == name and row['tolerance'] == tolerance]
            pooled[f'{name}-{tolerance}'] = integrate.ranking_metrics(
                np.asarray([row['correct'] for row in selected]), np.asarray([row['score'] for row in selected]),
            )
    experiment.write_json(output / 'review-rows.json.gz', rows)
    experiment.write_json(output / 'result.json.gz', {
        'complete': True, 'source': source_record(), 'motion_units': units['schema'],
        'replay_streams_match': True,
        'per_video': per_video, 'totals': pooled_scores(per_video), 'review_per_video': rankings, 'review': pooled,
    })
    experiment.progress(output, 'complete', totals=pooled_scores(per_video))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--five-frame-targets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.threads <= 28:
        parser.error('--threads must be between 1 and 28')
    with threadpool_limits(limits=args.threads):
        evaluate(args.run, args.five_frame_targets, args.output)


if __name__ == '__main__':
    main()
