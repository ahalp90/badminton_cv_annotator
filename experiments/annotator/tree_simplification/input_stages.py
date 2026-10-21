"""Fit contact inputs and export a raw-preprocessing rally stack."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

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
from threadpoolctl import threadpool_limits

from annotator.contacts.model import ContactModelConfig
from annotator.hybrid import predict_contact_features
from annotator.models import (
    AnnotatorModels,
    SideGeometry,
    fitted_trees,
    load_models,
    save_models,
)
from annotator.sequence.rally_model import (
    RallyLayerModels,
    RallyLayerPrediction,
    score_layer_confidence,
)
from annotator.training.contact import ContactTrainingVideo, fit_contact_model
from annotator.training.sequences import whole_rally_correctness
from annotator.training.trees import TreeBackend
from annotator.training.workflow import TrainingSettings
from experiments.annotator.heuristic_only_tune.data import (
    load_original_videos,
    score_stream,
)
from experiments.annotator.old_court_regression import retrain
from experiments.annotator.old_court_regression.retrain import stream_payload
from experiments.annotator.tree_hp_tune.run import development_for, pooled_scores
from experiments.annotator.tree_simplification import integrate, pilot
from experiments.annotator.tree_simplification.data import PreparedVideo
from experiments.annotator.tree_simplification.options import LabelledOptions
from experiments.annotator.tree_simplification.prepare import (
    preprocessing_from_record,
)


def _prepared_inputs(prepared_root: Path) -> dict[str, Any]:
    """Load and validate the saved raw preprocessing identity."""
    pilot.require_raw_features(prepared_root)
    manifest = read_json(prepared_root / 'preparation.json.gz')
    if manifest['source'] != source_record():
        raise ValueError('Prepared source differs from this checkout; use a fresh prepared directory')
    inputs = manifest['inputs']
    preprocessing_from_record(inputs['preprocessing'])
    if len(inputs['groups']) != 40:
        raise ValueError('Prepared development input must name all 40 fixed videos')
    return inputs


def _training_records(
    prepared_root: Path, inputs: dict[str, Any], development: Any,
) -> dict[str, ContactTrainingVideo]:
    """Load contact rows from the fixed 32-video development training split."""
    training_ids = development.videos(retrain.TRAINING_GROUPS)
    training: dict[str, ContactTrainingVideo] = {}
    for identity in training_ids:
        path = prepared_root / 'videos' / f'{identity}.joblib'
        sidecar = read_json(path.with_suffix('.json.gz'))
        if sidecar.get('video') != identity or sidecar.get('preprocessing') != inputs['preprocessing']:
            raise ValueError(f'{identity}: prepared sidecar differs from the saved preprocessing')
        prepared: PreparedVideo = joblib.load(path)
        expected_group = development.group_by_video[identity]
        if prepared.video.identity != identity or prepared.video.split != 'train' or prepared.group != expected_group:
            raise ValueError(f'{identity}: prepared identity or training group differs')
        frames = np.asarray(sorted(frame for rally in prepared.video.rallies for frame in rally.frames), dtype=np.int32)
        training[identity] = ContactTrainingVideo(
            identity, prepared.features.rows, frames, prepared.video.fps,
        )
    return training


def _pool_settings(settings: TrainingSettings, prepared_root: Path) -> dict[str, Any]:
    training_settings = asdict(settings)
    training_settings.pop('rally_layers')
    training_settings = json.loads(json.dumps(training_settings, default=sorted))
    return {'settings': training_settings, 'prepared': str(prepared_root.resolve())}


def fit_contacts(
    prepared_root: Path,
    output: Path,
    *,
    seed_offset: int = 0,
    threads: int = 2,
    device: str = 'cuda',
) -> dict[str, Any]:
    """Fit held-A/B/C/D and v_32 contacts using the preparation's config."""
    if not 1 <= threads <= 28:
        raise ValueError('threads must be between 1 and 28')
    if device not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda')
    prepared_root = prepared_root.resolve()
    inputs = _prepared_inputs(prepared_root)
    videos = load_original_videos(Path(inputs['paths']['shots_master']))
    development = development_for(videos)
    training_ids = development.videos(retrain.TRAINING_GROUPS)
    training = _training_records(prepared_root, inputs, development)
    if set(training) != set(training_ids):
        raise ValueError('Contact fit inputs do not match the fixed 32-video training split')

    settings = replace(
        pilot.settings_for(TreeBackend.XGBOOST, device, 5.0, seed_offset=seed_offset),
        preprocessing=preprocessing_from_record(inputs['preprocessing']),
    )
    pool_settings = _pool_settings(settings, prepared_root)
    source = source_record()
    output.mkdir(parents=True, exist_ok=True)
    settings_path = output / 'settings.json.gz'
    if settings_path.exists() and read_json(settings_path) != pool_settings:
        raise ValueError('Contact settings changed; use a fresh output directory')
    write_json(settings_path, pool_settings)

    fits = retrain.contact_fit_videos(development)
    expected_names = {'held_A', 'held_B', 'held_C', 'held_D', 'v_32'}
    if set(fits) != expected_names | {'final_40'}:
        raise ValueError('Contact fit definitions differ from the five production contact models')
    fit_identities = {
        name: {
            'fit': name,
            'sampling_order': list(sampling_order),
            'fit_video_order': list(development.id_order(sampling_order)),
        }
        for name, sampling_order in fits.items()
        if name != 'final_40'
    }
    stage = {
        'schema': 'tree-simplification-contacts/1',
        'source': source,
        'seed_offset': seed_offset,
        'device': device,
        'threads': threads,
        'prepared': str(prepared_root),
        'settings': pool_settings,
        'fits': {name: fit_identities[name] for name in sorted(fit_identities)},
    }
    stage_path = output / 'contact-stage.json.gz'
    if stage_path.exists() and read_json(stage_path) != stage:
        raise ValueError('Contact stage identity changed; use a fresh output directory')
    (output / 'contacts').mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=threads):
        for name, sampling_order in fits.items():
            if name == 'final_40':
                continue
            fit_order = development.id_order(sampling_order)
            identity = {
                **fit_identities[name],
                'source': source,
                'seed_offset': seed_offset,
                'device': device,
            }
            model_path = output / 'contacts' / f'{name}.joblib'
            record_path = model_path.with_suffix('.json.gz')
            if model_path.exists() != record_path.exists():
                raise ValueError(f'{name}: contact model and metadata must be present together')
            if model_path.exists():
                cached = read_json(record_path)
                if cached.get('identity') != identity:
                    raise ValueError(f'{name}: cached contact fit differs; choose a fresh output directory')
                continue

            started = time.perf_counter()
            pilot.progress(output, 'contact-fit-start', fit=name, videos=len(sampling_order))
            fitted = fit_contact_model(
                [training[identity] for identity in sampling_order], settings.contact,
                fit_video_order=fit_order,
            )
            retrain.dump_atomically(fitted.model, model_path)
            record = {
                'identity': identity,
                'source': source,
                'rows': len(fitted.selection.labels),
                'positive_rows': int(fitted.selection.labels.sum()),
                'wall_s': time.perf_counter() - started,
            }
            write_json(record_path, record)
            pilot.progress(output, 'contact-fit-complete', fit=name, rows=record['rows'])

    write_json(stage_path, stage)
    return stage


def _export_inputs(run: Path, targets: Path) -> tuple[dict[str, Any], dict[str, Any], Path, Path, int]:
    result = read_json(run / 'result.json.gz')
    settings = read_json(run / 'settings.json.gz')
    if not result.get('complete'):
        raise ValueError('Export requires a complete fitted rally run')
    if settings.get('specialists') or settings.get('original_options') or 'target_policy' not in settings:
        raise ValueError('Export requires a general-layer run on all options with five-frame targets')
    depth = settings['layers']
    if not isinstance(depth, int) or not 1 <= depth <= 3:
        raise ValueError('Export requires a saved rally stack with one to three layers')
    pools = settings['pool_settings']
    prepared_root = Path(pools['prepared']).resolve()
    contact_run_root = Path(pools['contact_run']).resolve()
    inputs = _prepared_inputs(prepared_root)
    if pools['settings']['preprocessing'] != inputs['preprocessing']:
        raise ValueError('Contact settings and prepared evidence use different preprocessing')
    if read_json(contact_run_root / 'settings.json.gz') != {
        'settings': pools['settings'], 'prepared': str(prepared_root),
    }:
        raise ValueError('Contact-stage settings differ from the fitted pool manifest')
    source = read_json(prepared_root / 'preparation.json.gz')['source']
    contact_source = read_json(contact_run_root / 'contact-stage.json.gz')['source']
    if contact_source != source or result['source'] != source:
        raise ValueError('Preparation, contact and selector sources differ; refit from one clean checkout')
    if settings['target_policy']['manifest'] != read_json(targets / 'targets.json.gz'):
        raise ValueError('Five-frame target manifest differs from the fitted rally run')
    return settings, inputs, prepared_root, contact_run_root, depth


def _review_rows(
    video: Any,
    identity: str,
    prediction: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    layer_prediction = prediction.refined
    if not isinstance(layer_prediction, RallyLayerPrediction):
        raise TypeError('Raw export requires a rally-layer prediction')
    selector = score_layer_confidence(layer_prediction, None)
    rankings: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    by_span = {sequence.span_id: sequence for sequence in layer_prediction.sequences}
    for tolerance in (5, 10):
        for name, confidence in (('selector', selector), ('fitted', prediction.confidence)):
            labels = np.asarray([
                whole_rally_correctness(
                    by_span[span], video.rallies, video.fps, tolerance_at_30_fps=tolerance,
                )
                for span in confidence.features.span_ids
            ], dtype=np.int8)
            rankings[f'{name}-{tolerance}'] = integrate.ranking_metrics(labels, confidence.scores)
            for span, label, score in zip(confidence.features.span_ids, labels, confidence.scores, strict=True):
                rows.append({
                    'video': identity,
                    'span': span,
                    'tolerance': tolerance,
                    'ranking': name,
                    'correct': int(label),
                    'score': float(score),
                })
    return rankings, rows


def export_run(run: Path, targets: Path, output: Path, *, threads: int = 2) -> dict[str, Any]:
    """Fit review, build a production bundle, and verify all validation streams."""
    if not 1 <= threads <= 28:
        raise ValueError('threads must be between 1 and 28')
    run, targets, output = run.resolve(), targets.resolve(), output.resolve()
    settings, inputs, prepared_root, contact_run_root, depth = _export_inputs(run, targets)
    receipt = {
        'schema': 'tree-simplification-raw-export/1',
        'run': str(run),
        'targets': str(targets),
        'settings': settings,
        'preprocessing': inputs['preprocessing'],
        'source': source_record(),
    }
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / 'settings.json.gz'
    if receipt_path.exists() and read_json(receipt_path) != receipt:
        raise ValueError('Export inputs changed; use a fresh output directory')
    write_json(receipt_path, receipt)
    with threadpool_limits(limits=threads):
        review = integrate.fit_review(run, depth, targets, output, settings)
    pool_settings = settings['pool_settings']
    layer_models = joblib.load(run / f'layers-{depth}' / 'rally-models.joblib')
    if len(layer_models) != depth:
        raise ValueError('Saved rally-layer model depth differs from the completed run')
    sequences = RallyLayerModels(
        tuple(layer_models), pool_settings['max_edits'], pool_settings['deletion_candidates'],
    )
    contact = joblib.load(contact_run_root / 'contacts' / 'v_32.joblib')
    preprocessing = preprocessing_from_record(inputs['preprocessing'])
    models = AnnotatorModels(
        contact,
        sequences,
        review,
        ContactModelConfig(**pool_settings['settings']['contact_prediction']),
        preprocessing,
        SideGeometry.VIDEO,
    )
    for tree in fitted_trees(models):
        if type(tree).__module__.startswith('xgboost'):
            tree.set_params(device='cpu')
    bundle = output / 'bundle'
    save_models(models, bundle)
    models = load_models(bundle)

    pools_root = Path(settings['pools']).resolve()
    validation_ids = [identity for identity, group in inputs['groups'].items() if group == retrain.VALIDATION_GROUP]
    if len(validation_ids) != 8:
        raise ValueError('Prepared manifest must contain the fixed eight validation videos')
    (output / 'streams').mkdir(exist_ok=True)
    per_video: dict[str, Any] = {}
    review_per_video: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    with threadpool_limits(limits=threads):
        for identity in sorted(validation_ids):
            prepared = pilot.load_prepared(prepared_root, identity)
            video = prepared.video
            if video.split != 'validation' or prepared.group != inputs['groups'][identity]:
                raise ValueError(f'{identity}: replay split or group differs from preparation')
            prediction = predict_contact_features(prepared.evidence, prepared.features, models)
            layer_prediction = prediction.refined
            if not isinstance(layer_prediction, RallyLayerPrediction):
                raise TypeError('Raw export requires a rally-layer prediction')
            saved: LabelledOptions = joblib.load(pools_root / 'videos' / f'{identity}.joblib')
            integrate.verify_pool(layer_prediction.pool, saved)
            stream = stream_payload(layer_prediction.sequences, layer_prediction.events)
            expected_path = run / f'layers-{depth}' / 'streams' / f'{identity}.json.gz'
            if stream != read_json(expected_path):
                raise ValueError(f'{identity}: raw production stream differs from the saved rally run')
            write_json(output / 'streams' / f'{identity}.json.gz', stream)
            per_video[identity] = score_stream(video, stream)
            review_per_video[identity], video_rows = _review_rows(video, identity, prediction)
            rows.extend(video_rows)
            pilot.progress(output, 'raw-replay-parity', video=identity, complete=len(per_video), planned=8)

    pooled_review = {}
    for name in ('selector', 'fitted'):
        for tolerance in (5, 10):
            selected = [row for row in rows if row['ranking'] == name and row['tolerance'] == tolerance]
            pooled_review[f'{name}-{tolerance}'] = integrate.ranking_metrics(
                np.asarray([row['correct'] for row in selected], dtype=np.int8),
                np.asarray([row['score'] for row in selected], dtype=np.float64),
            )
    write_json(output / 'review-rows.json.gz', rows)
    result = {
        'complete': True,
        'source': receipt['source'],
        'preprocessing': inputs['preprocessing'],
        'replay_streams_match': True,
        'per_video': per_video,
        'totals': pooled_scores(per_video),
        'review_per_video': review_per_video,
        'review': pooled_review,
    }
    write_json(output / 'result.json.gz', result)
    pilot.progress(output, 'raw-export-complete', videos=len(per_video), totals=result['totals'])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)

    contacts = commands.add_parser('contacts', help='fit held-group and full-training contact trees')
    contacts.add_argument('--prepared', type=Path, required=True)
    contacts.add_argument('--output', type=Path, required=True)
    contacts.add_argument('--seed-offset', type=int, default=0)
    contacts.add_argument('--threads', type=int, default=2)
    contacts.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')

    export = commands.add_parser('export', help='export a raw production bundle and replay validation')
    export.add_argument('--run', type=Path, required=True)
    export.add_argument('--five-frame-targets', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    export.add_argument('--threads', type=int, default=2)

    args = parser.parse_args()
    if args.command == 'contacts':
        fit_contacts(
            args.prepared, args.output, seed_offset=args.seed_offset,
            threads=args.threads, device=args.device,
        )
    else:
        export_run(args.run, args.five_frame_targets, args.output, threads=args.threads)


if __name__ == '__main__':
    main()
