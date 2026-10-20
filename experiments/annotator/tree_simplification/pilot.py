"""Compare HGB and GPU XGBoost with the existing eight-model annotator."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

sys.path.insert(0, str(REPOSITORY := Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(REPOSITORY / 'src'))

from experiments.annotator.heuristic_only_tune.run import (
    load_paths,
    read_json,
    source_record,
    write_json,
)

# isort: split

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from annotator.models import AnnotatorModels, SideGeometry, load_models, save_models
from annotator.sequence.refine import finish_sequences
from annotator.training.contact import ContactTrainingVideo, fit_contact_model
from annotator.training.trees import TreeBackend
from annotator.training.workflow import (
    DEFAULT_TRAINING_SETTINGS,
    Split,
    TrainingSettings,
    VideoInput,
    sequence_training_video,
)
from experiments.annotator.heuristic_only_tune.data import (
    COUNT_FIELDS,
    load_original_videos,
    score_stream,
)
from experiments.annotator.old_court_regression import retrain
from experiments.annotator.tree_hp_tune.run import development_for, pooled_scores
from experiments.annotator.tree_simplification.data import PreparedVideo


def settings_for(
    backend: TreeBackend, device: str, min_child_weight: float, *, seed_offset: int = 0,
) -> TrainingSettings:
    base = DEFAULT_TRAINING_SETTINGS
    options = {'backend': backend, 'device': device, 'min_child_weight': min_child_weight}
    return replace(
        base,
        contact=replace(base.contact, **options, random_seed=base.contact.random_seed + seed_offset),
        confidence=replace(base.confidence, **options, random_seed=base.confidence.random_seed + seed_offset),
        sequences=replace(
            base.sequences,
            serve=replace(base.sequences.serve, **options, random_seed=base.sequences.serve.random_seed + seed_offset),
            sequence=replace(
                base.sequences.sequence, **options, random_seed=base.sequences.sequence.random_seed + seed_offset,
            ),
        ),
    )


def progress(output: Path, stage: str, **details: Any) -> None:
    payload = {'stage': stage, **details}
    write_json(output / 'progress.json.gz', payload)
    print(payload, flush=True)


def load_prepared(directory: Path, identity: str) -> PreparedVideo:
    return joblib.load(directory / 'videos' / f'{identity}.joblib')


def training_pool(prepared: PreparedVideo, contact: Any, settings: TrainingSettings) -> Any:
    video = prepared.video
    return sequence_training_video(
        VideoInput(video.identity, prepared.group, Split(video.split), Path('unused-labels.csv')),
        prepared.evidence, prepared.features, video.rallies, contact, settings.contact_prediction, SideGeometry.VIDEO,
    )


def evaluate(prepared_root: Path, models: AnnotatorModels, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    preparation = read_json(prepared_root / 'preparation.json.gz')['inputs']
    preprocessing = json.loads(json.dumps(asdict(models.preprocessing), default=sorted))
    if preparation['preprocessing'] != preprocessing or models.side_geometry != SideGeometry.VIDEO:
        raise ValueError('Bundle preprocessing or side geometry differs from prepared evidence')
    (output / 'validation_streams').mkdir(parents=True, exist_ok=True)
    for tree in (models.contact, models.confidence, *vars(models.sequences).values()):
        if type(tree).__module__.startswith('xgboost'):
            tree.set_params(device='cpu')
    per_video = {}
    settings = replace(DEFAULT_TRAINING_SETTINGS, contact_prediction=models.contact_settings)
    for path in sorted((prepared_root / 'videos').glob('*.joblib')):
        if read_json(path.with_suffix('.json.gz'))['split'] != 'validation':
            continue
        prepared: PreparedVideo = joblib.load(path)
        video = prepared.video
        pool = training_pool(prepared, models.contact, settings)
        refined = finish_sequences(
            pool.initial, pool.events, pool.pool, models.sequences,
            fps=video.fps, frame_count=len(prepared.evidence.track),
        )
        stream = retrain.stream_payload(refined.sequences, refined.events)
        write_json(output / 'validation_streams' / f'{video.identity}.json.gz', stream)
        per_video[video.identity] = score_stream(video, stream)
        progress(output, 'validation', video=video.identity, complete=len(per_video), planned=8)
    if len(per_video) != 8:
        raise ValueError('Evaluation requires all eight prepared validation videos')
    result = {'per_video': per_video, 'totals': pooled_scores(per_video), 'wall_s': time.perf_counter() - started}
    write_json(output / 'validation.json.gz', result)
    return result


def verify_reference(result: dict[str, Any], expected_path: Path, output: Path) -> None:
    expected = read_json(expected_path)['per_video']
    actual = result['per_video']
    if set(actual) != set(expected):
        raise ValueError('Reference validation video membership differs')
    for identity, scores in actual.items():
        for tolerance in ('5', '10'):
            for field in COUNT_FIELDS:
                if scores[tolerance][field] != expected[identity][tolerance][field]:
                    raise ValueError(f'Reference differs: {identity}, tolerance {tolerance}, {field}')
    write_json(output / 'reference-parity.json.gz', {'matches': True, 'videos': len(actual)})


def fit(prepared_root: Path, paths: dict[str, Path], output: Path, settings: TrainingSettings) -> AnnotatorModels:
    videos = load_original_videos(paths['shots_master'])
    development = development_for(videos)
    training_ids = development.videos(retrain.TRAINING_GROUPS)
    training = {}
    for identity in training_ids:
        prepared = load_prepared(prepared_root, identity)
        if prepared.video.split != 'train' or prepared.group != development.group_by_video[identity]:
            raise ValueError(f'{identity}: training membership differs')
        frames = np.asarray(sorted(frame for rally in prepared.video.rallies for frame in rally.frames), dtype=np.int32)
        training[identity] = ContactTrainingVideo(identity, prepared.features.rows, frames, prepared.video.fps)
    del prepared
    contacts = {}
    for name, sampling_order in retrain.contact_fit_videos(development).items():
        if name == 'final_40':
            continue
        path = output / 'contacts' / f'{name}.joblib'
        if path.exists():
            contacts[name] = joblib.load(path)
            continue
        started = time.perf_counter()
        progress(output, 'contact-fit', fit=name)
        fitted = fit_contact_model(
            [training[identity] for identity in sampling_order], settings.contact,
            fit_video_order=development.id_order(sampling_order),
        )
        contacts[name] = fitted.model
        path.parent.mkdir(parents=True, exist_ok=True)
        retrain.dump_atomically(fitted.model, path)
        write_json(path.with_suffix('.json.gz'), {
            'wall_s': time.perf_counter() - started, 'rows': len(fitted.selection.labels),
            'positive_rows': int(fitted.selection.labels.sum()), 'training_videos': sampling_order,
        })
        del fitted
    del training
    gc.collect()
    pools = []
    for identity in training_ids:
        prepared = load_prepared(prepared_root, identity)
        pools.append(training_pool(prepared, contacts[f'held_{prepared.group}'], settings))
        progress(output, 'sequence-pool', video=identity, complete=len(pools), planned=32)
    del prepared
    gc.collect()
    started = time.perf_counter()
    progress(output, 'downstream-fit')
    sequence_fit, confidence, labels = retrain.fit_downstream(pools, settings)
    write_json(output / 'downstream-fit.json.gz', {
        'wall_s': time.perf_counter() - started, 'confidence_rows': len(labels),
    })
    models = AnnotatorModels(
        contacts['v_32'], sequence_fit.models, confidence, settings.contact_prediction,
        settings.preprocessing, SideGeometry.VIDEO,
    )
    save_models(models, output / 'bundle')
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=list(TreeBackend), default=TreeBackend.HIST_GRADIENT_BOOSTING)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--min-child-weight', type=float, default=5.0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed-offset', type=int, default=0)
    parser.add_argument('--reference-bundle', type=Path)
    parser.add_argument('--expected-validation', type=Path)
    args = parser.parse_args()
    if not 1 <= args.threads <= 28:
        parser.error('--threads must be between 1 and 28')
    if args.reference_bundle is not None and args.seed_offset != 0:
        parser.error('--seed-offset requires fitting models and cannot be combined with --reference-bundle')
    if args.reference_bundle is not None and not args.reference_bundle.is_dir():
        parser.error('--reference-bundle must name an existing bundle')
    source = source_record()
    settings = settings_for(TreeBackend(args.backend), args.device, args.min_child_weight, seed_offset=args.seed_offset)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {'settings': asdict(settings), 'prepared': str(args.prepared.resolve())}
    config = json.loads(json.dumps(config, default=sorted))
    config_path = args.output / 'settings.json.gz'
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError('Pilot settings changed; use a fresh output directory')
    write_json(config_path, config)
    started = time.perf_counter()
    with threadpool_limits(limits=args.threads):
        bundle = args.reference_bundle or args.output / 'bundle'
        fitted_this_run = not bundle.exists()
        models = load_models(bundle) if bundle.exists() else fit(args.prepared, load_paths(args.paths), args.output, settings)
        result = evaluate(args.prepared, models, args.output)
        if args.expected_validation:
            verify_reference(result, args.expected_validation, args.output)
    result.update(
        source=source, settings=config['settings'], total_wall_s=time.perf_counter() - started,
        threads=args.threads, fitted_this_run=fitted_this_run,
        training_device=args.device, prediction_device='cpu',
    )
    if args.backend == TreeBackend.XGBOOST:
        result['xgboost'] = version('xgboost')
    write_json(args.output / 'result.json.gz', result)
    progress(args.output, 'complete', totals=result['totals'])


if __name__ == '__main__':
    main()
