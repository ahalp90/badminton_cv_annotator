"""Compare and refit the selected annotator after changing court preparation.

Prepare each court variant once, then run one evaluate or refit process per seed.
The caller can bind each seed process to six CPU cores; spawned workers inherit
that affinity. CPU preparation and GPU fitting run in separate phases.
"""

from __future__ import annotations

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from annotator.config import BaseAnnotatorConfig, scale_config_to_fps
from annotator.inference.learned import (
    contact_features_from_inputs,
    predict_from_contact_features,
)
from annotator.inference.prepare_annotation_inputs import prepare_annotation_inputs
from annotator.training.contact import ContactFitConfig
from experiments.annotator import development
from experiments.annotator.contact_records import stream_payload
from experiments.annotator.heuristic_only_tune.data import (
    SearchVideo,
    load_original_videos,
    load_test_inputs,
    load_test_videos,
    score_stream,
)
from experiments.annotator.heuristic_only_tune.run import (
    load_paths,
    read_json,
    write_json,
)
from experiments.annotator.monotrack_benchmark.benchmark_scripts.common import (
    load_manifest,
)
from experiments.annotator.monotrack_benchmark.writeup_scripts import metrics
from experiments.annotator.tree_simplification.data import PreparedVideo
from experiments.annotator.tree_simplification.hp_search_finish import (
    _models,
    _score_prediction,
    _test_summary,
    export_bundle,
)
from experiments.annotator.tree_simplification.hp_search_workers import (
    Candidate,
    cache_contact_scores,
    contact_directory,
    fit_contact,
    fit_directory,
    fit_selector,
    initialise_cpu_worker,
    load_pools,
    pool_directory,
    prepare_pool,
)
from experiments.annotator.tree_simplification.input_stages import (
    _review_rows,
    _training_records,
)
from experiments.annotator.tree_simplification.input_test import (
    EXPECTED_LEFT_OUT,
    EXPECTED_VIDEOS,
)
from experiments.annotator.tree_simplification.integrate import ranking_metrics
from experiments.annotator.tree_simplification.prepare import (
    prepare_video,
    preprocessing_from_record,
    preprocessing_record,
)

CANDIDATE = Candidate(36, 63, 360, 0.9, 63, 400)
SEEDS = (0, 100, 200, 300, 400)


def cpu_pool(workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context('spawn'), initializer=initialise_cpu_worker,
    )


def finish_jobs(futures: list, stage: str) -> None:
    for completed, future in enumerate(as_completed(futures), 1):
        future.result()
        print(f'{stage}: {completed}/{len(futures)}', flush=True)


def prepare_test_video(video: SearchVideo, paths: dict[str, Path], output: Path, base: BaseAnnotatorConfig) -> None:
    destination = output / 'videos' / f'{video.identity}.joblib'
    if destination.exists():
        return
    inputs = load_test_inputs(video, paths['inpaint_root'], paths['pose_root'], paths['test_courts'])
    pose, court = inputs.pose, inputs.court
    with threadpool_limits(limits=1):
        evidence = prepare_annotation_inputs(
            inputs.track, pose.bboxes, pose.scores, pose.kps, pose.ndet,
            video_config=scale_config_to_fps(base, inputs.fps), resolution=court.resolution,
            court_present=inputs.court_present, homography_rows=court.homography_rows,
            shuttle_grades=inputs.shuttle_grades,
        )
        prepared = PreparedVideo(video, 'test', evidence, contact_features_from_inputs(evidence))
    development.dump_atomically(prepared, destination)


def prepare(paths_file: Path, output: Path, workers: int, evaluation_only: bool) -> None:
    paths = load_paths(paths_file)
    videos = load_original_videos(paths['shots_master'])
    groups = development.development_for(videos).group_by_video
    base = BaseAnnotatorConfig(overrides_base30={'body_unit_half_window': CANDIDATE.body_window})
    inputs = {
        'paths': {key: str(value) for key, value in paths.items()},
        'preprocessing': preprocessing_record(base), 'groups': groups,
    }
    manifest = output / 'development' / 'preparation.json.gz'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists() and read_json(manifest)['inputs'] != inputs:
        raise ValueError('Court input paths or preprocessing changed; choose a fresh prepared directory')
    write_json(manifest, {'inputs': inputs})
    test_videos = load_test_videos(paths['test_inputs'])
    if len(test_videos) != EXPECTED_VIDEOS:
        raise ValueError('The test set requires 46 videos')
    with cpu_pool(workers) as executor:
        futures = [executor.submit(prepare_video, video, groups[video.identity], paths, output / 'development', base)
                   for video in videos if not evaluation_only or video.split == 'validation']
        futures.extend(executor.submit(prepare_test_video, video, paths, output / 'test', base) for video in test_videos)
        finish_jobs(futures, 'prepare')


def evaluate_video(
    prepared_path: Path, model_path: Path, output: Path,
    item: dict[str, Any] | None, windows: list[tuple[int, int]], device: str = 'cpu',
) -> str:
    identity = prepared_path.stem
    destination = output / 'videos' / f'{identity}.json.gz'
    if destination.exists():
        return identity
    destination.parent.mkdir(parents=True, exist_ok=True)
    (output / 'streams').mkdir(parents=True, exist_ok=True)
    prepared = joblib.load(prepared_path)
    models = _models(str(model_path), device)
    prediction = predict_from_contact_features(prepared.evidence, prepared.features, models)
    if item is None:
        stream = stream_payload(prediction.sequences, prediction.events)
        _, review_rows = _review_rows(prepared.video, identity, prediction)
        result = {'counts': score_stream(prepared.video, metrics.contact_endpoint_stream(stream)),
                  'review_rows': review_rows}
        stream['confidence'] = {
            str(sequence.span_id): score
            for sequence, score in zip(prediction.sequences, prediction.rally_confidence, strict=True)
        }
    else:
        result = _score_prediction(item, windows, prediction)
        stream = result.pop('stream')
    stream['source_span_ids'] = list(prediction.source_span_ids)
    write_json(output / 'streams' / f'{identity}.json.gz', stream)
    write_json(destination, result)
    return identity


def validation_summary(videos: tuple[SearchVideo, ...], output: Path, seed: int) -> dict[str, Any]:
    per_video, rows = {}, []
    for video in videos:
        record = read_json(output / 'videos' / f'{video.identity}.json.gz')
        per_video[video.identity] = record['counts']
        rows.extend(record['review_rows'])
    review = {}
    for ranking in ('selector', 'fitted'):
        for tolerance in (5, 10):
            selected = [row for row in rows if row['ranking'] == ranking and row['tolerance'] == tolerance]
            review[f'{ranking}-{tolerance}'] = ranking_metrics(
                np.asarray([row['correct'] for row in selected], dtype=np.int8),
                np.asarray([row['score'] for row in selected], dtype=np.float64),
            )
    totals = development.pooled_scores(per_video)
    if totals['10']['labelled_rallies'] != 668:
        raise ValueError('The fixed validation set requires 668 labelled rallies')
    result = {'complete': True, 'seed': seed, 'per_video': per_video, 'totals': totals, 'review': review}
    write_json(output / 'result.json.gz', result)
    return result


def evaluate(
    paths_file: Path, prepared_root: Path, model_path: Path, output: Path, seed: int, executor: ProcessPoolExecutor,
    device: str = 'cpu',
) -> None:
    paths = load_paths(paths_file)
    inputs = read_json(prepared_root / 'development' / 'preparation.json.gz')['inputs']
    if preprocessing_record(_models(str(model_path), device).preprocessing) != inputs['preprocessing']:
        raise ValueError('Model preprocessing differs from prepared evidence')
    validation = tuple(video for video in load_original_videos(paths['shots_master']) if video.split == 'validation')
    test = load_test_videos(paths['test_inputs'])
    items = {item['item_id']: item for item in load_manifest(paths['test_inputs'])['items']}
    metrics.INPUTS = paths['test_inputs']
    windows = {identity: metrics.left_out_windows(rallies) for identity, rallies in metrics.test_left_out().items()}
    if len(test) != EXPECTED_VIDEOS or sum(map(len, windows.values())) != EXPECTED_LEFT_OUT:
        raise ValueError('The fixed test set requires 46 videos and 524 excluded rally windows')
    settings_path = output / f'seed-{seed}-settings.json.gz'
    output.mkdir(parents=True, exist_ok=True)
    settings = {'models': str(model_path.resolve()), 'prepared': str(prepared_root.resolve()),
                'paths': str(paths_file), 'device': device}
    if settings_path.exists() and read_json(settings_path) != settings:
        raise ValueError('Evaluation models or inputs changed; choose a fresh output directory')
    write_json(settings_path, settings)
    futures = []
    for split, videos in (('validation', validation), ('test', test)):
        directory = output / split / f'seed-{seed}'
        source = 'development' if split == 'validation' else 'test'
        for video in videos:
            futures.append(executor.submit(
                evaluate_video, prepared_root / source / 'videos' / f'{video.identity}.joblib', model_path,
                directory, items[video.identity] if split == 'test' else None, windows.get(video.identity, []), device,
            ))
    finish_jobs(futures, f'evaluate seed {seed}')
    validation_summary(validation, output / 'validation' / f'seed-{seed}', seed)
    _test_summary({'label': '', 'seed': seed}, test, output / 'test')


def refit(
    paths_file: Path, prepared_root: Path, output: Path, seed: int, workers: int, gpu_fits: int, device: str,
) -> None:
    paths = load_paths(paths_file)
    videos = load_original_videos(paths['shots_master'])
    split = development.development_for(videos)
    training_ids = split.id_order(split.videos(development.TRAINING_GROUPS))
    prepared = prepared_root / 'development'
    inputs = read_json(prepared / 'preparation.json.gz')['inputs']
    records = _training_records(prepared, inputs, split)
    config = replace(ContactFitConfig(), max_leaf_nodes=CANDIDATE.contact_leaves, max_iter=CANDIDATE.contact_rounds,
                     random_seed=20260824 + seed, device=device)
    deadline = float('inf')
    names = development.contact_fit_videos(split)
    names.pop('final_40')
    contacts = contact_directory(output, CANDIDATE, seed)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / f'seed-{seed}-settings.json.gz', {'candidate': asdict(CANDIDATE), 'seed': seed,
               'prepared': str(prepared_root.resolve()), 'device': device, 'workers': workers, 'gpu_fits': gpu_fits})
    with threadpool_limits(limits=2), ThreadPoolExecutor(max_workers=gpu_fits) as gpu, cpu_pool(workers) as cpu:
        finish_jobs([gpu.submit(fit_contact, records, order, split.id_order(order), config,
                                contacts / f'{name}.joblib', deadline) for name, order in names.items()], 'contact fits')
        del records
        finish_jobs([cpu.submit(cache_contact_scores, prepared / 'videos' / f'{identity}.joblib',
                                contacts / f'held_{split.group_by_video[identity]}.joblib',
                                contacts / 'scores' / f'{identity}.joblib', deadline)
                     for identity in training_ids], 'contact scores')
        directory = pool_directory(output, CANDIDATE, seed, 'train')
        finish_jobs([cpu.submit(prepare_pool, prepared / 'videos' / f'{identity}.joblib',
                                contacts / 'scores' / f'{identity}.joblib', CANDIDATE.cutoff,
                                directory / f'{identity}.joblib', deadline)
                     for identity in training_ids], 'rally proposals')
        pools = load_pools(directory, training_ids)
        groups = {'full': development.TRAINING_GROUPS}
        groups.update({held: tuple(group for group in development.TRAINING_GROUPS if group != held)
                       for held in development.TRAINING_GROUPS})
        futures = {name: gpu.submit(fit_selector, pools, CANDIDATE, seed, selected, output, device, deadline)
                   for name, selected in groups.items()}
        finish_jobs(list(futures.values()), 'selector fits')
        held_paths = {group: futures[group].result() for group in development.TRAINING_GROUPS}
        bundle_output = output / 'models' / f'seed-{seed}'
        bundle_output.mkdir(parents=True, exist_ok=True)
        bundle = export_bundle(
            contacts / 'v_32.joblib', fit_directory(output, CANDIDATE, seed) / 'fits' / 'ABCD' / 'selector.joblib',
            held_paths, pools, preprocessing_from_record(inputs['preprocessing']), CANDIDATE.cutoff, seed,
            bundle_output, device,
        )
        del pools
        evaluate(paths_file, prepared_root, bundle, output / 'evaluation', seed, cpu, device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    preparation = commands.add_parser('prepare')
    preparation.add_argument('--evaluation-only', action='store_true')
    evaluation = commands.add_parser('evaluate')
    evaluation.add_argument('--models', type=Path, required=True, help='One fitted model bundle')
    fitting = commands.add_parser('refit')
    fitting.add_argument('--gpu-fits', type=int, default=3)
    for command in (preparation, evaluation, fitting):
        command.add_argument('--paths', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--workers', type=int, default=6)
    for command in (evaluation, fitting):
        command.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
        command.add_argument('--prepared', type=Path, required=True)
        command.add_argument('--seed', type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.command == 'prepare':
        prepare(args.paths, args.output, args.workers, args.evaluation_only)
    elif args.command == 'evaluate':
        with cpu_pool(args.workers) as executor:
            evaluate(args.paths, args.prepared, args.models, args.output, args.seed, executor, args.device)
    else:
        if not 1 <= args.gpu_fits <= 3:
            parser.error('--gpu-fits must be between one and three')
        refit(args.paths, args.prepared, args.output, args.seed, args.workers, args.gpu_fits, args.device)


if __name__ == '__main__':
    main()
