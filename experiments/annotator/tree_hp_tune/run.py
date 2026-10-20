"""Refit a frozen preprocessing grid on original ShuttleSet and score fitted bundles on validation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing
import resource
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Bound inline: ruff's E402 allows sys.path calls above imports, but not assignments.
sys.path.insert(0, str(REPOSITORY := Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(REPOSITORY / 'src'))

# This import sets numerical-library thread defaults before NumPy loads.
from experiments.annotator.heuristic_only_tune.run import (
    file_manifest,
    input_paths,
    keep_run_identity,
    load_candidates,
    load_paths,
    read_json,
    source_record,
    write_json,
)

# isort: split

import numpy as np
from threadpoolctl import threadpool_limits

from annotator.hybrid import features_from_evidence
from annotator.models import AnnotatorModels, SideGeometry, load_models, save_models
from annotator.run_video import RunCapture, run_video
from annotator.training.contact import ContactTrainingVideo, fit_contact_model
from annotator.training.workflow import (
    DEFAULT_TRAINING_SETTINGS,
    Split,
    VideoInput,
    sequence_training_video,
)
from experiments.annotator.heuristic_only_tune.cache import cache_sticky_result
from experiments.annotator.heuristic_only_tune.data import (
    COUNT_FIELDS,
    SearchVideo,
    add_rates,
    load_original_inputs,
    load_original_videos,
    run_configuration,
    run_video_kwargs,
    score_stream,
)
from experiments.annotator.heuristic_tuning.corpus_sweep import candidate_id
from experiments.annotator.heuristic_tuning.sweep import CandidateSpec, _base_and_serve
from experiments.annotator.old_court_regression import retrain

GROUPS_FILE = REPOSITORY / 'scratch/contact_det_full_ds_fit/records/training_video_score_groups.json'
MAX_CORES = 26


def settings_record(settings: Any) -> dict[str, Any]:
    return {name: asdict(getattr(settings, name)) for name in ('contact', 'sequences', 'confidence', 'contact_prediction')}


def check_fit_settings(record: dict[str, Any], settings: Any, bundle: Path) -> None:
    """Recover omitted prediction settings from the fitted bundle itself."""
    recorded = dict(record)
    if 'contact_prediction' not in recorded:
        recorded['contact_prediction'] = asdict(load_models(bundle).contact_settings)
    if recorded != settings_record(settings):
        raise ValueError(f'{bundle}: training settings changed')


def keep_grid_identity(output: Path, snapshot: dict[str, Any]) -> None:
    """Use a frozen repeat plan for legacy grids that did not save fit settings."""
    path = output / 'snapshot.json.gz'
    current = settings_record(DEFAULT_TRAINING_SETTINGS)
    snapshot['settings'] = current
    if path.exists():
        saved = read_json(path)
        recorded = saved.get('settings')
        if recorded is None:
            plan_path = output.parent / 'repeats' / 'plan.json.gz'
            if not plan_path.exists():
                raise ValueError('legacy grid training settings are unknown; use a fresh output directory')
            recorded = read_json(plan_path)['seed_settings']['0']
            # Retain the historical snapshot; the repeat plan is the available evidence.
            snapshot.pop('settings')
        if 'contact_prediction' not in recorded:
            bundles = list((output / 'trials').glob('*/bundle_v32'))
            if not bundles:
                raise ValueError('legacy grid contact prediction settings are unknown; use a fresh output directory')
            for bundle in bundles:
                check_fit_settings(recorded, DEFAULT_TRAINING_SETTINGS, bundle)
            if 'settings' in snapshot:
                snapshot['settings'] = recorded
        elif recorded != current:
            raise ValueError('grid training settings changed; use a fresh output directory')
    keep_run_identity(path, snapshot)


def development_for(videos: tuple[SearchVideo, ...], groups_file: Path = GROUPS_FILE) -> retrain.Development:
    document = json.loads(groups_file.read_text())
    groups = {}
    for group in document['groups']:
        for video in group['videos']:
            identity = video['fixture']
            if identity in groups:
                raise ValueError(f'duplicate group member: {identity}')
            groups[identity] = group['group']
    for identity in document['fixed_validation_videos']:
        if identity in groups:
            raise ValueError(f'validation overlaps training: {identity}')
        groups[identity] = retrain.VALIDATION_GROUP
    by_id = {video.identity: video for video in videos}
    if len(groups) != 40 or set(groups) != set(by_id):
        raise ValueError('groups must match the original 40 development videos')
    for group in (*retrain.TRAINING_GROUPS, retrain.VALIDATION_GROUP):
        if list(groups.values()).count(group) != 8:
            raise ValueError(f'group {group} must have eight videos')
    for identity, group in groups.items():
        expected = 'validation' if group == retrain.VALIDATION_GROUP else 'train'
        if by_id[identity].split != expected or by_id[identity].dataset != 'ShuttleSet':
            raise ValueError(f'{identity}: original training/validation membership differs')
    return retrain.Development(
        groups, {video.identity: video.fps for video in videos},
        {identity: int(identity.removeprefix('sset_')) for identity in groups},
    )


def preprocessing(spec: CandidateSpec) -> Any:
    base, serve = _base_and_serve(spec)
    if serve is not None:
        raise ValueError('serve configurations cannot be preserved in the fitted bundle')
    return base


def pooled_scores(per_video: dict[str, Any]) -> dict[str, Any]:
    totals = {}
    for tolerance in ('5', '10'):
        counts = {name: 0 for name in COUNT_FIELDS}
        for scores in per_video.values():
            for name in COUNT_FIELDS:
                counts[name] += int(scores[tolerance][name])
        totals[tolerance] = add_rates(counts)
    return totals


def rank_key(result: dict[str, Any]) -> tuple[float, float, float]:
    totals = result['totals']
    return (totals['10']['attributed_f1'], totals['10']['fully_correct_rallies'], totals['5']['attributed_f1'])


def progress(root: Path, stage: str, started: float, **details: Any) -> None:
    record = {'stage': stage, 'elapsed_s': time.perf_counter() - started, **details}
    write_json(root / 'progress.json.gz', record)
    print(json.dumps({'trial': root.name, **record}), flush=True)


def fit_trial(spec: CandidateSpec, paths: dict[str, Path], output: Path, threads: int) -> dict[str, Any]:
    source = source_record()
    started, cpu_started = time.perf_counter(), time.process_time()
    identity = candidate_id(spec)
    root = output / 'trials' / identity
    root.mkdir(parents=True, exist_ok=True)
    (root / 'training_streams').mkdir(exist_ok=True)
    with threadpool_limits(limits=threads):
        videos = load_original_videos(paths['shots_master'])
        development = development_for(videos)
        by_id = {video.identity: video for video in videos}
        training_ids = development.videos(retrain.TRAINING_GROUPS)
        base = preprocessing(spec)
        evidence_by_id, features_by_id, contact_training = {}, {}, {}
        timings = {}
        stage_started = time.perf_counter()
        for video_id in development.id_order(training_ids):
            video = by_id[video_id]
            inputs = load_original_inputs(video, paths['stages'], paths['development_courts'])
            capture = RunCapture()
            pose = inputs.pose
            with cache_sticky_result(output / 'sticky', video_id):
                run_video(
                    inputs.track, pose.bboxes, pose.scores, pose.kps, pose.ndet,
                    base=base, heuristic_only=True, capture=capture, **run_video_kwargs(inputs),
                )
            evidence = capture.contact_evidence
            if evidence is None:
                raise ValueError(f'{video_id}: missing contact evidence')
            features = features_from_evidence(evidence)
            evidence_by_id[video_id], features_by_id[video_id] = evidence, features
            frames = np.asarray(sorted(frame for rally in video.rallies for frame in rally.frames), dtype=np.int32)
            contact_training[video_id] = ContactTrainingVideo(video_id, features.rows, frames, video.fps)
            progress(root, 'features', started, video=video_id)
        timings['features_s'] = time.perf_counter() - stage_started
        del inputs, capture, pose, evidence, features
        stage_started = time.perf_counter()
        contacts, fit_records = {}, {}
        for name, sampling_order in retrain.contact_fit_videos(development).items():
            if name == 'final_40':
                continue
            fit_order = development.id_order(sampling_order)
            fit = fit_contact_model(
                [contact_training[video_id] for video_id in sampling_order],
                DEFAULT_TRAINING_SETTINGS.contact, fit_video_order=fit_order,
            )
            contacts[name] = fit.model
            fit_records[name] = {
                'sampling_order': list(sampling_order), 'fit_order': list(fit_order),
                'selected_rows': len(fit.selection.labels), 'positive_rows': int(fit.selection.labels.sum()),
            }
            del fit
            progress(root, 'contact_fit', started, fit=name)
        timings['contacts_s'] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        pools = []
        for video_id in training_ids:
            group = development.group_by_video[video_id]
            pools.append(sequence_training_video(
                VideoInput(video_id, group, Split.TRAIN, root / 'unused-labels.csv'),
                evidence_by_id[video_id], features_by_id[video_id], by_id[video_id].rallies,
                contacts[f'held_{group}'], DEFAULT_TRAINING_SETTINGS.contact_prediction, SideGeometry.VIDEO,
            ))
            progress(root, 'pool', started, video=video_id)
        timings['pools_s'] = time.perf_counter() - stage_started
        del evidence_by_id, features_by_id, contact_training
        gc.collect()
        stage_started = time.perf_counter()
        progress(root, 'downstream_fit', started)
        sequence_fit, confidence, labels = retrain.fit_downstream(pools)
        timings['downstream_s'] = time.perf_counter() - stage_started
        save_models(AnnotatorModels(
            contacts['v_32'], sequence_fit.models, confidence,
            DEFAULT_TRAINING_SETTINGS.contact_prediction, base, SideGeometry.VIDEO,
        ), root / 'bundle_v32')
        per_video = {}
        for video_id in training_ids:
            prediction = sequence_fit.cross_fitted[video_id]
            stream = retrain.stream_payload(prediction.sequences, prediction.events)
            write_json(root / 'training_streams' / f'{video_id}.json.gz', stream)
            per_video[video_id] = score_stream(by_id[video_id], stream)
        result = {
            'source': source,
            'settings': settings_record(DEFAULT_TRAINING_SETTINGS),
            'id': identity, 'label': spec.label, 'per_video': per_video, 'totals': pooled_scores(per_video),
            'contact_fits': fit_records, 'confidence_rows': len(labels), 'timings': timings,
            'wall_s': time.perf_counter() - started, 'cpu_s': time.process_time() - cpu_started,
            'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'threads': threads,
        }
        write_json(root / 'result.json.gz', result)
        progress(root, 'complete', started)
        return result


def validate_trial(spec: CandidateSpec, paths: dict[str, Path], output: Path, threads: int) -> dict[str, Any]:
    source = source_record()
    started, cpu_started = time.perf_counter(), time.process_time()
    identity = candidate_id(spec)
    root = output / 'trials' / identity
    (root / 'validation_streams').mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=threads):
        models = load_models(root / 'bundle_v32')
        if models.contact_settings != DEFAULT_TRAINING_SETTINGS.contact_prediction:
            raise ValueError('contact prediction settings changed')
        per_video = {}
        for video in load_original_videos(paths['shots_master']):
            if video.split != 'validation':
                continue
            inputs = load_original_inputs(video, paths['stages'], paths['development_courts'])
            with cache_sticky_result(output / 'sticky', video.identity):
                stream = run_configuration(inputs, models.preprocessing, models=models)
            write_json(root / 'validation_streams' / f'{video.identity}.json.gz', stream)
            per_video[video.identity] = score_stream(video, stream)
        if len(per_video) != 8:
            raise ValueError('validation must contain exactly eight original videos')
        result = {
            'source': source,
            'settings': settings_record(DEFAULT_TRAINING_SETTINGS),
            'id': identity, 'label': spec.label, 'per_video': per_video, 'totals': pooled_scores(per_video),
            'wall_s': time.perf_counter() - started, 'cpu_s': time.process_time() - cpu_started,
            'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        write_json(root / 'validation.json.gz', result)
        return result


def completed(output: Path, specs: list[CandidateSpec], filename: str) -> list[dict[str, Any]]:
    results = []
    for spec in specs:
        path = output / 'trials' / candidate_id(spec) / filename
        if path.exists():
            row = read_json(path)
            if row['id'] != candidate_id(spec) or row['label'] != spec.label:
                raise ValueError(f'{path}: result identity differs')
            if 'settings' in row:
                check_fit_settings(row['settings'], DEFAULT_TRAINING_SETTINGS, path.parent / 'bundle_v32')
            results.append(row)
    return sorted(results, key=rank_key, reverse=True)


def run_jobs(specs: list[CandidateSpec], paths: dict[str, Path], args: argparse.Namespace) -> None:
    done = {row['id'] for row in completed(args.output, specs, 'result.json.gz')}
    pending = iter(spec for spec in specs if candidate_id(spec) not in done)
    deadline = time.monotonic() + args.hours * 3600
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1,
    ) as executor:
        active = {}
        while True:
            while len(active) < args.workers and time.monotonic() < deadline:
                spec = next(pending, None)
                if spec is None:
                    break
                future = executor.submit(fit_trial, spec, paths, args.output, args.threads)
                active[future] = spec
            if not active:
                break
            finished, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
            for future in finished:
                spec = active.pop(future)
                try:
                    future.result()
                except BaseException:
                    write_json(args.output / 'failure.json.gz', {
                        'id': candidate_id(spec), 'action': 'run', 'traceback': traceback.format_exc(),
                    })
                    for unfinished in active:
                        unfinished.cancel()
                    raise


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--hours', type=float, default=3)
    parser.add_argument('--only', help='run one labelled pilot while retaining the full planned grid')
    return parser


def main() -> None:
    parser = argument_parser()
    args = parser.parse_args()
    if not 1 <= args.workers <= MAX_CORES or args.threads < 1 or args.workers * args.threads > MAX_CORES:
        parser.error('workers and threads must be positive and their product at most 26')
    if args.hours <= 0:
        parser.error('hours must be positive')
    paths = load_paths(args.paths)
    document, specs = load_candidates(args.candidates)
    if len({spec.label for spec in specs}) != len(specs) or sum(spec.label == 'shipped' for spec in specs) != 1:
        raise ValueError('candidate labels must be unique and include exactly one shipped baseline')
    for spec in specs:
        preprocessing(spec)
    videos = load_original_videos(paths['shots_master'])
    development_for(videos)
    source = source_record()
    source['files_md5'][str(GROUPS_FILE.relative_to(REPOSITORY))] = hashlib.md5(GROUPS_FILE.read_bytes()).hexdigest()
    files = input_paths(paths, videos, False)
    files['training_groups'] = GROUPS_FILE
    snapshot = {'source': source, 'inputs': file_manifest(files), 'candidates': document}
    args.output.mkdir(parents=True, exist_ok=True)
    keep_grid_identity(args.output, snapshot)
    chosen = [spec for spec in specs if args.only is None or spec.label == args.only]
    if not chosen:
        raise ValueError(f'unknown pilot label: {args.only}')
    run_jobs(chosen, paths, args)
    results = completed(args.output, specs, 'result.json.gz')
    summary = {'planned': len(specs), 'complete': len(results), 'results': results}
    write_json(args.output / 'training.json.gz', summary)
    print(json.dumps({'planned': len(specs), 'complete': len(results)}), flush=True)


if __name__ == '__main__':
    main()
