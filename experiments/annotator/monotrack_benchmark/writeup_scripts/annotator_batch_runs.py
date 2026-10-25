"""Resume boundary-policy inference while sharing each video's preparation and pools.

A worker prepares one video/court/preprocessing condition, then uses its contact
features for all seeds and contact-training sizes. Each seed's rally options are
built once and scored by all three policies. Saved predictions are retained and,
when another policy needs the same pool, checked against the shared computation.
"""
from __future__ import annotations

import os

os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                  BLIS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2')

import argparse
import multiprocessing
import sys
import time
import tomllib
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from math import isfinite
from pathlib import Path
from typing import Any

sys.path.insert(0, str(REPO := Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(REPO / 'src'))

from threadpoolctl import threadpool_limits

from annotator.config import resolve
from annotator.contacts.features import ContactFeatures
from annotator.hybrid import (
    HybridPrediction,
    features_from_evidence,
    predict_contact_features,
)
from annotator.models import AnnotatorModels, load_models
from annotator.preparation import AnnotationEvidence, prepare_annotation
from annotator.sequence.contacts import stream_from_payload, stream_payload
from annotator.sequence.rally_model import predict_rally_options, score_rally_review
from annotator.training.sequences import RallyBoundaryPolicy
from experiments.annotator.development import pooled_scores
from experiments.annotator.heuristic_only_tune import data
from experiments.annotator.heuristic_only_tune.run import read_json, write_json
from experiments.annotator.monotrack_benchmark.benchmark_scripts.common import (
    load_manifest,
)
from experiments.annotator.monotrack_benchmark.writeup_scripts.annotator_runs import (
    INPUTS,
    SEEDS,
    SUPPLIED_CONDITIONS,
    Run,
    add_paths,
    bundle_path,
    planned_runs,
)
from experiments.annotator.old_court_regression.old_inputs import OldVideoInputs

POLICIES = {'2_3': RallyBoundaryPolicy(2, 3), '3_3': RallyBoundaryPolicy(3, 3), '5_5': RallyBoundaryPolicy(5, 5)}
COURT_FOLDERS = {'our_courts': 'ours', 'monotrack_courts': 'monotrack_for_ours'}


@dataclass(frozen=True)
class BatchPaths:
    training: Path
    models_40: Path
    output: Path
    inputs: dict[str, Path]


@dataclass(frozen=True)
class VideoGroup:
    video: str
    split: str
    condition: str
    preprocessing: str
    runs: tuple[Run, ...]

    @property
    def label(self) -> str:
        return f'{self.split}/{self.condition}/{self.preprocessing}/{self.video}'


def prediction_path(paths: BatchPaths, policy: str, run: Run, video: str) -> Path:
    return run.directory(paths.output / policy) / f'{video}.json.gz'


@lru_cache(maxsize=64)
def cached_models(path: Path) -> AnnotatorModels:
    return load_models(path)


def require_shared_pool(models: tuple[AnnotatorModels, ...]) -> None:
    """Only selectors and review trees may differ when reusing contact options."""
    first = models[0]
    contact = first.contact.get_booster().save_raw()
    for other in models[1:]:
        if (other.preprocessing != first.preprocessing or other.contact_settings != first.contact_settings
                or other.side_geometry != first.side_geometry
                or other.sequences.max_edits != first.sequences.max_edits
                or other.sequences.deletion_candidates != first.sequences.deletion_candidates):
            raise ValueError('Shared rally pools require identical preparation, contact settings and proposal budgets')
        if other.contact.get_booster().save_raw() != contact:
            raise ValueError('Shared rally pools require the same fitted contact tree')


@lru_cache(maxsize=64)
def policy_models(bundle_paths: tuple[Path, ...]) -> tuple[AnnotatorModels, ...]:
    models = tuple(cached_models(path) for path in bundle_paths)
    require_shared_pool(models)
    return models


def predict_shared_pool(
    evidence: AnnotationEvidence, features: ContactFeatures, models: tuple[AnnotatorModels, ...],
) -> tuple[HybridPrediction, ...]:
    """Run each policy's own selector and review against one contact-derived pool."""
    first = predict_contact_features(evidence, features, models[0])
    predictions = [first]
    for model in models[1:]:
        refined = predict_rally_options(model.sequences, first.refined.pool)
        predictions.append(replace(first, refined=refined, confidence=score_rally_review(refined, model.confidence)))
    return tuple(predictions)


def prediction_record(run: Run, video: str, prediction: HybridPrediction) -> dict[str, Any]:
    stream = stream_payload(prediction.refined.sequences, prediction.refined.events)
    if run.condition != 'discovery':
        return {'item_id': video, 'condition': SUPPLIED_CONDITIONS[run.condition], 'status': 'completed',
                'mode': 'full', 'boundaries': 'provided', 'stream': stream, 'seconds': None}
    review = prediction.confidence
    scores = dict(zip(review.features.span_ids, map(float, review.scores), strict=True))
    return stream | {'confidence': {str(sequence.span_id): scores[sequence.span_id]
                                   for sequence in prediction.refined.sequences}}


def read_prediction(path: Path, run: Run, video: str) -> dict[str, Any]:
    """Read complete prediction files; malformed or failed records stop resumption."""
    record = read_json(path)
    if run.condition == 'discovery':
        stream = record
        sequences, _events = stream_from_payload(stream)
        if set(record['confidence']) != {str(sequence.span_id) for sequence in sequences}:
            raise ValueError(f'{path}: confidence scores do not cover the saved rallies')
        if not all(isfinite(value) for value in record['confidence'].values()):
            raise ValueError(f'{path}: non-finite confidence score')
    else:
        expected = {'item_id': video, 'condition': SUPPLIED_CONDITIONS[run.condition], 'status': 'completed',
                    'mode': 'full', 'boundaries': 'provided'}
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError(f'{path}: saved prediction has different metadata or failed annotation')
        stream = record['stream']
        stream_from_payload(stream)
    return record


def comparable_prediction(record: dict[str, Any], run: Run) -> dict[str, Any]:
    if run.condition == 'discovery':
        return {key: record[key] for key in ('sequences', 'events', 'confidence')}
    return record['stream']


def save_or_compare(path: Path, run: Run, video: str, record: dict[str, Any]) -> bool:
    """Keep existing predictions byte-for-byte; publish new gzip JSON atomically."""
    if path.exists():
        saved = read_prediction(path, run, video)
        if comparable_prediction(saved, run) != comparable_prediction(record, run):
            raise ValueError(f'{path}: shared-pool inference differs from the saved prediction')
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, record)
    return True


@lru_cache(maxsize=2)
def videos_by_split(split: str) -> dict[str, data.SearchVideo]:
    videos = (data.load_test_videos(INPUTS) if split == 'test' else
              data.load_original_videos(REPO / 'training/data/shuttleset/annotations/shots_master.csv'))
    return {video.identity: video for video in videos if split == 'test' or video.split == split}


@lru_cache(maxsize=1)
def test_items() -> dict[str, dict[str, Any]]:
    return {item['item_id']: item for item in load_manifest(INPUTS)['items']}


def prepare_group(group: VideoGroup, paths: BatchPaths, models: AnnotatorModels) -> tuple[AnnotationEvidence, ContactFeatures]:
    video = videos_by_split(group.split)[group.video]
    if group.split == 'validation':
        inputs = data.load_original_inputs(video, paths.inputs['stages'], paths.inputs['development_courts'])
    else:
        court_root = (paths.inputs['test_courts'] if group.condition == 'discovery'
                      else INPUTS / COURT_FOLDERS[group.condition])
        inputs = data.load_test_inputs(video, paths.inputs['inpaint_root'], paths.inputs['pose_root'], court_root)
    spans = None
    if group.condition != 'discovery':
        spans = [(span['start_frame'], span['end_frame']) for span in test_items()[group.video]['spans']]
    evidence = prepare_inputs(inputs, models, spans)
    return evidence, features_from_evidence(evidence)


def prepare_inputs(
    inputs: OldVideoInputs, models: AnnotatorModels, spans: list[tuple[int, int]] | None,
) -> AnnotationEvidence:
    """The preparation part of run_video, retaining its supplied-bound semantics."""
    pose, court = inputs.pose, inputs.court
    return prepare_annotation(
        inputs.track, pose.bboxes, pose.scores, pose.kps, pose.ndet,
        resolved=resolve(models.preprocessing, inputs.fps), resolution=court.resolution,
        video_id=inputs.identity, net_band=court.net_band, gate_court_info=court.gate_court_info,
        gate_resolution_table=court.gate_resolution_table, court_present=inputs.court_present,
        homography_rows=court.homography_rows, cut_frames=[end for _start, end in inputs.raw_cuts[:-1]],
        keep_vote=inputs.keep_vote, inpaint_codes=inputs.guard_codes, court_invalid_is_excluded=True, spans=spans,
    )


def run_group(group: VideoGroup, paths: BatchPaths) -> dict[str, Any]:
    started = time.monotonic()
    complete = {}
    for run in group.runs:
        present = []
        for policy in POLICIES:
            path = prediction_path(paths, policy, run, group.video)
            if path.exists():
                read_prediction(path, run, group.video)
                present.append(policy)
        complete[run] = len(present) == len(POLICIES)
    reused = len(POLICIES) * sum(complete.values())
    created, compared = 0, 0
    pending = [run for run in group.runs if not complete[run]]
    if pending:
        with threadpool_limits(limits=2):
            first_path = bundle_path(pending[0], paths.models_40 / '2_3', paths.training / '2_3')
            baseline = cached_models(first_path)
            evidence, features = prepare_group(group, paths, baseline)
            for run in pending:
                bundle_paths = tuple(bundle_path(run, paths.models_40 / policy, paths.training / policy)
                                     for policy in POLICIES)
                models = policy_models(bundle_paths)
                if models[0].preprocessing != baseline.preprocessing:
                    raise ValueError(f'{group.label}: grouped runs have different preprocessing')
                predictions = predict_shared_pool(evidence, features, models)
                for policy, prediction in zip(POLICIES, predictions, strict=True):
                    path = prediction_path(paths, policy, run, group.video)
                    written = save_or_compare(path, run, group.video, prediction_record(run, group.video, prediction))
                    created += int(written)
                    compared += int(not written)
    return {'group': group.label, 'created': created, 'reused': reused + compared, 'compared': compared,
            'seconds': time.monotonic() - started}


def prepare_run_policy(paths: BatchPaths, policy: str) -> None:
    directory = paths.output / policy
    receipt = directory / 'run-policy.json.gz'
    expected = {'tree_results': str((paths.training / policy).resolve()),
                'models_40': str((paths.models_40 / policy).resolve()), 'boundary_policy': asdict(POLICIES[policy])}
    if receipt.exists():
        if read_json(receipt) != expected:
            raise ValueError(f'{receipt}: model paths or evaluation policy differ from the saved run')
    else:
        if directory.exists() and any(directory.glob('**/*.json.gz')):
            raise ValueError(f'{directory}: predictions exist without their run-policy receipt')
        directory.mkdir(parents=True, exist_ok=True)
        write_json(receipt, expected)


def grouped_jobs(runs: list[Run], wanted: list[str] | None = None) -> list[VideoGroup]:
    groups: dict[tuple[str, str, str], list[Run]] = defaultdict(list)
    for run in runs:
        preprocessing = 'body_window_36' if run.family == 'body_window_36' else 'raw_baseline'
        groups[run.split, run.condition, preprocessing].append(run)
    jobs = []
    for (split, condition, preprocessing), grouped_runs in groups.items():
        for video in videos_by_split(split):
            if wanted is None or video in wanted:
                jobs.append(VideoGroup(video, split, condition, preprocessing, tuple(grouped_runs)))
    return jobs


def finish_saved_run(task: tuple[Run, str, BatchPaths]) -> None:
    run, policy, paths = task
    directory = run.directory(paths.output / policy)
    videos = videos_by_split(run.split)
    execution = directory / 'execution.json.gz'
    saved_records = read_json(execution)['records'] if execution.exists() else []
    by_video = {record['video']: record for record in saved_records}
    if set(by_video) - set(videos):
        raise ValueError(f'{execution}: saved timings include unexpected videos')
    per_video = {}
    for video, metadata in videos.items():
        record = read_prediction(directory / f'{video}.json.gz', run, video)
        if run.condition == 'discovery':
            per_video[video] = data.score_stream(metadata, record, boundary_policy=POLICIES[policy])
        if video not in by_video:
            by_video[video] = {'video': video, 'seconds': None, 'peak_rss_mib': None}
    if run.condition == 'discovery':
        write_json(directory / 'scores.json.gz', {'by_video': per_video, 'by_tolerance': pooled_scores(per_video)})
    else:
        add_paths()
        from score import score_run
        summary = score_run(INPUTS, directory)
        if summary['failed_videos'] or summary['failed_rallies']:
            raise ValueError(f'{directory}: annotation failures')
    if not execution.exists() or len(saved_records) != len(videos):
        write_json(execution, {'records': [by_video[video] for video in sorted(videos)],
                               'timing_note': 'New predictions share preparation and pools; see batch-execution.json.gz.'})


def run_groups(jobs: list[VideoGroup], paths: BatchPaths, workers: int) -> list[dict[str, Any]]:
    results = []
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'))
    try:
        pending = iter(jobs)
        active = {}
        while True:
            while len(active) < workers:
                group = next(pending, None)
                if group is None:
                    break
                active[executor.submit(run_group, group, paths)] = group
            if not active:
                break
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                group = active.pop(future)
                result = future.result()
                results.append(result)
                print(f"{group.label}: {result['created']} written, {result['reused']} retained "
                      f"({result['compared']} checked), {result['seconds']:.1f}s", flush=True)
    finally:
        # At most one group per worker is submitted, so failure drains only those
        # already running, without starting the rest of the benchmark.
        executor.shutdown(wait=True, cancel_futures=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--models-40', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=14)
    parser.add_argument('--videos', nargs='+')
    parser.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--only', nargs='+', help='run-directory prefixes, e.g. test/discovery/32_videos')
    args = parser.parse_args()
    if not 1 <= args.workers <= 14:
        parser.error('--workers must be between 1 and 14 for two-core trees')
    with args.paths.open('rb') as handle:
        configured = {name: Path(value).expanduser().resolve() for name, value in tomllib.load(handle)['paths'].items()}
    for name in ('stages', 'development_courts', 'test_courts', 'inpaint_root', 'pose_root'):
        if not configured[name].is_dir():
            raise NotADirectoryError(f'{name}: {configured[name]}')
    paths = BatchPaths(args.training_root.resolve(), args.models_40.resolve(), args.out.resolve(), configured)
    runs = [run for run in planned_runs() if run.seed in args.seeds]
    if args.only:
        runs = [run for run in runs if any(str(run.directory(Path())).startswith(prefix) for prefix in args.only)]
    if not runs:
        parser.error('no runs match the requested filters')
    for policy in POLICIES:
        prepare_run_policy(paths, policy)
        for run in runs:
            path = bundle_path(run, paths.models_40 / policy, paths.training / policy)
            if not (path / 'models.joblib').is_file():
                raise FileNotFoundError(path)
    jobs = grouped_jobs(runs, args.videos)
    if not jobs:
        parser.error('no videos match the requested filters')
    print(f'{len(jobs)} video groups; {args.workers} persistent workers, two cores per tree', flush=True)
    results = run_groups(jobs, paths, args.workers)
    write_json(paths.output / 'batch-execution.json.gz', {'groups': results, 'workers': args.workers})
    if args.videos is None:
        tasks = [(run, policy, paths) for policy in POLICIES for run in runs]
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as executor:
            list(executor.map(finish_saved_run, tasks))
    print(f"Complete: {sum(result['created'] for result in results)} written, "
          f"{sum(result['reused'] for result in results)} retained", flush=True)


if __name__ == '__main__':
    main()
