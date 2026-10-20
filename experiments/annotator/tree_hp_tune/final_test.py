"""Compare the frozen winner and baseline across all five seeds on the fixed final test set."""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from pathlib import Path
from typing import Any

# Bound inline: ruff's E402 allows sys.path calls above imports, but not assignments.
sys.path.insert(0, str(REPOSITORY := Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(REPOSITORY / 'src'))

from experiments.annotator.heuristic_only_tune.run import resume_identity
from experiments.annotator.tree_hp_tune import repeats

# isort: split

from annotator.models import model_files
from experiments.annotator.heuristic_only_tune.data import (
    load_test_inputs,
    load_test_videos,
)

SCHEMA = 'tree-retune-final-test-v1'
grid = repeats.grid


def frozen_selection(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = grid.read_json(root / 'repeats' / 'selected.json.gz')
    progress = grid.read_json(root / 'repeats' / 'progress.json.gz')
    if progress.get('phase') != 'complete' or progress.get('selected_id') != selected.get('config_id'):
        raise ValueError('final test requires complete, frozen repeat selection')
    if selected.get('schema') != repeats.SCHEMA or selected.get('seed_offsets') != list(repeats.SEED_OFFSETS):
        raise ValueError('final test requires the complete five-seed selection schema')
    ranking = selected['ranking']
    if not ranking or ranking[0] != selected['config_id']:
        raise ValueError('selected configuration differs from the frozen repeat ranking')
    by_id = {row['id']: row for row in selected['results']}
    identities = list(dict.fromkeys((selected['config_id'], selected['baseline']['config_id'])))
    candidates = []
    for identity in identities:
        row = by_id[identity]
        candidate = {key: row[key] for key in ('id', 'label', 'overrides_base30', 'strategies', 'bundles')}
        repeats.spec_for(candidate)
        bundles = candidate['bundles']
        if [bundle['seed_offset'] for bundle in bundles] != list(repeats.SEED_OFFSETS):
            raise ValueError('final test must retain every seed, without choosing an individual fit')
        for bundle in bundles:
            expected = repeats.seed_root(root, bundle['seed_offset']) / 'trials' / identity / 'bundle_v32'
            if Path(bundle['bundle']).resolve(strict=True) != expected.resolve(strict=True):
                raise ValueError('selected bundle differs from its original configuration and seed')
        candidates.append(candidate)
    if selected['bundles'] != by_id[selected['config_id']]['bundles']:
        raise ValueError('winner bundle references differ from frozen repeat results')
    baseline = by_id[selected['baseline']['config_id']]
    if baseline['label'] != 'shipped' or baseline['bundles'] != selected['baseline']['bundles']:
        raise ValueError('baseline bundle references differ from frozen repeat results')
    return selected, candidates


def freeze_plan(root: Path, repository: Path, test_paths_file: Path | None) -> tuple[dict[str, Any], dict[str, Path]]:
    selected, candidates = frozen_selection(root)
    original_snapshot, _, paths = repeats.verify_snapshot(root, repository)
    seed_settings = {str(offset): repeats.settings_record(repeats.seeded_settings(offset)) for offset in repeats.SEED_OFFSETS}
    frozen_settings = grid.read_json(root / 'repeats' / 'plan.json.gz')['seed_settings']
    if frozen_settings != seed_settings:
        raise ValueError('training settings changed since the repeat plan was frozen')
    if test_paths_file is not None:
        test_paths = grid.load_paths(test_paths_file)
        for name in ('test_inputs', 'test_courts', 'inpaint_root', 'pose_root'):
            paths[name] = test_paths[name]
    missing = [name for name in ('test_inputs', 'test_courts', 'inpaint_root', 'pose_root') if name not in paths]
    if missing:
        raise ValueError(f'missing final test paths {missing}; supply --test-paths TOML')
    # Reading test labels and inputs starts only after frozen_selection succeeds.
    videos = load_test_videos(paths['test_inputs'])
    if len(videos) != 46 or len({video.identity for video in videos}) != 46:
        raise ValueError('final test requires the fixed 46 distinct ShuttleSet22 fixtures')
    if any(video.dataset != 'ShuttleSet22' or video.split != 'test' for video in videos):
        raise ValueError('final test population differs from ShuttleSet22 test fixtures')
    bundle_paths = {}
    for candidate in candidates:
        for bundle in candidate['bundles']:
            for name, path in model_files(Path(bundle['bundle'])).items():
                bundle_paths[f"{candidate['id']}/seed-{bundle['seed_offset']}/{name}"] = path
    for candidate in candidates:
        for bundle in candidate['bundles']:
            models = grid.load_models(Path(bundle['bundle']))
            if models.contact_settings != repeats.ORIGINAL_SETTINGS.contact_prediction:
                raise ValueError('contact prediction settings changed')
    bundle_identity = grid.file_manifest(bundle_paths)
    for name, path in bundle_paths.items():
        bundle_identity[name]['md5'] = hashlib.md5(path.read_bytes()).hexdigest()
    plan = {
        'schema': SCHEMA, 'selected': selected, 'candidates': candidates, 'seed_settings': seed_settings,
        'contact_prediction': grid.settings_record(repeats.ORIGINAL_SETTINGS)['contact_prediction'],
        'seed_offsets': list(repeats.SEED_OFFSETS), 'original_snapshot': original_snapshot,
        'test_inputs': grid.file_manifest(grid.input_paths(paths, videos, False)),
        'test_videos': [video.identity for video in videos], 'bundles': bundle_identity,
        'comparison': 'all five paired seeds; final_40 contact refit; frozen v32 downstream; no further selection',
    }
    output = root / 'final_test'
    output.mkdir(exist_ok=True)
    plan_path = output / 'plan.json.gz'
    if plan_path.exists():
        saved = grid.read_json(plan_path)
        # Older final plans inherit settings from the frozen repeat plan checked above.
        saved.setdefault('seed_settings', frozen_settings)
        comparison = dict(plan)
        if 'contact_prediction' not in saved:
            comparison.pop('contact_prediction')
        if resume_identity(saved) != resume_identity(comparison):
            raise ValueError('frozen final test plan changed; resume requires identical selection, bundles and inputs')
    else:
        grid.write_json(plan_path, plan)
    return plan, paths


def trial_root(root: Path, identity: str, offset: int) -> Path:
    return root / 'final_test' / 'trials' / identity / f'seed-{offset}'


def trial_progress(trial: Path, stage: str, **details: Any) -> None:
    grid.write_json(trial / 'progress.json.gz', {'stage': stage, **details})


def fit_final_contact(root: Path, candidate: dict[str, Any], offset: int, paths: dict[str, Path]) -> dict[str, Any]:
    """Refit only the contact tree on original 40, preserving the paired downstream fit."""
    trial = trial_root(root, candidate['id'], offset)
    trial.mkdir(parents=True, exist_ok=True)
    source = next(bundle for bundle in candidate['bundles'] if bundle['seed_offset'] == offset)
    models = grid.load_models(Path(source['bundle']))
    if models.contact_settings != repeats.ORIGINAL_SETTINGS.contact_prediction:
        raise ValueError('contact prediction settings changed')
    spec = repeats.spec_for(candidate)
    if models.preprocessing != grid.preprocessing(spec):
        raise ValueError('selected bundle preprocessing differs from its frozen configuration')
    videos = grid.load_original_videos(paths['shots_master'])
    development = grid.development_for(videos)
    by_id = {video.identity: video for video in videos}
    sampling_order = grid.retrain.contact_fit_videos(development)['final_40']
    fit_order = development.id_order(sampling_order)
    if len(sampling_order) != 40 or set(sampling_order) != set(by_id):
        raise ValueError('final contact tree must use all 40 original development videos')
    training = []
    for video_id in sampling_order:
        video = by_id[video_id]
        inputs = grid.load_original_inputs(video, paths['stages'], paths['development_courts'])
        capture = grid.RunCapture()
        pose = inputs.pose
        with grid.cache_sticky_result(root / 'grid48' / 'sticky', video_id):
            grid.run_video(
                inputs.track, pose.bboxes, pose.scores, pose.kps, pose.ndet,
                base=models.preprocessing, heuristic_only=True, capture=capture, **grid.run_video_kwargs(inputs),
            )
        evidence = capture.contact_evidence
        if evidence is None:
            raise ValueError(f'{video_id}: missing final-contact features')
        features = grid.features_from_evidence(evidence)
        frames = grid.np.asarray(sorted(frame for rally in video.rallies for frame in rally.frames), dtype=grid.np.int32)
        training.append(grid.ContactTrainingVideo(video_id, features.rows, frames, video.fps))
        trial_progress(trial, 'features', video=video_id, complete=len(training), planned=40)
    settings = repeats.seeded_settings(offset)
    trial_progress(trial, 'contact_fit', seed_offset=offset)
    fit = grid.fit_contact_model(training, settings.contact, fit_video_order=fit_order)
    grid.save_models(replace(models, contact=fit.model), trial / 'bundle_final40')
    record = {
        'seed_offset': offset, 'source_bundle': source['bundle'],
        'sampling_order': list(sampling_order), 'fit_order': list(fit_order),
        'selected_rows': len(fit.selection.labels), 'positive_rows': int(fit.selection.labels.sum()),
        'contact_settings': repeats.settings_record(settings)['contact'],
    }
    grid.write_json(trial / 'contact-fit.json.gz', record)
    return record


def final_job(root: Path, candidate: dict[str, Any], offset: int, paths: dict[str, Path], threads: int = 1) -> None:
    source = grid.source_record()
    started = time.perf_counter()
    trial = trial_root(root, candidate['id'], offset)
    trial.mkdir(parents=True, exist_ok=True)
    with grid.threadpool_limits(limits=threads):
        if not (trial / 'contact-fit.json.gz').exists():
            fit_final_contact(root, candidate, offset, paths)
        contact_fit = grid.read_json(trial / 'contact-fit.json.gz')
        expected_contact = repeats.settings_record(repeats.seeded_settings(offset))['contact']
        if contact_fit['contact_settings'] != expected_contact:
            raise ValueError('final contact training settings changed')
        models = grid.load_models(trial / 'bundle_final40')
        videos = load_test_videos(paths['test_inputs'])
        if len(videos) != 46:
            raise ValueError('final test must contain exactly 46 fixtures')
        (trial / 'test_streams').mkdir(exist_ok=True)
        per_video = {}
        for video in videos:
            stream_path = trial / 'test_streams' / f'{video.identity}.json.gz'
            if stream_path.exists():
                stream = grid.read_json(stream_path)
            else:
                inputs = load_test_inputs(video, paths['inpaint_root'], paths['pose_root'], paths['test_courts'])
                with grid.cache_sticky_result(root / 'final_test' / 'sticky', video.identity):
                    stream = grid.run_configuration(inputs, models.preprocessing, models=models)
                grid.write_json(stream_path, stream)
            per_video[video.identity] = grid.score_stream(video, stream)
            trial_progress(trial, 'test', video=video.identity, complete=len(per_video), planned=46)
        result = {
            'source': source,
            'settings': grid.settings_record(repeats.seeded_settings(offset)),
            'schema': SCHEMA, 'id': candidate['id'], 'label': candidate['label'], 'seed_offset': offset,
            'per_video': per_video, 'totals': grid.pooled_scores(per_video),
            'bundle': str((trial / 'bundle_final40').resolve()),
            'contact_fit': contact_fit, 'wall_s': time.perf_counter() - started,
            'threads': threads,
        }
        grid.write_json(trial / 'result.json.gz', result)
        trial_progress(trial, 'complete', complete=46, planned=46)


def completed(root: Path, plan: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for candidate in plan['candidates']:
        for offset in repeats.SEED_OFFSETS:
            path = trial_root(root, candidate['id'], offset) / 'result.json.gz'
            if path.exists():
                row = grid.read_json(path)
                if (row['id'], row['label'], row['seed_offset']) != (candidate['id'], candidate['label'], offset):
                    raise ValueError(f'{path}: final test result identity differs')
                settings = repeats.seeded_settings(offset)
                if row['contact_fit']['contact_settings'] != repeats.settings_record(settings)['contact']:
                    raise ValueError(f'{path}: final contact training settings changed')
                if 'settings' in row:
                    grid.check_fit_settings(row['settings'], settings, path.parent / 'bundle_final40')
                if set(row['per_video']) != set(plan['test_videos']):
                    raise ValueError(f'{path}: final test result population differs')
                results.append(row)
    return results


def write_progress(root: Path, plan: dict[str, Any], phase: str, **extra: Any) -> None:
    grid.write_json(root / 'final_test' / 'progress.json.gz', {
        'schema': SCHEMA, 'phase': phase, 'complete': len(completed(root, plan)),
        'planned': len(plan['candidates']) * len(repeats.SEED_OFFSETS), **extra,
    })


def run_jobs(root: Path, plan: dict[str, Any], paths: dict[str, Path], workers: int, threads: int = 1) -> None:
    done = {(row['id'], row['seed_offset']) for row in completed(root, plan)}
    jobs = [(candidate, offset) for candidate in plan['candidates'] for offset in repeats.SEED_OFFSETS
            if (candidate['id'], offset) not in done]
    pending = iter(jobs)
    write_progress(root, plan, 'final_test')
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1,
    ) as executor:
        active = {}
        while True:
            while len(active) < workers:
                job = next(pending, None)
                if job is None:
                    break
                candidate, offset = job
                future = executor.submit(final_job, root, candidate, offset, paths, threads)
                active[future] = job
            if not active:
                break
            finished, _ = wait(active, timeout=5, return_when=FIRST_COMPLETED)
            for future in finished:
                candidate, offset = active.pop(future)
                try:
                    future.result()
                except BaseException:
                    write_progress(
                        root, plan, 'failed', config_id=candidate['id'], seed_offset=offset,
                        traceback=traceback.format_exc(),
                    )
                    for unfinished in active:
                        unfinished.cancel()
                    raise
                write_progress(root, plan, 'final_test')


def summarise(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    results = completed(root, plan)
    if len(results) != len(plan['candidates']) * len(repeats.SEED_OFFSETS):
        raise ValueError('final summary requires every frozen configuration and seed')
    rows = []
    for candidate in plan['candidates']:
        scores = [row for row in results if row['id'] == candidate['id']]
        means, spread = repeats.aggregate(scores)
        rows.append({**candidate, 'seeds': scores, 'mean_test': means, 'spread_test': spread})
    by_id = {row['id']: row for row in rows}
    selected = plan['selected']
    winner, baseline = by_id[selected['config_id']], by_id[selected['baseline']['config_id']]
    deltas = []
    for after, before in zip(winner['seeds'], baseline['seeds'], strict=True):
        if after['seed_offset'] != before['seed_offset']:
            raise ValueError('final test seed pairs differ')
        delta = {}
        for tolerance in ('5', '10'):
            delta[tolerance] = {
                metric: float(value) - float(before['totals'][tolerance][metric])
                for metric, value in after['totals'][tolerance].items()
            }
        deltas.append({'seed_offset': after['seed_offset'], 'totals': delta})
    mean_delta, spread_delta = repeats.aggregate(deltas)
    return {
        'schema': SCHEMA, 'config_id': winner['id'], 'label': winner['label'],
        'baseline_id': baseline['id'], 'seed_offsets': list(repeats.SEED_OFFSETS), 'results': rows,
        'paired_deltas': deltas, 'mean_paired_delta': mean_delta, 'spread_paired_delta': spread_delta,
        'comparison': plan['comparison'],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--repository', type=Path, default=REPOSITORY)
    parser.add_argument('--test-paths', type=Path, help='TOML supplying the four final test input roots')
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.workers <= 20:
        parser.error('workers must be between one and twenty')
    if args.threads < 1 or args.workers * args.threads > repeats.CPU_BUDGET:
        parser.error(f'workers times threads must be at most {repeats.CPU_BUDGET}')
    root, repository = args.run_root.resolve(), args.repository.resolve()
    if repository != REPOSITORY:
        parser.error('--repository must identify the checkout containing this script')
    (root / 'final_test').mkdir(parents=True, exist_ok=True)
    try:
        plan, paths = freeze_plan(root, repository, args.test_paths)
        run_jobs(root, plan, paths, args.workers, args.threads)
        grid.write_json(root / 'final_test' / 'summary.json.gz', summarise(root, plan))
        write_progress(root, plan, 'complete')
    except BaseException:
        grid.write_json(root / 'final_test' / 'failure.json.gz', {'schema': SCHEMA, 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
