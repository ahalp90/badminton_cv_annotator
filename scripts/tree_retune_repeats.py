"""Repeat a frozen training shortlist and select its mean original-ShuttleSet validation score."""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing
import statistics
import sys
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / 'src'))

from experiments.annotator.tree_hp_tune import run as grid

SEED_OFFSETS = (0, 100, 200, 300, 400)
SCHEMA = 'tree-retune-repeats-v1'
ORIGINAL_SETTINGS = grid.DEFAULT_TRAINING_SETTINGS


def seeded_settings(offset: int) -> Any:
    """Change every fitting/sampling seed while retaining all other settings."""
    settings = ORIGINAL_SETTINGS
    return replace(
        settings,
        contact=replace(settings.contact, random_seed=settings.contact.random_seed + offset),
        sequences=replace(
            settings.sequences,
            serve=replace(settings.sequences.serve, random_seed=settings.sequences.serve.random_seed + offset),
            sequence=replace(settings.sequences.sequence, random_seed=settings.sequences.sequence.random_seed + offset),
        ),
        confidence=replace(settings.confidence, random_seed=settings.confidence.random_seed + offset),
    )


def settings_record(settings: Any) -> dict[str, Any]:
    return {name: asdict(getattr(settings, name)) for name in ('contact', 'sequences', 'confidence')}


def seed_root(root: Path, offset: int) -> Path:
    return root / 'grid48' if offset == 0 else root / 'repeats' / f'seed-{offset}'


def freeze_candidates(results: list[dict[str, Any]], specs: list[Any]) -> list[dict[str, Any]]:
    if len(results) != 48 or len(specs) != 48:
        raise ValueError('repeat selection requires all 48 original grid fits')
    by_id = {grid.candidate_id(spec): spec for spec in specs}
    if len(by_id) != 48 or set(by_id) != {row['id'] for row in results}:
        raise ValueError('completed grid identities differ from the candidate document')
    ordered = sorted(results, key=grid.rank_key, reverse=True)
    baseline = next(row for row in ordered if row['label'] == 'shipped')
    chosen = ordered[:4]
    if baseline['id'] not in {row['id'] for row in chosen}:
        chosen.append(baseline)
    candidates = []
    for row in chosen:
        spec = by_id[row['id']]
        candidates.append({
            'id': row['id'], 'label': spec.label,
            'overrides_base30': dict(spec.overrides_base30), 'strategies': dict(spec.strategies),
        })
    return candidates


def verify_snapshot(root: Path, repository: Path) -> tuple[dict[str, Any], list[Any], dict[str, Path]]:
    paths = grid.load_paths(root / 'paths.toml')
    document, specs = grid.load_candidates(repository / 'experiments/annotator/tree_hp_tune/grid-48.json.gz')
    videos = grid.load_original_videos(paths['shots_master'])
    grid.development_for(videos)
    source = grid.source_record()
    source['files_md5'][str(grid.GROUPS_FILE.relative_to(repository))] = hashlib.md5(
        grid.GROUPS_FILE.read_bytes(),
    ).hexdigest()
    files = grid.input_paths(paths, videos, False)
    files['training_groups'] = grid.GROUPS_FILE
    snapshot = {'source': source, 'inputs': grid.file_manifest(files), 'candidates': document}
    snapshot_path = root / 'grid48' / 'snapshot.json.gz'
    if not snapshot_path.exists():
        raise FileNotFoundError('repeat fitting requires an existing original grid snapshot')
    grid.keep_run_identity(snapshot_path, snapshot)
    return grid.read_json(snapshot_path), specs, paths


def freeze_plan(root: Path, repository: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    snapshot, specs, paths = verify_snapshot(root, repository)
    results = grid.completed(root / 'grid48', specs, 'result.json.gz')
    candidates = freeze_candidates(results, specs)
    parity = grid.read_json(root / 'grid48' / 'baseline-parity.json.gz')
    if parity.get('videos') != 32 or parity.get('different_selections') != []:
        raise ValueError('repeat fitting requires a passing 32-video baseline parity check')
    plan = {
        'schema': SCHEMA, 'candidates': candidates, 'seed_offsets': list(SEED_OFFSETS),
        'original_snapshot': snapshot, 'baseline_parity': parity,
        'script_md5': hashlib.md5(Path(__file__).read_bytes()).hexdigest(),
        'seed_settings': {str(offset): settings_record(seeded_settings(offset)) for offset in SEED_OFFSETS},
        'selection': 'mean original 8-video validation attributed_f1 at 10; train only shortlists',
    }
    directory = root / 'repeats'
    directory.mkdir(exist_ok=True)
    path = directory / 'plan.json.gz'
    if path.exists() and grid.read_json(path) != plan:
        raise ValueError('frozen repeat plan changed; resume requires identical source, inputs and repeat code')
    grid.write_json(path, plan)
    sticky = (root / 'grid48' / 'sticky').resolve(strict=True)
    for offset in SEED_OFFSETS[1:]:
        output = seed_root(root, offset)
        output.mkdir(exist_ok=True)
        link = output / 'sticky'
        if link.is_symlink():
            if link.resolve(strict=True) != sticky:
                raise ValueError(f'{link}: sticky cache points outside the verified grid')
        elif link.exists():
            raise ValueError(f'{link}: expected a link to the verified original grid cache')
        else:
            link.symlink_to(sticky, target_is_directory=True)
    # These are the only inputs fit_trial and validate_trial need.
    return plan, {name: paths[name] for name in ('shots_master', 'stages', 'development_courts')}


def spec_for(candidate: dict[str, Any]) -> Any:
    spec = grid.CandidateSpec(candidate['label'], candidate['overrides_base30'], candidate['strategies'])
    if grid.candidate_id(spec) != candidate['id']:
        raise ValueError('frozen candidate identity differs')
    return spec


def result_path(root: Path, candidate: dict[str, Any], offset: int, action: str) -> Path:
    name = 'result.json.gz' if action == 'fit' else 'validation.json.gz'
    return seed_root(root, offset) / 'trials' / candidate['id'] / name


def repeat_job(root: Path, candidate: dict[str, Any], offset: int, paths: dict[str, Path], action: str) -> None:
    spec = spec_for(candidate)
    settings = seeded_settings(offset)
    # Both modules bind their own settings name; fits run in a fresh process.
    grid.DEFAULT_TRAINING_SETTINGS = settings
    grid.retrain.DEFAULT_TRAINING_SETTINGS = settings
    output = seed_root(root, offset)
    if action == 'fit':
        trial = output / 'trials' / candidate['id']
        trial.mkdir(parents=True, exist_ok=True)
        metadata = {'schema': SCHEMA, 'id': candidate['id'], 'seed_offset': offset, 'settings': settings_record(settings)}
        grid.write_json(trial / 'repeat-seed.json.gz', metadata)
        grid.fit_trial(spec, paths, output, 1)
    elif action == 'validate':
        grid.validate_trial(spec, paths, output, 1)
    else:
        raise ValueError(f'unknown repeat action: {action}')


def read_result(root: Path, candidate: dict[str, Any], offset: int, action: str) -> dict[str, Any] | None:
    path = result_path(root, candidate, offset, action)
    if not path.exists():
        return None
    result = grid.read_json(path)
    if result['id'] != candidate['id'] or result['label'] != candidate['label']:
        raise ValueError(f'{path}: result identity differs from repeat plan')
    if offset:
        metadata = grid.read_json(path.parent / 'repeat-seed.json.gz')
        expected = {
            'schema': SCHEMA, 'id': candidate['id'], 'seed_offset': offset,
            'settings': settings_record(seeded_settings(offset)),
        }
        if metadata != expected:
            raise ValueError(f'{path}: repeat seed identity differs')
    return result


def write_progress(root: Path, plan: dict[str, Any], phase: str, **extra: Any) -> None:
    fits, validations = 0, 0
    for candidate in plan['candidates']:
        for offset in SEED_OFFSETS:
            fits += read_result(root, candidate, offset, 'fit') is not None
            validations += read_result(root, candidate, offset, 'validate') is not None
    planned = len(plan['candidates']) * len(SEED_OFFSETS)
    grid.write_json(root / 'repeats' / 'progress.json.gz', {
        'schema': SCHEMA, 'phase': phase, 'fits_complete': fits, 'fits_planned': planned,
        'fresh_fits_complete': fits - len(plan['candidates']),
        'fresh_fits_planned': planned - len(plan['candidates']),
        'validation_complete': validations, 'validation_planned': planned, **extra,
    })


def run_phase(root: Path, plan: dict[str, Any], paths: dict[str, Path], action: str, workers: int) -> None:
    jobs = []
    for candidate in plan['candidates']:
        for offset in SEED_OFFSETS:
            if read_result(root, candidate, offset, action) is None:
                if action == 'fit' and offset == 0:
                    raise ValueError('original grid fit disappeared after freezing')
                jobs.append((candidate, offset))
    phase = 'repeat_training' if action == 'fit' else 'repeat_validation'
    write_progress(root, plan, phase)
    pending = iter(jobs)
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
                future = executor.submit(repeat_job, root, candidate, offset, paths, action)
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
                        root, plan, 'failed', action=action, config_id=candidate['id'], seed_offset=offset,
                        traceback=traceback.format_exc(),
                    )
                    for unfinished in active:
                        unfinished.cancel()
                    raise
                write_progress(root, plan, phase)


def aggregate(results: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    means, spread = {}, {}
    for tolerance in ('5', '10'):
        means[tolerance], spread[tolerance] = {}, {}
        for metric in results[0]['totals'][tolerance]:
            values = [float(result['totals'][tolerance][metric]) for result in results]
            means[tolerance][metric] = statistics.mean(values)
            spread[tolerance][metric] = {
                'sd': statistics.stdev(values), 'min': min(values), 'max': max(values),
            }
    return means, spread


def summary_rank(row: dict[str, Any]) -> tuple[float, float, float, bool]:
    means = row['mean_validation']
    return (
        means['10']['attributed_f1'], means['10']['fully_correct_rallies'],
        means['5']['attributed_f1'], row['label'] == 'shipped',
    )


def summarise(root: Path, plan: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = []
    for candidate in plan['candidates']:
        training, validation, bundles = [], [], []
        for offset in SEED_OFFSETS:
            fit = read_result(root, candidate, offset, 'fit')
            checked = read_result(root, candidate, offset, 'validate')
            if fit is None or checked is None:
                raise ValueError('summary requires all five fits and validation scores for every configuration')
            training.append({'seed_offset': offset, **fit})
            validation.append({'seed_offset': offset, **checked})
            bundle = result_path(root, candidate, offset, 'fit').parent / 'bundle_v32'
            bundles.append({'seed_offset': offset, 'bundle': str(bundle.resolve(strict=True))})
        mean_training, spread_training = aggregate(training)
        mean_validation, spread_validation = aggregate(validation)
        rows.append({
            **candidate, 'bundles': bundles, 'training': training, 'validation': validation,
            'mean_training': mean_training, 'spread_training': spread_training,
            'mean_validation': mean_validation, 'spread_validation': spread_validation,
        })
    ranked = sorted(rows, key=summary_rank, reverse=True)
    baseline = next(row for row in ranked if row['label'] == 'shipped')
    for row in ranked:
        row['paired_validation_delta_from_baseline'] = [
            {
                'seed_offset': offset,
                'attributed_f1_10': result['totals']['10']['attributed_f1'] - before['totals']['10']['attributed_f1'],
            }
            for offset, result, before in zip(SEED_OFFSETS, row['validation'], baseline['validation'], strict=True)
        ]
    summary = {
        'schema': SCHEMA, 'seed_offsets': list(SEED_OFFSETS), 'results': ranked,
        'selection': plan['selection'], 'ranking': [row['id'] for row in ranked],
    }
    winner = ranked[0]
    selected = {
        'schema': SCHEMA, 'config_id': winner['id'], 'label': winner['label'],
        'seed_offsets': list(SEED_OFFSETS), 'bundles': winner['bundles'],
        'baseline': {'config_id': baseline['id'], 'label': baseline['label'], 'bundles': baseline['bundles']},
        'mean_validation': winner['mean_validation'], 'spread_validation': winner['spread_validation'],
        'mean_training': winner['mean_training'], 'selection': plan['selection'],
        'ranking': summary['ranking'], 'results': ranked,
    }
    return summary, selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--repository', type=Path, default=REPOSITORY)
    args = parser.parse_args()
    if not 1 <= args.workers <= 20:
        parser.error('workers must be between one and twenty')
    root, repository = args.run_root.resolve(), args.repository.resolve()
    if repository != REPOSITORY:
        parser.error('--repository must identify the checkout containing this script')
    (root / 'repeats').mkdir(parents=True, exist_ok=True)
    try:
        plan, paths = freeze_plan(root, repository)
        run_phase(root, plan, paths, 'fit', args.workers)
        run_phase(root, plan, paths, 'validate', min(args.workers, 6))
        summary, selected = summarise(root, plan)
        grid.write_json(root / 'repeats' / 'summary.json.gz', summary)
        grid.write_json(root / 'repeats' / 'selected.json.gz', selected)
        write_progress(root, plan, 'complete', selected_id=selected['config_id'])
    except BaseException:
        grid.write_json(root / 'repeats' / 'failure.json.gz', {'schema': SCHEMA, 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
