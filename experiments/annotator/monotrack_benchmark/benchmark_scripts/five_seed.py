"""Select MonoTrack on validation rallies, then evaluate all five seeds on test."""
from __future__ import annotations

import argparse
import itertools
import math
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from common import HERE, load_config, read_json, write_json
from monotrack_runner import evaluation_items
from score import score_run

from experiments.annotator.monotrack_benchmark.writeup_scripts.metrics import (
    left_out_windows,
    rates,
    score_stream_for_item,
    test_left_out,
)

SEEDS = (0, 100, 200, 300, 400)
L2_VALUES = (0.0, 1e-6, 1e-4)
TRAINING_SETTINGS = tuple(itertools.product(('kept', 'zeroed'), L2_VALUES))
SETTINGS = tuple(itertools.product(('kept', 'zeroed'), L2_VALUES, ('off', 'on')))


def training_name(arm: str, l2: float) -> str:
    return f'{arm}_l2{l2:g}'


def checkpoint(trials: Path, arm: str, l2: float, seed: int) -> Path:
    directory = trials / training_name(arm, l2) / f'seed-{seed}'
    summary = read_json(directory / 'summary.json.gz')
    expected = {'arm': arm, 'seed': seed, 'dropout': 0, 'l2': l2, 'smoke': False}
    if any(summary[key] != value for key, value in expected.items()):
        raise ValueError(f'{arm}/seed-{seed}: incompatible checkpoint')
    if not math.isfinite(summary['best_val_loss']):
        raise ValueError(f'{arm}/seed-{seed}: non-finite validation loss')
    path = directory / 'hitnet.h5'
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def report_scores(items: list[dict], run: Path, windows: dict[str, list[tuple[int, int]]]) -> dict:
    """Use exactly the report's supplied-rally judgement and pooled contact counts."""
    totals = {}
    by_video = {}
    for item in items:
        result = read_json(run / f'{item["item_id"]}.json.gz')
        if result['item_id'] != item['item_id'] or result['status'] != 'completed':
            raise ValueError(f'{item["item_id"]}: missing or incomplete inference')
        by_video[item['item_id']] = {}
        for tolerance in (10, 5):
            counts, _ = score_stream_for_item(
                item, result['stream'], tolerance, windows[item['item_id']], supplied=True)
            by_video[item['item_id']][str(tolerance)] = counts
    for tolerance in ('10', '5'):
        frame = pd.DataFrame([video[tolerance] for video in by_video.values()])
        counts = frame.sum()
        totals[tolerance] = {'counts': {key: int(value) for key, value in counts.items()},
                             'rates': {key: float(value) for key, value in rates(counts).items()}}
    return {'by_tolerance': totals, 'by_video': by_video}


def select_configuration(candidates: list[dict]) -> dict:
    """Rank five-seed means; exact ties retain the declared setting order."""
    by_setting = {(row['arm'], row['l2'], row['interpolation']): row for row in candidates}
    if len(candidates) != len(SETTINGS) or set(by_setting) != set(SETTINGS):
        raise ValueError('Selection requires the complete shuttle/interpolation grid')
    ranked = []
    for setting in SETTINGS:
        row = by_setting[setting]
        by_seed = {entry['seed']: entry for entry in row['seeds']}
        if len(row['seeds']) != len(SEEDS) or set(by_seed) != set(SEEDS):
            raise ValueError('Selection requires all five distinct seeds')
        means = {}
        for metric in ('fully_correct', 'player_f1'):
            values = [by_seed[seed]['rates'][metric] for seed in SEEDS]
            if not all(math.isfinite(value) for value in values):
                raise ValueError('Selection metrics must be finite')
            means[metric] = sum(values) / len(values)
        ranked.append({**row, 'mean_validation': means})
    winner = max(ranked, key=lambda row: (row['mean_validation']['fully_correct'],
                                         row['mean_validation']['player_f1']))
    return {'arm': winner['arm'], 'l2': winner['l2'], 'interpolation': winner['interpolation'], 'seeds': list(SEEDS),
            'selection_split': 'validation', 'selection_courts': 'monotrack', 'tolerance': 10,
            'primary_metric': 'fully_correct', 'tie_breaker': 'player_f1',
            'mean_validation': winner['mean_validation'], 'candidates': ranked}


def validation_directory(root: Path, arm: str, l2: float, interpolation: str, seed: int) -> Path:
    return root / 'validation' / f'{training_name(arm, l2)}_{interpolation}' / f'seed-{seed}'


def _score_path(output: Path, arm: str, l2: float, interpolation: str, seed: int) -> Path:
    return validation_directory(output, arm, l2, interpolation, seed) / 'report_scores.json.gz'


def _rescore_validation_setting(source: Path, output: Path, items: list[dict[str, Any]], arm: str,
                                l2: float, interpolation: str, seed: int) -> None:
    windows = {item['item_id']: [] for item in items}
    result = report_scores(
        items, validation_directory(source, arm, l2, interpolation, seed), windows,
    )
    write_json(_score_path(output, arm, l2, interpolation, seed), result)


def _load_rescored_candidates(output: Path) -> list[dict[str, Any]]:
    candidates = []
    for arm, l2, interpolation in SETTINGS:
        seeds = []
        for seed in SEEDS:
            scores = read_json(_score_path(output, arm, l2, interpolation, seed))
            seeds.append({'seed': seed, **scores['by_tolerance']['10']})
        candidates.append({'arm': arm, 'l2': l2, 'interpolation': interpolation, 'seeds': seeds})
    return candidates


def rescore_saved_validation(config: dict[str, Any], source: Path, output: Path,
                             parallel: int) -> tuple[dict[str, Any], bool]:
    """Rescore saved validation streams and select a configuration without inference."""
    if not 1 <= parallel <= 28:
        raise ValueError('parallel must be between 1 and 28')
    source, output = source.resolve(), output.resolve()
    if source == output:
        raise ValueError('Rescoring output must be separate from its saved prediction root')
    items = evaluation_items(config, 'validation')
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for arm, l2, interpolation in SETTINGS:
        for seed in SEEDS:
            score_path = _score_path(output, arm, l2, interpolation, seed)
            if not score_path.is_file():
                jobs.append((source, output, items, arm, l2, interpolation, seed))
    if jobs:
        with ProcessPoolExecutor(max_workers=min(parallel, len(jobs))) as pool:
            futures = [pool.submit(_rescore_validation_setting, *job) for job in jobs]
            for future in as_completed(futures):
                future.result()

    selected = select_configuration(_load_rescored_candidates(output))
    original = read_json(source / 'selected.json.gz')
    write_json(output / 'selected.json.gz', selected)
    changed = any(selected[key] != original[key] for key in ('arm', 'l2', 'interpolation'))
    return selected, changed


def run_reselected_test(config_file: Path, config: dict[str, Any], trials: Path, output: Path,
                        selected: dict[str, Any], parallel: int) -> None:
    """Run and score only missing test conditions for the selected setting."""
    items = evaluation_items(config, 'test')
    settings = []
    for court, seed in itertools.product(('monotrack', 'ours'), SEEDS):
        directory = output / 'test' / court / f'seed-{seed}'
        if not directory.exists():
            settings.append((selected['arm'], selected['l2'], selected['interpolation'], court, seed))
            continue
        for item in items:
            if not (directory / f'{item["item_id"]}.json.gz').is_file():
                raise ValueError(f'{directory}: incomplete video output; missing {item["item_id"]}')
    if settings:
        (output / 'logs').mkdir(exist_ok=True)
        run_grid(config_file, config, trials.resolve(), output, 'test', settings, parallel)

    windows = {item: left_out_windows(rallies) for item, rallies in test_left_out().items()}
    for court, seed in itertools.product(('monotrack', 'ours'), SEEDS):
        directory = output / 'test' / court / f'seed-{seed}'
        if not (directory / 'report_scores.json.gz').is_file():
            write_json(directory / 'report_scores.json.gz', report_scores(items, directory, windows))
        if not (directory / 'scores.json.gz').is_file():
            score_run(config['paths']['inputs'], directory)


def reselect_saved_validation(config_file: Path, trials: Path | None, source: Path, output: Path,
                              parallel: int, selection_only: bool) -> tuple[dict[str, Any], bool]:
    """Rescore cached validation predictions and optionally run the selected test setting."""
    config_file = config_file.resolve()
    source, output = source.resolve(), output.resolve()
    config = load_config(config_file)
    selected, changed = rescore_saved_validation(config, source, output, parallel)
    print(f'SELECTED {selected["arm"]}/l2={selected["l2"]:g}/{selected["interpolation"]}: '
          f'{selected["mean_validation"]}', flush=True)
    if not changed:
        print(f'TEST RESULTS reuse saved runs at {source}', flush=True)
    elif selection_only or trials is None:
        print('TEST RESULTS pending; rerun with --trials to test the changed selection', flush=True)
    else:
        run_reselected_test(config_file, config, trials, output, selected, parallel)
        print(f'TEST RESULTS inferred at {output}', flush=True)
    return selected, changed


def infer(config_file: Path, config: dict, trials: Path, output: Path,
          split: str, arm: str, l2: float, interpolation: str, court: str, seed: int) -> Path:
    if split == 'validation':
        directory = output / split / f'{training_name(arm, l2)}_{interpolation}' / f'seed-{seed}'
    else:
        directory = output / split / court / f'seed-{seed}'
    weights = checkpoint(trials, arm, l2, seed)
    command = [config['execution']['monotrack_python'], '-u', str(HERE / 'monotrack_runner.py'),
               '--config', str(config_file), '--split', split, '--arm', arm, '--interpolation', interpolation,
               '--courts', court, '--weights-file', str(weights), '--output', str(directory)]
    log = output / 'logs' / f'{split}-{training_name(arm, l2)}-{interpolation}-{court}-{seed}.log'
    print(f'START {split} {arm}/l2={l2:g}/{interpolation}/{court}/seed-{seed}', flush=True)
    with log.open('w') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print(f'DONE {split} {arm}/l2={l2:g}/{interpolation}/{court}/seed-{seed}', flush=True)
    return directory


def run_grid(config_file: Path, config: dict, trials: Path, output: Path, split: str,
             settings: list[tuple[str, float, str, str, int]], parallel: int) -> None:
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        jobs = [pool.submit(infer, config_file, config, trials, output, split, *setting) for setting in settings]
        for job in as_completed(jobs):
            job.result()


def evaluate(config_file: Path, trials: Path, output: Path, parallel: int) -> None:
    config = load_config(config_file)
    for (arm, l2), seed in itertools.product(TRAINING_SETTINGS, SEEDS):
        checkpoint(trials, arm, l2, seed)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    validation_items = evaluation_items(config, 'validation')
    windows = {item['item_id']: [] for item in validation_items}
    settings = [(arm, l2, interpolation, 'monotrack', seed) for arm, l2, interpolation in SETTINGS for seed in SEEDS]
    run_grid(config_file, config, trials, output, 'validation', settings, parallel)
    candidates = []
    for arm, l2, interpolation in SETTINGS:
        scores = []
        for seed in SEEDS:
            directory = output / 'validation' / f'{training_name(arm, l2)}_{interpolation}' / f'seed-{seed}'
            result = report_scores(validation_items, directory, windows)
            write_json(directory / 'report_scores.json.gz', result)
            scores.append({'seed': seed, **result['by_tolerance']['10']})
        candidates.append({'arm': arm, 'l2': l2, 'interpolation': interpolation, 'seeds': scores})
    selected = select_configuration(candidates)
    write_json(output / 'selected.json.gz', selected)
    print(f'SELECTED {selected["arm"]}/l2={selected["l2"]:g}/{selected["interpolation"]}: '
          f'{selected["mean_validation"]}', flush=True)
    settings = [(selected['arm'], selected['l2'], selected['interpolation'], court, seed)
                for court in ('monotrack', 'ours') for seed in SEEDS]
    run_grid(config_file, config, trials, output, 'test', settings, parallel)
    items = evaluation_items(config, 'test')
    windows = {item: left_out_windows(rallies) for item, rallies in test_left_out().items()}
    for court, seed in itertools.product(('monotrack', 'ours'), SEEDS):
        directory = output / 'test' / court / f'seed-{seed}'
        write_json(directory / 'report_scores.json.gz', report_scores(items, directory, windows))
        score_run(config['paths']['inputs'], directory)
    for (arm, l2), seed in itertools.product(TRAINING_SETTINGS, SEEDS):
        destination = output / 'weights' / training_name(arm, l2) / f'seed-{seed}'
        destination.mkdir(parents=True)
        for name in ('hitnet.h5', 'history.csv.gz', 'summary.json.gz'):
            shutil.copyfile(trials / training_name(arm, l2) / f'seed-{seed}' / name, destination / name)
    (output / 'complete').write_text('Validation selection and ten test runs completed.\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--trials', type=Path, help='ARM_l2VALUE/seed-N training directories')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parallel', type=int, default=16)
    parser.add_argument('--rescore-validation', type=Path,
                        help='saved run root with all validation predictions and its original selection')
    parser.add_argument('--selection-only', action='store_true',
                        help='rescore/select without running test inference; rerun to continue from the saved result')
    args = parser.parse_args()
    if not 1 <= args.parallel <= 28:
        parser.error('--parallel must be between 1 and 28')
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'
    if args.rescore_validation is not None:
        if args.config is None:
            parser.error('--config is required when rescoring saved validation predictions')
        reselect_saved_validation(
            args.config, args.trials, args.rescore_validation, args.output, args.parallel, args.selection_only,
        )
        return
    if args.selection_only:
        parser.error('--selection-only requires --rescore-validation')
    if args.config is None or args.trials is None:
        parser.error('--config and --trials are required for a fresh benchmark run')
    evaluate(args.config.resolve(), args.trials.resolve(), args.output.resolve(), args.parallel)


if __name__ == '__main__':
    main()
