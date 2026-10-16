"""Run the two shuttle comparisons and the five-condition MonoTrack/annotator benchmark."""
from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from common import HERE, load_config, read_json, write_json
from score import score_run

OURS = ('ours_our_courts', 'ours_mono_courts', 'ours_discovery')


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def run_mono(config: dict[str, Any], config_path: Path, output: Path,
             name: str, arm: str, courts: str = 'monotrack', interpolation: str = 'on') -> None:
    run_command([
        config['execution']['monotrack_python'], '-u', str(HERE / 'monotrack_runner.py'),
        '--config', str(config_path), '--arm', arm, '--courts', courts, '--interpolation', interpolation,
        '--output', str(output / name),
    ], output / 'logs' / f'{name}.log')


def summarise(output: Path) -> dict[str, Any]:
    selection = read_json(output / 'shuttle_comparison.json.gz')
    winner = selection['selected_arm']
    runs = {name: read_json(output / name / 'scores.json.gz') for name in
            (*OURS, f'mono_{winner}_own_on', 'mono_our_courts_on', 'mono_kept_own_off', 'mono_zeroed_own_off')}
    summary = {
        'shuttle_comparison': selection,
        'headline': {
            'monotrack_own_courts': runs[f'mono_{winner}_own_on']['by_tolerance'],
            'monotrack_our_courts': runs['mono_our_courts_on']['by_tolerance'],
            **{name: runs[name]['by_tolerance'] for name in OURS},
        },
        'interpolation_off': {arm: runs[f'mono_{arm}_own_off']['by_tolerance'] for arm in ('kept', 'zeroed')},
        'failures': {name: {key: result[key] for key in ('failed_videos', 'failed_rallies')}
                     for name, result in runs.items()},
    }
    write_json(output / 'comparison.json.gz', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'config.local.toml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('all', 'ours', 'monotrack', 'summarise'), default='all')
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.phase in ('all', 'ours'):
        for name in OURS:
            run_command([
                config['execution']['annotator_python'], '-u', str(HERE / 'annotator_runner.py'),
                '--config', str(config_path), '--condition', name, '--output', str(output / name),
            ], output / 'logs' / f'{name}.log')
            score_run(config['paths']['inputs'], output / name)
    if args.phase in ('all', 'monotrack'):
        # The two shuttle arms are judged with the released inference interpolation.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_mono, config, config_path, output, f'mono_{arm}_own_on', arm)
                       for arm in ('kept', 'zeroed')]
            for future in futures:
                future.result()
        results = {arm: score_run(config['paths']['inputs'], output / f'mono_{arm}_own_on')['by_tolerance']
                   for arm in ('kept', 'zeroed')}
        # This test-set choice is deliberate: both results are reported in the footnote.
        winner = max(('kept', 'zeroed'), key=lambda arm: results[arm]['10']['f1'])
        write_json(output / 'shuttle_comparison.json.gz', {
            'results': results, 'selected_arm': winner,
            'selection': 'ShuttleSet22 test contact F1 at +/-10 frames; kept wins an exact tie',
        })
        run_mono(config, config_path, output, 'mono_our_courts_on', winner, courts='ours')
        score_run(config['paths']['inputs'], output / 'mono_our_courts_on')
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_mono, config, config_path, output, f'mono_{arm}_own_off', arm,
                                   interpolation='off') for arm in ('kept', 'zeroed')]
            for future in futures:
                future.result()
        for arm in ('kept', 'zeroed'):
            score_run(config['paths']['inputs'], output / f'mono_{arm}_own_off')
    if args.phase in ('all', 'summarise'):
        summarise(output)
    print(f'Completed {args.phase}', flush=True)


if __name__ == '__main__':
    main()
