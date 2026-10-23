"""Run the annotator comparison; use five_seed.py for validation-selected MonoTrack."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from common import HERE, load_config
from score import score_run

OURS = ('ours_our_courts', 'ours_mono_courts', 'ours_discovery')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'config.local.toml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('ours',), default='ours')
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    output = args.output.resolve()
    (output / 'logs').mkdir(parents=True, exist_ok=True)
    for name in OURS:
        command = [config['execution']['annotator_python'], '-u', str(HERE / 'annotator_runner.py'),
                   '--config', str(config_path), '--condition', name, '--output', str(output / name)]
        with (output / 'logs' / f'{name}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        score_run(config['paths']['inputs'], output / name)
    print('Completed annotator comparison', flush=True)


if __name__ == '__main__':
    main()
