"""Package the saved discovery predictions for the core benchmark scorer.

The core benchmark judges whole rallies inside supplied ±1-second windows. This
wraps the frozen heuristic's discovered-boundary test predictions in the
benchmark's per-video result format, without rerunning inference. Score the
output with the benchmark's ``score.py``. Contact counts stay the same; only
the rally judgement changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.annotator.monotrack_benchmark.benchmark_scripts.common import (
    load_manifest,
    read_json,
    write_json,
)

HERE = Path(__file__).resolve().parent
PREDICTIONS = HERE / 'results' / 'test' / 'predictions.json.gz'
INPUTS = HERE.parent / 'monotrack_benchmark' / 'benchmark_results' / 'inputs'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='new directory for the per-video results')
    args = parser.parse_args()
    predictions = read_json(PREDICTIONS)
    items = load_manifest(INPUTS)['items']
    if set(predictions['streams']) != {item['item_id'] for item in items}:
        raise ValueError('saved predictions do not cover the benchmark videos')
    args.output.mkdir(parents=True, exist_ok=False)
    for item in items:
        identity = item['item_id']
        write_json(args.output / f'{identity}.json.gz', {
            'item_id': identity, 'condition': 'ours_discovery', 'mode': 'heuristic',
            'status': 'completed', 'boundaries': 'discovered',
            'candidate_id': predictions['candidate_id'], 'spec': predictions['spec'],
            'stream': predictions['streams'][identity],
        })


if __name__ == '__main__':
    main()
