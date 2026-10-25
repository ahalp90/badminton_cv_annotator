"""Survey discovery boundaries without repeating full-video contact matching."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
sys.path.insert(0, str(REPO := Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(REPO / 'src'))

import pandas as pd

from annotator.sequence.contacts import scale_frames, stream_from_payload
from annotator.training.sequences import RallyBoundaryPolicy
from experiments.annotator.monotrack_benchmark.benchmark_scripts.common import (
    labels_for_item,
    load_manifest,
    read_json,
)
from experiments.annotator.monotrack_benchmark.writeup_scripts.boundary_comparison import (
    lock_record,
)
from experiments.annotator.monotrack_benchmark.writeup_scripts.metrics import (
    INPUTS,
    SEEDS,
    judge_found_rallies,
    left_out_windows,
    prepare_scoring_stream,
    test_left_out,
)

POLICIES = {'1_1': RallyBoundaryPolicy(1, 1), '2_3': RallyBoundaryPolicy(2, 3)}


def judge_video(task: tuple[dict, Path, list[tuple[int, int]]]) -> dict[str, list[dict]]:
    item, path, windows = task
    stream = read_json(path)
    labels = labels_for_item(item)
    scored, _unmasked, _ignored = prepare_scoring_stream(item, stream, labels, windows, supplied=False)
    sequences, _events = stream_from_payload(scored)
    result = {}
    for name, policy in POLICIES.items():
        judged = judge_found_rallies(sequences, labels, scale_frames(10, item['fps']), fps=item['fps'],
                                    boundary_policy=policy)
        result[name] = [row | {'video': int(item['video_id']), 'confidence': stream['confidence'][str(row['span_id'])]}
                        for row in judged]
    return result


def run_survey(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=REPO)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', nargs=2, action='append', required=True, metavar=('NAME', 'ANNOTATOR_RUNS'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=14)
    args = parser.parse_args()
    items = load_manifest(INPUTS)['items']
    windows = {video: left_out_windows(rallies) for video, rallies in test_left_out().items()}
    commands = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for name, directory in args.run:
            runs = Path(directory)
            destinations = {policy: args.out / name / policy for policy in POLICIES}
            for policy, destination in destinations.items():
                lock_record(destination / 'evaluation-policy.json.gz', {
                    'annotator_runs': str(runs.resolve()), 'boundary_policy': asdict(POLICIES[policy]),
                    'stage': 'discovery before dataset export',
                })
            for seed in SEEDS:
                if all((destination / f'found_rallies_annotator_discovery_seed{seed}.csv.gz').exists()
                       for destination in destinations.values()):
                    continue
                tasks = [(item, runs / 'test/discovery/32_videos' / f'seed-{seed}' / f'{item["item_id"]}.json.gz',
                          windows[item['item_id']]) for item in items]
                judged = list(executor.map(judge_video, tasks))
                for policy, destination in destinations.items():
                    destination.mkdir(parents=True, exist_ok=True)
                    rows = [row for result in judged for row in result[policy]]
                    output = destination / f'found_rallies_annotator_discovery_seed{seed}.csv.gz'
                    temporary = output.with_suffix('.gz.tmp')
                    pd.DataFrame(rows).to_csv(temporary, index=False, compression='gzip')
                    temporary.replace(output)
            for policy, limits in POLICIES.items():
                destination = args.out / name / policy
                if (destination / 'confidence_cutoffs.json.gz').exists():
                    continue
                commands.append([
                    sys.executable, str(Path(__file__).with_name('confidence_cutoffs.py')),
                    '--validation', str(runs / 'validation/32_videos'),
                    '--metrics', str(destination), '--out', str(destination),
                    '--shots-master', str(REPO / 'training/data/shuttleset/annotations/shots_master.csv'),
                    '--boundary-before-seconds', str(limits.before_seconds),
                    '--boundary-after-seconds', str(limits.after_seconds),
                ])
    with ThreadPoolExecutor(max_workers=min(4, args.workers)) as executor:
        list(executor.map(run_survey, commands))


if __name__ == '__main__':
    main()
