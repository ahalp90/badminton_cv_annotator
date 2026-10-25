"""Compare consumed selector targets with aligned proposal rows across boundary policies."""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(REPO := Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(REPO / 'src'))

from dataset_builder.vision import load_npy_xz
from experiments.annotator.heuristic_only_tune.run import read_json, write_json
from experiments.annotator.tree_simplification.pools import original_options

POLICIES = ('2_3', '3_3', '5_5')


def compare_arrays(left: np.ndarray, right: np.ndarray, spans: np.ndarray) -> dict[str, int]:
    if left.shape != right.shape or left.shape != spans.shape:
        raise ValueError('Targets and proposal span IDs must align')
    if not np.isin(left, (-1, 0, 1)).all() or not np.isin(right, (-1, 0, 1)).all():
        raise ValueError('Unexpected target label')
    transition_ids = (left.astype(np.int64) + 1) * 3 + right.astype(np.int64) + 1
    transitions = np.bincount(transition_ids, minlength=9).reshape(3, 3)
    result = {'rows': len(left), 'changed': int(np.count_nonzero(left != right))}
    for before in (-1, 0, 1):
        for after in (-1, 0, 1):
            result[f'{before}_to_{after}'] = int(transitions[before + 1, after + 1])
    left_positive = set(spans[left == 1])
    right_positive = set(spans[right == 1])
    result.update(
        spans=len(np.unique(spans)), left_positive_spans=len(left_positive),
        right_positive_spans=len(right_positive),
        lost_positive_spans=len(left_positive - right_positive),
        gained_positive_spans=len(right_positive - left_positive),
    )
    return result


def compare_video(roots: dict[str, Path], settings: dict[str, Any], identity: str) -> list[dict[str, Any]]:
    pools, targets = {}, {}
    for policy, root in roots.items():
        pool = joblib.load(root / 'pools/videos' / f'{identity}.joblib')
        if settings[policy]['original_options']:
            pool = original_options(pool)
        pools[policy] = pool
        targets[policy] = load_npy_xz(root / 'selector/targets' / f'{identity}.npy.xz')
    reference = pools[POLICIES[0]]
    for policy in POLICIES[1:]:
        if reference.sequences != pools[policy].sequences:
            raise ValueError(f'{identity}: policy proposal rows differ')
        if not np.array_equal(reference.matrix, pools[policy].matrix, equal_nan=True):
            raise ValueError(f'{identity}: policy proposal features differ')
    spans = np.array([sequence.span_id for sequence in reference.sequences])
    return [{'left': left, 'right': right, **compare_arrays(targets[left], targets[right], spans)}
            for left, right in combinations(POLICIES, 2)]


def compare_group(training: Path, out: Path, candidate: str, seed: int) -> list[dict[str, Any]]:
    roots = {policy: training / policy / 'test-run/candidates' / candidate / f'seed-{seed}'
             for policy in POLICIES}
    settings = {policy: read_json(root / 'selector/settings.json.gz') for policy, root in roots.items()}
    counts = read_json(roots[POLICIES[0]] / 'selector/pool-counts.json.gz')
    if any(set(counts) != set(read_json(root / 'selector/pool-counts.json.gz')) for root in roots.values()):
        raise ValueError('Policy fits use different videos')
    checkpoint_root = out / 'per_video' / candidate / f'seed-{seed}'
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    records = []
    with threadpool_limits(limits=1):
        for index, (identity, count) in enumerate(counts.items(), start=1):
            checkpoint = checkpoint_root / f'{identity}.json.gz'
            if checkpoint.exists():
                rows = read_json(checkpoint)
            else:
                rows = []
                for comparison in compare_video(roots, settings, identity):
                    rows.append({'candidate': candidate, 'seed': seed, 'video': identity,
                                 'split': count['split'], **comparison})
                write_json(checkpoint, rows)
            records.extend(rows)
            print(f'{candidate}/seed-{seed}: {index}/{len(counts)} videos', flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=10, choices=range(1, 29), metavar='1..28')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    settings_path = args.out / 'comparison-settings.json.gz'
    settings = {'training': str(args.training.resolve()), 'policies': list(POLICIES)}
    if settings_path.exists() and read_json(settings_path) != settings:
        raise ValueError('Output directory belongs to another training run; use a fresh directory')
    write_json(settings_path, settings)
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(compare_group, args.training, args.out, candidate, seed)
                   for candidate in ('raw_baseline', 'body_window_36') for seed in (0, 100, 200, 300, 400)]
        for future in futures:
            records.extend(future.result())
    data = pd.DataFrame(records)
    data.to_csv(args.out / 'target_differences_per_video.csv.gz', index=False)
    groups = ['candidate', 'seed', 'split', 'left', 'right']
    summary = data.drop(columns='video').groupby(groups, as_index=False).sum(numeric_only=True)
    summary.to_csv(args.out / 'target_differences.csv.gz', index=False)
    write_json(args.out / 'target_comparison.json.gz', {
        'policies': POLICIES, 'targets': 'actual consumed selector targets',
        'alignment': 'identical proposal sequences and feature arrays', 'rows': summary.to_dict('records'),
    })
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
