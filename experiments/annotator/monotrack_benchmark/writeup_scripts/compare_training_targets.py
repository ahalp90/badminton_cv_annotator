"""Compare consumed selector targets with aligned proposal rows across boundary policies."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(REPO := Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(REPO / 'src'))

from dataset_builder.vision import load_npy_xz
from experiments.annotator.heuristic_only_tune.run import read_json, write_json
from experiments.annotator.tree_simplification.pools import original_options

POLICIES = ('2_3', '3_3', '5_5')


def compare_arrays(left: np.ndarray, right: np.ndarray, spans: np.ndarray) -> dict[str, int]:
    if left.shape != right.shape or left.shape != spans.shape:
        raise ValueError('Targets and proposal span IDs must align')
    transitions = Counter(zip(left.astype(int), right.astype(int), strict=True))
    result = {'rows': len(left), 'changed': int(np.count_nonzero(left != right))}
    for before in (-1, 0, 1):
        for after in (-1, 0, 1):
            result[f'{before}_to_{after}'] = transitions[before, after]
    span_ids = np.unique(spans)
    left_positive = np.array([np.any(left[spans == span] == 1) for span in span_ids])
    right_positive = np.array([np.any(right[spans == span] == 1) for span in span_ids])
    result.update(
        spans=len(span_ids), left_positive_spans=int(left_positive.sum()),
        right_positive_spans=int(right_positive.sum()),
        lost_positive_spans=int((left_positive & ~right_positive).sum()),
        gained_positive_spans=int((~left_positive & right_positive).sum()),
    )
    if sum(transitions.values()) != sum(result[f'{before}_to_{after}']
                                       for before in (-1, 0, 1) for after in (-1, 0, 1)):
        raise ValueError('Unexpected target label')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    records = []
    for candidate in ('raw_baseline', 'body_window_36'):
        for seed in (0, 100, 200, 300, 400):
            roots = {policy: args.training / policy / 'test-run/candidates' / candidate / f'seed-{seed}'
                     for policy in POLICIES}
            settings = {policy: read_json(root / 'selector/settings.json.gz') for policy, root in roots.items()}
            counts = read_json(roots[POLICIES[0]] / 'selector/pool-counts.json.gz')
            if any(set(counts) != set(read_json(root / 'selector/pool-counts.json.gz')) for root in roots.values()):
                raise ValueError('Policy fits use different videos')
            for identity, count in counts.items():
                pools, targets = {}, {}
                for policy, root in roots.items():
                    pool = joblib.load(root / 'pools/videos' / f'{identity}.joblib')
                    if settings[policy]['original_options']:
                        pool = original_options(pool)
                    pools[policy] = pool
                    targets[policy] = load_npy_xz(root / 'selector/targets' / f'{identity}.npy.xz')
                for left, right in combinations(POLICIES, 2):
                    if pools[left].sequences != pools[right].sequences:
                        raise ValueError(f'{candidate}/{seed}/{identity}: policy proposal rows differ')
                    if not np.array_equal(pools[left].matrix, pools[right].matrix, equal_nan=True):
                        raise ValueError(f'{candidate}/{seed}/{identity}: policy proposal features differ')
                    spans = np.array([sequence.span_id for sequence in pools[left].sequences])
                    records.append({'candidate': candidate, 'seed': seed, 'video': identity,
                                    'split': count['split'], 'left': left, 'right': right,
                                    **compare_arrays(targets[left], targets[right], spans)})
    data = pd.DataFrame(records)
    args.out.mkdir(parents=True, exist_ok=True)
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
