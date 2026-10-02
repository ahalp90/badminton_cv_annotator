"""Explore four untuned rankings on frozen B directions and cached B fits.

Run inspect_cached.py first. This does not run the matcher, optimize a court,
substitute SVD-refitted directions, or change any source record.
"""
from pathlib import Path
import gzip
import json
import numpy as np

BASE = Path(__file__).resolve().parent
BUDGETS = (4, 6, 8, 10, 12, 14, 16)


def read(path: Path) -> dict:
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def main() -> None:
    data = read(BASE / 'cached_extension.json.gz')
    records_dir = BASE / 'task3_inputs/scratch/court_det_fix/direction_agreement/runs/direction_agreement_20260915_144900/e3'
    metrics = {
        'svd_rms': lambda g: (g['svd_rms'], -g['support_count'], g['group']),
        'support_only': lambda g: (-g['support_count'], g['group']),
        'svd_gap_desc': lambda g: (-g['normalized_gap'], -g['support_count'], g['group']),
        'sigma3_over_sigma2': lambda g: (
            g['singular_values'][2] / max(g['singular_values'][1], np.finfo(float).tiny),
            -g['support_count'], g['group']),
    }
    results = []
    for name, key in metrics.items():
        for budget in BUDGETS:
            cases = []
            for case in data['cases']:
                fits = read(records_dir / (case['case'] + '.json.gz'))['sets']['B']['fits']
                order = [g['group'] for g in sorted(case['groups'], key=key)]
                kept = set(order[:budget])
                surviving = [p for p in fits['records']
                             if all(g in kept for g in p['groups'])
                             and p.get('max_corner_working_px') is not None
                             and np.isfinite(p['max_corner_working_px'])]
                if not surviving:
                    raise ValueError(f"No finite cached fit for {case['case']}, {name}, {budget}")
                best = min(surviving, key=lambda p: p['max_corner_working_px'])
                cases.append({
                    'case': case['case'], 'approved': case['approved'],
                    'kept_original_best': all(g in kept for g in case['B_best_groups']),
                    'error_delta_px': best['max_corner_working_px'] - case['B_best_px'],
                    'kept_group_ids': order[:budget],
                    'baseline_best_groups': case['B_best_groups'],
                    'baseline_best_ranks': [order.index(g) + 1 for g in case['B_best_groups']],
                    'best_surviving_pair': best['pair_id'],
                })
            result = {
                'metric': name, 'k': budget, 'n_pairs': budget * (budget - 1),
                'kept_best': sum(c['kept_original_best'] for c in cases),
                'max_delta_px': max(c['error_delta_px'] for c in cases), 'cases': cases,
            }
            if budget == 16:
                assert result['kept_best'] == 9 and result['max_delta_px'] == 0
            if name == 'svd_rms' and budget in (8, 10, 12, 14, 16):
                assert result['kept_best'] == {8: 2, 10: 3, 12: 9, 14: 9, 16: 9}[budget]
            results.append(result)
            print(f"{name:23s} k={budget:2d} retained={result['kept_best']}/9 "
                  f"worst_delta={result['max_delta_px']:.6f} px")
    output = {
        'caveat': 'Exploratory comparison on the same nine development views; no detector run, '
                  'independent validation, or deployment recommendation. Four predefined metrics '
                  'without numerical threshold tuning. Ranking uses no controls; evaluation does.',
        'results': results,
    }
    with gzip.open(BASE / 'exploratory_rankings.json.gz', 'wt') as stream:
        json.dump(output, stream, allow_nan=False, indent=2)


if __name__ == '__main__':
    main()
