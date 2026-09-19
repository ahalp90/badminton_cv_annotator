#!/usr/bin/env python3
"""Recalculate selected W5 numerical witnesses. Does not run the court detector.

Inputs contain selected real-data fields transcribed from pinned GitHub connector
reads. The original paths, revision, and source distinctions are in the input.
Uses Python's standard library only. Run from any directory:
  python numerical_checks.py --input inputs/review_inputs.json --out results
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Sequence


def corner_error(corners: Sequence[Sequence[float]], reference: Sequence[Sequence[float]]) -> float:
    if len(corners) != 4 or len(reference) != 4:
        raise ValueError('Exactly four corners are required')
    if not all(len(p) == 2 and all(math.isfinite(float(x)) for x in p) for p in [*corners, *reference]):
        raise ValueError('Corners must be finite XY pairs')
    direct = max(math.dist(a, b) for a, b in zip(corners, reference))
    rotated = max(math.dist(corners[i], reference[(i + 2) % 4]) for i in range(4))
    return min(direct, rotated)


def run(data: dict, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    c2 = data['c2']
    errors = {}
    for name, row in c2['candidates'].items():
        actual = corner_error(row['corners_working_px'], c2['reference_working_px'])
        if not math.isclose(actual, row['reported_error'], rel_tol=0, abs_tol=1e-10):
            raise AssertionError(f'C2 witness mismatch: {name}: {actual}')
        errors[name] = actual
    c2_result = {
        'control_kind': c2['control_kind'],
        'errors_working_px': errors,
        'pair_143_proxy_change_filtered_minus_baseline_px': errors['filtered_pair_143'] - errors['baseline_pair_143'],
        'all_pair_shortlists_nearest_change_filtered_minus_baseline_px': errors['filtered_full_pool'] - errors['baseline_full_pool'],
    }
    l1_rows = []
    for row in data['l1']:
        delta = row['diversified_nearest_shortlist_px'] - row['original_nearest_shortlist_px']
        l1_rows.append({**row, 'change_px': delta, 'direction': 'improved' if delta < -1e-6 else 'worsened' if delta > 1e-6 else 'unchanged'})
    l1_result = {
        'targeted_pairs_not_independent_views': len(l1_rows),
        'improved': sum(x['direction'] == 'improved' for x in l1_rows),
        'worsened': sum(x['direction'] == 'worsened' for x in l1_rows),
        'unchanged': sum(x['direction'] == 'unchanged' for x in l1_rows),
        'retained_cartesian_products_per_arm_per_pair': 512 * 512,
        'rows': l1_rows,
    }
    l2_rows = []
    for row in data['l2_union']:
        l2_rows.append({**row,
            'line_minus_nearest_px': row['line_error_px'] - row['nearest_error_px'],
            'paint_minus_nearest_px': row['paint_error_px'] - row['nearest_error_px']})
    l3_rows = []
    for name, row in data['l3_selected_courts'].items():
        if any(len(row[k]) != 7 for k in ['line', 'forward', 'reverse', 'paint', 'reference_error_px']):
            raise ValueError('L3 witness requires all seven frames')
        score_residual = max(abs((f + r) / 2 - s) for f, r, s in zip(row['forward'], row['reverse'], row['line']))
        if score_residual > 1e-14:
            raise AssertionError('L3 forward/reverse score equation mismatch')
        l3_rows.append({'candidate_id': name,
                       'median_line': median(row['line']),
                       'median_forward': median(row['forward']),
                       'median_reverse': median(row['reverse']),
                       'median_paint': median(row['paint']),
                       'median_reference_error_px': median(row['reference_error_px']),
                       'max_score_equation_residual': score_residual})
    bad = data['l3_selected_courts']['106:93818']
    comparisons = {}
    for name in ('22:4579', '22:4580'):
        row = data['l3_selected_courts'][name]
        comparisons[name] = {
            'frames_with_greater_reverse_than_106_93818': sum(a > b for a, b in zip(row['reverse'], bad['reverse'])),
            'frames_with_greater_paint_than_106_93818': sum(a > b for a, b in zip(row['paint'], bad['paint'])),
            'frames_with_greater_line_than_106_93818': sum(a > b for a, b in zip(row['line'], bad['line'])),
        }
    result = {
        'scope': 'Selected real-data arithmetic only; no pixel inspection, no upstream execution, no complete L3-panel reranking.',
        'revision': data['revision'], 'working_size': [960, 540],
        'c2': c2_result, 'l1': l1_result, 'l2_union': l2_rows,
        'l3_selected_courts': l3_rows, 'l3_pairwise_checks': comparisons,
    }
    (out / 'numerical_checks.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for name, rows in [('l1_changes', l1_rows), ('l2_selection_gaps', l2_rows), ('l3_selected_court_medians', l3_rows)]:
        with (out / f'{name}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    return result


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=here / 'inputs/review_inputs.json')
    parser.add_argument('--out', type=Path, default=here / 'results')
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    print(json.dumps(run(data, args.out), indent=2, allow_nan=False))

if __name__ == '__main__':
    main()
