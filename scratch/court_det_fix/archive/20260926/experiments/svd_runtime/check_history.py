"""Compare completed matcher outputs with historical pairs after the batch finishes."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_benchmark import comparable, equal, read, write


def compare_pairs(expected_pairs: list[dict], actual_pairs: list[dict], label: str,
                  drop_counts: bool = False) -> dict:
    counts = {'compared_count': 0, 'exact_count': 0, 'tolerance_only_count': 0, 'mismatches': []}
    for actual in actual_pairs:
        pair_id = actual['pair_id']
        expected = expected_pairs[pair_id]
        expected_value, actual_value = comparable(expected), comparable(actual)
        if drop_counts:
            actual_value.pop('proposed_count', None)
            actual_value.pop('retained_count', None)
        counts['compared_count'] += 1
        if expected_value == actual_value:
            counts['exact_count'] += 1
            continue
        try:
            equal(expected_value, actual_value, f'{label}.pair.{pair_id}')
        except AssertionError as error:
            counts['mismatches'].append({'pair_id': pair_id, 'error': str(error)[:240]})
        else:
            counts['tolerance_only_count'] += 1
    return counts


def compare_result(result: dict, cached: dict) -> dict:
    full_pairs = result['arms']['full16']['pairs']
    shared_pairs = [pair for pair in result['arms']['svd12']['pairs']
                    if pair['status'] != 'skipped_svd_mask']
    historical = compare_pairs(cached['pairs'], full_pairs, f"{result['case_id']}.cache", drop_counts=True)
    shared_history = compare_pairs(full_pairs, shared_pairs, f"{result['case_id']}.shared")
    return {'case_id': result['case_id'], 'repeat': result['repeat'],
            'historical_cache': historical, 'shared_arms': shared_history}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True, help='Completed full-run result directory')
    parser.add_argument('--cache', type=Path, required=True, help='Historical automatic results directory')
    parser.add_argument('--output', type=Path, required=True, help='Gzipped JSON receipt path')
    args = parser.parse_args()

    summary = read(args.results / 'summary.json.gz')
    if summary['smoke'] or summary['max_pairs'] is not None:
        parser.error('--results must name a full run, not a smoke run')
    case_results = []
    for task in summary['tasks']:
        case_id = task['case_id']
        result = read(args.results / Path(task['file']).name)
        if (task['smoke'] or result['case_id'] != case_id or result['repeat'] != task['repeat'] or result['smoke']
                or result['max_pairs'] is not None):
            raise ValueError(f'Expected full result for {case_id}: {task["file"]}')
        cached = read(args.cache / f'{case_id}.json.gz')
        if cached['case_id'] != case_id:
            raise ValueError(f'Historical cache case does not match {case_id}')
        case_results.append(compare_result(result, cached))

    mismatches = sum(len(section['mismatches']) for case in case_results
                     for section in (case['historical_cache'], case['shared_arms']))
    receipt = {'schema': 'svd-matcher-history-check/1', 'results': str(args.results.resolve()),
               'cache': str(args.cache.resolve()), 'cases': case_results, 'mismatch_count': mismatches,
               'passed': mismatches == 0}
    write(args.output, receipt)
    print('historical comparison mismatches', mismatches, 'receipt', args.output, flush=True)
    return 1 if mismatches else 0


if __name__ == '__main__':
    raise SystemExit(main())
