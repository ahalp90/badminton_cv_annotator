"""Summarise complete SVD matcher timings without implying end-to-end speedup."""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Any

REPORT_SCHEMA = 'svd-matcher-runtime-report/1'
FULL_PAIR_COUNT = 16 * 15
ARM_NAMES = ('full16', 'svd12')
SKIPPED = 'skipped_svd_mask'
SUM_FIELDS = ('wall_s', 'process_s', 'selected_pairs', 'matched_pairs', 'bound_pairs', 'skipped_pairs',
              'proposed_candidate_count', 'shortlist_count')


def read(path: Path) -> dict[str, Any]:
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def summarise_arm(arm: dict[str, Any]) -> dict[str, Any]:
    pairs = arm['pairs']
    assert len(pairs) == FULL_PAIR_COUNT
    assert [pair['pair_id'] for pair in pairs] == list(range(FULL_PAIR_COUNT))
    counts = Counter(pair['status'] for pair in pairs)
    matched = [pair for pair in pairs if pair['status'] == 'matched']
    return {
        'wall_s': arm['elapsed_s'], 'process_s': arm['process_s'],
        'selected_pairs': FULL_PAIR_COUNT - counts[SKIPPED],
        'matched_pairs': counts['matched'], 'bound_pairs': counts['camera_direction_bound'],
        'skipped_pairs': counts[SKIPPED],
        'proposed_candidate_count': sum(pair['proposed_count'] for pair in matched),
        'shortlist_count': sum(len(pair['shortlist']) for pair in matched),
        'per_pair_shortlist_counts': {
            str(pair['pair_id']): len(pair['shortlist']) if pair['status'] == 'matched' else None
            for pair in pairs
        },
    }


def summarise_case(task: dict[str, Any], case_path: Path) -> dict[str, Any]:
    if not case_path.is_file():
        raise FileNotFoundError(f'Missing per-case result named by summary task: {case_path}')
    result = read(case_path)
    assert result['smoke'] is False and result['max_pairs'] is None and task['smoke'] is False
    assert (result['case_id'], result['repeat']) == (task['case_id'], task['repeat'])

    raw_arms = result['arms']
    arms = {name: summarise_arm(raw_arms[name]) for name in ARM_NAMES}
    full_pairs, svd_pairs = raw_arms['full16']['pairs'], raw_arms['svd12']['pairs']
    shared = [(full, svd) for full, svd in zip(full_pairs, svd_pairs, strict=True) if svd['status'] != SKIPPED]
    full_wall = sum(full['elapsed_s'] for full, _ in shared)
    svd_wall = sum(svd['elapsed_s'] for _, svd in shared)
    full_cpu = sum(full['process_s'] for full, _ in shared)
    svd_cpu = sum(svd['process_s'] for _, svd in shared)
    rank = result['svd_compute_and_rank']
    net_wall = arms['svd12']['wall_s'] + rank['elapsed_s']
    net_cpu = arms['svd12']['process_s'] + rank['process_s']
    return {
        'case_id': result['case_id'], 'repeat': result['repeat'], 'result_file': case_path.name,
        'selected_original_direction_ids': result['selected_original_direction_ids'],
        'historical_cache_check': result['historical_cache_check'], 'arms': arms,
        'svd_compute_and_rank': {'wall_s': rank['elapsed_s'], 'process_s': rank['process_s']},
        'independently_timed_arm_comparison': {
            'svd12_with_rank_wall_s': net_wall, 'svd12_with_rank_process_s': net_cpu,
            'wall_speedup_ratio': arms['full16']['wall_s'] / net_wall,
            'process_speedup_ratio': arms['full16']['process_s'] / net_cpu,
            'wall_saving_s': arms['full16']['wall_s'] - net_wall,
            'process_saving_s': arms['full16']['process_s'] - net_cpu,
        },
        'retained_pair_comparison': {
            'shared_pair_count': len(shared), 'svd12_retained_pair_fraction': len(shared) / FULL_PAIR_COUNT,
            'full16_retained_work_fraction_wall': full_wall / arms['full16']['wall_s'],
            'full16_retained_work_fraction_process': full_cpu / arms['full16']['process_s'],
            'full16_retained_pairs_wall_s': full_wall, 'svd12_retained_pairs_wall_s': svd_wall,
            'full16_retained_pairs_process_s': full_cpu, 'svd12_retained_pairs_process_s': svd_cpu,
            'wall_drift_ratio_svd12_over_full16': svd_wall / full_wall,
            'process_drift_ratio_svd12_over_full16': svd_cpu / full_cpu,
        },
    }


def make_report(input_dir: Path) -> dict[str, Any]:
    summary_path = input_dir / 'summary.json.gz'
    summary = read(summary_path)
    assert summary['smoke'] is False and summary['max_pairs'] is None
    tasks = summary['tasks']
    assert tasks

    cases, seen_tasks, seen_files = [], set(), set()
    repeats_by_case: dict[str, set[int]] = defaultdict(set)
    for task in tasks:
        identity = (task['case_id'], task['repeat'])
        assert identity not in seen_tasks
        seen_tasks.add(identity)
        repeats_by_case[task['case_id']].add(task['repeat'])
        basename = PurePosixPath(task['file'].replace('\\', '/')).name
        assert basename not in seen_files
        seen_files.add(basename)
        cases.append(summarise_case(task, input_dir / basename))
    expected_repeats = set(range(summary['repeats']))
    for case_id, found in repeats_by_case.items():
        assert found == expected_repeats, (case_id, found, expected_repeats)

    totals = {name: {field: sum(case['arms'][name][field] for case in cases) for field in SUM_FIELDS}
              for name in ARM_NAMES}
    rank_wall = sum(case['svd_compute_and_rank']['wall_s'] for case in cases)
    rank_cpu = sum(case['svd_compute_and_rank']['process_s'] for case in cases)
    net_wall, net_cpu = totals['svd12']['wall_s'] + rank_wall, totals['svd12']['process_s'] + rank_cpu
    pair_rows = [case['retained_pair_comparison'] for case in cases]
    shared_count = sum(row['shared_pair_count'] for row in pair_rows)
    full_wall = sum(row['full16_retained_pairs_wall_s'] for row in pair_rows)
    svd_wall = sum(row['svd12_retained_pairs_wall_s'] for row in pair_rows)
    full_cpu = sum(row['full16_retained_pairs_process_s'] for row in pair_rows)
    svd_cpu = sum(row['svd12_retained_pairs_process_s'] for row in pair_rows)
    return {
        'schema': REPORT_SCHEMA,
        'timing_scope': ('Matcher-stage timing only. The svd12 comparison includes SVD ranking; preparation, image loading, '
                         'full court scoring, scene processing and detector timing are excluded.'),
        'smoke': False, 'source_summary_file': summary_path.name, 'workers': summary['workers'],
        'repeats': summary['repeats'], 'case_count': len(repeats_by_case), 'task_count': len(cases),
        'batch_wall_s': summary['batch_wall_s'], 'sum_task_elapsed_s': summary['sum_task_elapsed_s'],
        'aggregates': {
            'full16': totals['full16'], 'svd12': totals['svd12'],
            'svd_compute_and_rank_wall_s': rank_wall, 'svd_compute_and_rank_process_s': rank_cpu,
            'svd12_with_rank_wall_s': net_wall, 'svd12_with_rank_process_s': net_cpu,
            'independently_timed_arm_wall_speedup_ratio': totals['full16']['wall_s'] / net_wall,
            'independently_timed_arm_process_speedup_ratio': totals['full16']['process_s'] / net_cpu,
            'independently_timed_arm_wall_saving_s': totals['full16']['wall_s'] - net_wall,
            'independently_timed_arm_process_saving_s': totals['full16']['process_s'] - net_cpu,
            'median_per_case_independently_timed_arm_wall_speedup_ratio': median(
                case['independently_timed_arm_comparison']['wall_speedup_ratio'] for case in cases
            ),
            'svd12_retained_pair_fraction': shared_count / (FULL_PAIR_COUNT * len(cases)),
            'full16_retained_work_fraction_wall': full_wall / totals['full16']['wall_s'],
            'full16_retained_work_fraction_process': full_cpu / totals['full16']['process_s'],
            'full16_retained_pairs_wall_s': full_wall, 'svd12_retained_pairs_wall_s': svd_wall,
            'full16_retained_pairs_process_s': full_cpu, 'svd12_retained_pairs_process_s': svd_cpu,
            'shared_pair_wall_drift_ratio_svd12_over_full16': svd_wall / full_wall,
            'shared_pair_process_drift_ratio_svd12_over_full16': svd_cpu / full_cpu,
        },
        'cases': cases,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = gzip.compress(json.dumps(report, allow_nan=False, separators=(',', ':')).encode(), mtime=0)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(data)
    temporary.replace(path)


def print_table(report: dict[str, Any]) -> None:
    print('Matcher-stage seconds: W/C = wall/process; svd12 net includes SVD ranking.')
    print('case                             rep     full16 W/C     svd12 W/C      rank W/C   net x  saved W  retain%  drift W/C')
    for case in report['cases']:
        full, svd = case['arms']['full16'], case['arms']['svd12']
        rank, net, retained = (case['svd_compute_and_rank'], case['independently_timed_arm_comparison'],
                               case['retained_pair_comparison'])
        print(f"{case['case_id']:<32} {case['repeat']:>3}"
              f" {full['wall_s']:>7.3f}/{full['process_s']:<7.3f} {svd['wall_s']:>7.3f}/{svd['process_s']:<7.3f}"
              f" {rank['wall_s']:>7.3f}/{rank['process_s']:<7.3f} {net['wall_speedup_ratio']:>6.2f}"
              f" {net['wall_saving_s']:>8.3f} {retained['full16_retained_work_fraction_wall']:>7.1%}"
              f" {retained['wall_drift_ratio_svd12_over_full16']:>5.2f}/"
              f"{retained['process_drift_ratio_svd12_over_full16']:<5.2f}")
    print(f"Batch wall, separate: {report['batch_wall_s']:.3f}s. "
          f"Median independently timed matcher-stage speedup: "
          f"{report['aggregates']['median_per_case_independently_timed_arm_wall_speedup_ratio']:.2f}x.")
    print('Retain% is full16 wall time spent on retained pairs; drift W/C compares shared pairs between arms.')
    print('Selected/matched/bound/skipped pairs and per-pair shortlist counts are in the JSON report.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Directory with summary.json.gz and case files')
    parser.add_argument('--output', type=Path, required=True, help='Output path ending in .json.gz')
    args = parser.parse_args()
    input_dir, output_path = args.input.resolve(), args.output.resolve()
    if not input_dir.is_dir():
        parser.error(f'Input directory does not exist: {input_dir}')
    if not output_path.name.endswith('.json.gz'):
        parser.error('Output path must end in .json.gz')
    report = make_report(input_dir)
    input_paths = {input_dir / 'summary.json.gz'} | {input_dir / case['result_file'] for case in report['cases']}
    if output_path in input_paths:
        parser.error('Output path must not overwrite an input summary or per-case result')
    write_report(output_path, report)
    print_table(report)
    print(f'Wrote {output_path}')


if __name__ == '__main__':
    main()
