"""Build the fact packet: compact per-case tables from every stage record of one run.

Reads only the committed records under ``runs/<run>`` and writes ``summary.json.gz`` and
``summary.md`` beside them. Nothing is recomputed from raw geometry.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
from pathlib import Path

import numpy as np
from common import ARMS, CASE_IDS, read, write

SET_NAMES = tuple(name for arm in ARMS for name in (arm, f'{arm}_svd'))
SHORT_CASE = {
    'gxBQ_window_00_frame_0': 'GX0', 'gxBQ_window_00_frame_5': 'GX5', 'am2_window_00_frame_150': 'Am2-150',
    'am2_window_01_frame_28019': 'Am2-28019', 'am3_window_00_frame_0': 'Am3-0', 'shuttleset_03_scene_0017': 'SS03-17',
    'shuttleset_03_scene_0019': 'SS03-19', 'shuttleset_03_scene_0016': 'SS03-16', 'shuttleset_21_scene_0020': 'SS21-20',
}


def table(header: list[str], rows: list[list], align_right_from: int = 1) -> str:
    """Markdown table; numeric columns right-aligned from the given column index."""
    alignment = ['---' if index < align_right_from else '---:' for index in range(len(header))]
    lines = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(alignment) + ' |']
    for row in rows:
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f'{value:.3f}')
            elif value is None:
                cells.append('')
            else:
                cells.append(str(value))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def membership_rows(base: Path) -> list[dict]:
    rows = []
    for case_id in CASE_IDS:
        e0 = read(base / 'e0' / f'{case_id}.json.gz')
        e1 = read(base / 'e1' / f'{case_id}.json.gz')
        used = e0['membership'][:e0['direction_lines_used']]
        sizes = [len(entry['member_observation_ids']) for entry in used]
        summary = e1['summary']
        rows.append({
            'case_id': case_id, 'case': SHORT_CASE[case_id], 'lines_used': e0['direction_lines_used'],
            'merged_groups': e0['merged_lines_total'], 'merge_input_fragments': e0['merge_input_fragments'],
            'single_fragment_lines': int(sum(size == 1 for size in sizes)), 'members_median': float(np.median(sizes)),
            'members_max': int(max(sizes)), 'candidates': e1['candidate_count'],
            'finite_candidates': len(e1['finite_candidate_ids']),
            'membership_changed': summary['membership_entries_changed'],
            'membership_changed_pct': 100. * summary['membership_entries_changed'] / summary['membership_entries_total'],
            'gaining': summary['candidates_gaining_support'], 'losing': summary['candidates_losing_support'],
            'eligible_foot': summary['eligible_foot'], 'eligible_midpoint': summary['eligible_midpoint'],
            'abs_change_median_deg': summary['residual_change_abs_deg']['median'],
            'abs_change_p90_deg': summary['residual_change_abs_deg']['p90'],
            'abs_change_max_deg': summary['residual_change_abs_deg']['max'],
            'displacement_median_px': summary['anchor_displacement_px']['median'],
            'displacement_max_px': summary['anchor_displacement_px']['max'],
            'anchor_length_min_px': summary['anchor_fragment_length_px']['min'],
            'anchor_length_median_px': summary['anchor_fragment_length_px']['median'],
            'top_change_near_foot': int(sum(record['residual_foot_deg'] > 85 for record in e1['largest_residual_changes'])),
        })
    return rows


def allocation_rows(base: Path) -> list[dict]:
    rows = []
    for case_id in CASE_IDS:
        arms = read(base / 'e2' / f'{case_id}.json.gz')['arms']
        baseline_leaders = set(arms['B']['leader_candidate_ids'])
        for arm in ARMS:
            record = arms[arm]
            rows.append({
                'case_id': case_id, 'case': SHORT_CASE[case_id], 'arm': arm, 'directions': record['direction_count'],
                'leaders_shared_with_B': len(baseline_leaders & set(record['leader_candidate_ids'])),
                'representatives_changed': record['representatives_differ_from_leaders'],
                'support_min': min(record['support_counts']), 'support_median': float(np.median(record['support_counts'])),
                'bucket_median': float(np.median([group['bucket_size'] for group in record['groups']])),
                'bucket_max': max(group['bucket_size'] for group in record['groups']),
                'redundant': record['allocation_status_counts'].get('redundant', 0),
                'capped': record['allocation_status_counts'].get('capped', 0),
            })
    return rows


def fit_rows(base: Path) -> list[dict]:
    rows = []
    for case_id in CASE_IDS:
        e3 = read(base / 'e3' / f'{case_id}.json.gz')
        row = {'case_id': case_id, 'case': SHORT_CASE[case_id], 'control_source': e3['control']['control_source'],
               'visually_approved': e3['control']['visually_approved'],
               'replay_passed': None if e3['replay_against_svd_fixed'] is None else e3['replay_against_svd_fixed']['passed']}
        for name in SET_NAMES:
            record = e3['sets'][name]
            if record['status'] != 'fitted':
                row[name] = None
                row[f'{name}_pair'] = None
                continue
            best = record['summary']['best_finite']
            converged = record['summary']['best_converged']
            row[name] = best['max_corner_working_px']
            row[f'{name}_pair'] = best['pair_id']
            row[f'{name}_best_converged'] = converged['max_corner_working_px']
            row[f'{name}_best_finite_converged'] = bool(best['converged']) and best['pair_id'] == converged['pair_id']
            row[f'{name}_failed'] = record['summary']['failed']
            row[f'{name}_nonconverged'] = record['summary']['attempted'] - record['summary']['converged']
            row[f'{name}_min_separation_deg'] = min(pair['angle_deg'] for pair in record['pair_separations'])
        rows.append(row)
    return rows


def matcher_rows(base: Path, allow_partial: bool) -> list[dict] | None:
    """Typed accounting rows; a partial matrix is refused unless explicitly allowed."""
    path = base / 'e4' / 'accounting.csv.gz'
    if not path.exists():
        return None
    with gzip.open(path, 'rt') as stream:
        rows = [{key: number(value) for key, value in row.items()} for row in csv.DictReader(io.StringIO(stream.read()))]
    expected = len(CASE_IDS) * 3 * 3
    complete = len(rows) == expected and all(row['status'] == 'diagnosed' for row in rows)
    if not complete and not allow_partial:
        raise SystemExit(f'accounting has {len(rows)} rows, expected {expected} all diagnosed; pass --allow-partial to summarise anyway')
    return rows


def number(value: str) -> float | int | bool | str | None:
    """CSV cells back to the types the accounting wrote."""
    if value in ('', 'None'):
        return None
    if value in ('True', 'False'):
        return value == 'True'
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def markdown(membership: list[dict], allocation: list[dict], fits: list[dict], matcher: list[dict] | None) -> str:
    parts = ['# Fact packet', '', '## E0-E1 membership and anchors', '',
             table(['case', 'lines', 'groups', 'single-fragment lines', 'candidates', 'membership entries changed',
                    '% changed', 'gain', 'lose', 'eligible foot', 'eligible midpoint', 'median |Δ| deg', 'p90 |Δ| deg',
                    'max |Δ| deg', 'median shift px', 'max shift px', 'shortest anchor px'],
                   [[row['case'], row['lines_used'], row['merged_groups'], row['single_fragment_lines'], row['candidates'],
                     row['membership_changed'], row['membership_changed_pct'], row['gaining'], row['losing'],
                     row['eligible_foot'], row['eligible_midpoint'], row['abs_change_median_deg'], row['abs_change_p90_deg'],
                     row['abs_change_max_deg'], row['displacement_median_px'], row['displacement_max_px'],
                     row['anchor_length_min_px']] for row in membership]),
             '', '## E2 allocation and representatives', '',
             table(['case', 'arm', 'directions', 'leaders shared with B', 'representatives changed', 'min support',
                    'median support', 'median bucket', 'max bucket', 'redundant', 'capped'],
                   [[row['case'], row['arm'], row['directions'], row['leaders_shared_with_B'], row['representatives_changed'],
                     row['support_min'], row['support_median'], row['bucket_median'], row['bucket_max'], row['redundant'],
                     row['capped']] for row in allocation], align_right_from=2),
             '', '## E3 best finite control fit per set (max corner error, working px)', '',
             table(['case', 'control approved', *SET_NAMES],
                   [[row['case'], 'yes' if row['visually_approved'] else 'no', *[row[name] for name in SET_NAMES]]
                    for row in fits], align_right_from=2),
             '', 'Best finite pair converged in every set: '
             + str(all(row[f'{name}_best_finite_converged'] for row in fits for name in SET_NAMES)) + '.']
    if matcher is not None:
        columns = ['case_id', 'arm', 'stage', 'status', 'pairs_matched', 'basis_failed', 'pooled', 'final',
                   'final_camera_eligible', 'final_floor_pass', 'line_winner_id', 'paint_winner_id',
                   'line_control_working_px', 'paint_control_working_px', 'nearest_pre_global_px',
                   'nearest_pre_global_camera_eligible_px', 'nearest_final_px', 'nearest_final_camera_eligible_px',
                   'identity_reused']
        rows = [[SHORT_CASE.get(row['case_id'], row['case_id']), *[row[column] for column in columns[1:]]]
                for row in matcher]
        parts.extend(['', '## E4 accounting (from diagnose_matrix)', '', table(columns, rows, align_right_from=4)])
    return '\n'.join(parts) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true', help='summarise an incomplete E4 accounting')
    args = parser.parse_args()
    membership = membership_rows(args.run_dir)
    allocation = allocation_rows(args.run_dir)
    fits = fit_rows(args.run_dir)
    matcher = matcher_rows(args.run_dir, args.allow_partial)
    write(args.run_dir / 'summary.json.gz', {'schema': 'direction-agreement-summary/1', 'membership': membership,
                                              'allocation': allocation, 'fits': fits, 'matcher': matcher})
    (args.run_dir / 'summary.md').write_text(markdown(membership, allocation, fits, matcher), encoding='utf-8')
    print('summary written; matcher rows:', None if matcher is None else len(matcher))


if __name__ == '__main__':
    main()
