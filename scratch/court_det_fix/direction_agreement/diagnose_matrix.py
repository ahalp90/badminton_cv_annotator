"""E4 accounting: control-selected diagnostics for B, M and R at every selection stage.

The saved diagnose_automatic.diagnose is the accounting reference; its control input is
the frozen control for each case. E3's ordered-pair fitting is reported separately so
direction-fit potential stays distinct from candidate availability and ranking.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    ARMS,
    BASELINE_STAGES,
    CASE_IDS,
    MATCHER_ARMS,
    SAVED_ESTIMATORS,
    STAGES,
    code_md5,
    frozen_control,
    load_sources,
    native_size,
    read,
    run_dir,
    write,
    write_csv,
)
from diagnose_automatic import diagnose

REPORTED_ARMS = ('B', *MATCHER_ARMS)
BASELINE_RUN = 'automatic_axes_20260914'
COLUMNS = [
    'run', 'arm', 'case_id', 'stage', 'selection_stage', 'status', 'identity_reused', 'control_source', 'visually_approved',
    'pairs_attempted', 'camera_bound_rejected', 'pairs_matched', 'generated', 'geometry_valid', 'geometry_players',
    'per_pair_retained', 'pooled', 'final', 'final_camera_eligible', 'final_floor_pass', 'final_paint_available',
    'line_winner_id', 'paint_winner_id', 'line_control_working_px', 'paint_control_working_px',
    'line_reference_display_px', 'paint_reference_display_px', 'line_floor_score', 'paint_floor_score',
    'nearest_pre_global_id', 'nearest_pre_global_px', 'nearest_pre_global_camera_eligible_id',
    'nearest_pre_global_camera_eligible_px', 'nearest_final_id', 'nearest_final_px',
    'nearest_final_camera_eligible_id', 'nearest_final_camera_eligible_px', 'generation_elapsed_s', 'stage_elapsed_s',
]


def nearest_fields(entry: dict | None) -> tuple[str | None, float | None]:
    return (None, None) if entry is None else (entry['candidate_id'], entry['max_corner_px'])


def accounting(case_id: str, arm: str, stage: str, record: dict, diagnosis: dict, control: dict) -> dict:
    matched = [pair for pair in record['pairs'] if pair['status'] == 'matched']
    entries = record['entries']
    limit = record['camera_error_limit']
    camera_eligible = [entry for entry in entries
                       if entry['gates']['camera_error'] is not None and entry['gates']['camera_error'] <= limit]
    floor_pass = 0
    paint_available = 0
    for entry in entries:
        floor_pass += entry['gates']['floor_score'] is not None and entry['gates']['floor_score'] >= 0
        paint_available += entry['profile']['score'] is not None
    winners = diagnosis['winners']
    row = {
        'run': record.get('run', BASELINE_RUN), 'arm': arm, 'case_id': case_id, 'stage': stage,
        'selection_stage': record.get('selection_stage'), 'status': 'diagnosed',
        'identity_reused': 'identity_reused_from' in record,
        'control_source': control['control_source'], 'visually_approved': control['visually_approved'],
        'pairs_attempted': len(record['pairs']), 'camera_bound_rejected': diagnosis['counts']['camera_direction_rejected'],
        'pairs_matched': len(matched),
        'generated': sum(pair['role']['combined'] for pair in matched),
        'geometry_valid': sum(pair['role']['geometry_valid'] for pair in matched),
        'geometry_players': sum(pair['role']['geometry_players'] for pair in matched),
        'per_pair_retained': sum(len(pair['shortlist']) for pair in matched),
        'pooled': diagnosis['counts']['pooled'], 'final': len(entries), 'final_camera_eligible': len(camera_eligible),
        'final_floor_pass': int(floor_pass), 'final_paint_available': int(paint_available),
        'line_winner_id': record['line_winner_id'], 'paint_winner_id': record['paint_winner_id'],
        'generation_elapsed_s': record.get('generation_elapsed_s', record['elapsed_s']), 'stage_elapsed_s': record['elapsed_s'],
    }
    for name in ('line', 'paint'):
        winner = winners[name]
        row[f'{name}_control_working_px'] = None if winner is None else winner['control_working']
        row[f'{name}_reference_display_px'] = None if winner is None else winner['reference_display']
        row[f'{name}_floor_score'] = None if winner is None else winner['gates']['floor_score']
    for prefix, key in (('nearest_pre_global', 'before_global_cap'),
                        ('nearest_pre_global_camera_eligible', 'before_global_camera_eligible'),
                        ('nearest_final', 'after_global_cap'), ('nearest_final_camera_eligible', 'after_camera_check')):
        row[f'{prefix}_id'], row[f'{prefix}_px'] = nearest_fields(diagnosis[key])
    return row


def missing_row(case_id: str, arm: str, stage: str, control: dict, status: str, run: str) -> dict:
    row = dict.fromkeys(COLUMNS)
    row.update({'run': BASELINE_RUN if arm == 'B' else run, 'case_id': case_id, 'arm': arm, 'stage': stage,
                'status': status, 'control_source': control['control_source'],
                'visually_approved': control['visually_approved']})
    return row


def potential(e3: dict) -> dict:
    """E3 best finite ordered-pair fits per set; a separate metric from any generated court."""
    result = {}
    for arm in ARMS:
        for name in (arm, f'{arm}_svd'):
            summary = e3['sets'][name].get('summary')
            result[name] = None if summary is None else summary['best_finite']
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--run', required=True)
    parser.add_argument('--ids', nargs='+', default=list(CASE_IDS))
    parser.add_argument('--allow-missing', action='store_true',
                        help='exit 0 even when a case-arm stage record is absent (partial matrix)')
    args = parser.parse_args()
    sources, references = load_sources(args.root)
    output = run_dir(args.root, args.run)
    records = []
    rows = []
    missing = []
    for case_id in args.ids:
        source = sources[case_id]
        saved = read(args.root / SAVED_ESTIMATORS / f'{case_id}.json.gz')
        control = frozen_control(case_id, args.root, native_size(source), tuple(saved['working_size']))
        given = {'control_corners_px': control['corners_native_px'], 'given_direction_source': control['control_source']}
        e3_path = output / 'e3' / f'{case_id}.json.gz'
        fit_potential = potential(read(e3_path)) if e3_path.exists() else None
        for arm in REPORTED_ARMS:
            for stage in STAGES:
                path = (args.root / BASELINE_STAGES[stage] / f'{case_id}.json.gz' if arm == 'B'
                        else output / 'e4' / arm / stage / f'{case_id}.json.gz')
                if not path.exists():
                    rows.append(missing_row(case_id, arm, stage, control, 'missing', args.run))
                    missing.append((case_id, arm, stage))
                    continue
                record = read(path)
                if 'court_result' in record:
                    rows.append(missing_row(case_id, arm, stage, control, 'empty_' + record['court_result']['reason'], args.run))
                    continue
                diagnosis = diagnose(source, references[case_id], record, given)
                row = accounting(case_id, arm, stage, record, diagnosis, control)
                rows.append(row)
                records.append({'case_id': case_id, 'arm': arm, 'stage': stage, 'record': str(path),
                                'control': control, 'diagnosis': diagnosis, 'accounting': row})
                print(case_id, arm, stage, {key: row[key] for key in ('final', 'final_camera_eligible', 'line_winner_id',
                                                                      'paint_winner_id', 'line_control_working_px',
                                                                      'paint_control_working_px', 'nearest_final_px')},
                      flush=True)
        records.append({'case_id': case_id, 'arm': None, 'stage': 'e3_direction_fit_potential', 'control': control,
                        'potential': fit_potential})
    write(output / 'e4' / 'diagnosis.json.gz', {'schema': 'direction-agreement-e4-diagnosis/1', 'run': args.run,
                                                 'code_md5': code_md5(args.root), 'label_guided_diagnostics': True,
                                                 'records': records})
    write_csv(output / 'e4' / 'accounting.csv.gz', COLUMNS, [[row[column] for column in COLUMNS] for row in rows])
    if missing:
        print('missing case-arm stages:', missing, flush=True)
        if not args.allow_missing:
            raise SystemExit(f'{len(missing)} case-arm stage records are missing; the matrix is incomplete')


if __name__ == '__main__':
    main()
