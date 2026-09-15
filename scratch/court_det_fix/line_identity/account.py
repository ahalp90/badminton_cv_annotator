"""Account for every matcher record of the filter arms against the frozen controls, beside the baseline.

Uses the direction experiment's accounting unchanged: `diagnose_automatic.diagnose` measures each
stage record against the control, and `diagnose_matrix.accounting` flattens it to one row per
view, arm and stage (nearest pooled court before the global cap, the line and paint winners and
their distances to the control, counts per filter). The baseline rows come from the direction
experiment's `e4/accounting.csv.gz`, so the comparison table reads the baseline as that
experiment measured it.

Writes runs/<run>/matcher/accounting.csv.gz, diagnosis.json.gz and comparison.md.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np

from shared import (
    CASE_IDS,
    DIRECTION_AGREEMENT,
    DIRECTION_RUN,
    LABELS,
    PACK_OF,
    PACKS,
    add_helper_paths,
    control_corners,
    load_estimator,
    read,
    write,
)

add_helper_paths()

from diagnose_automatic import diagnose

STAGES = ('results', 'camera_first', 'all_camera')
HERE = Path(__file__).resolve().parent


def direction_experiment_module(name: str):
    """Import a direction-experiment module under its own `common`; this folder's shared.py stays separate."""
    sys.path.insert(0, str(DIRECTION_AGREEMENT))
    try:
        spec = importlib.util.spec_from_file_location(f'direction_agreement_{name}', DIRECTION_AGREEMENT / f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(DIRECTION_AGREEMENT))
    return module


def baseline_rows() -> dict[tuple[str, str], dict]:
    with gzip.open(DIRECTION_RUN / 'e4' / 'accounting.csv.gz', 'rt', newline='') as stream:
        rows = list(csv.DictReader(stream))
    return {(row['case_id'], row['stage']): row for row in rows if row['arm'] == 'B'}


def number(value) -> str:
    if value in (None, '', 'None'):
        return 'none'
    return f'{float(value):.1f}'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--arms', nargs='+', required=True)
    parser.add_argument('--cases', nargs='+', default=list(CASE_IDS))
    args = parser.parse_args()
    matrix = direction_experiment_module('diagnose_matrix')
    run_dir = HERE / 'runs' / args.run / 'matcher'
    baseline = baseline_rows()
    rows, records, missing = [], [], []
    for case_id in args.cases:
        pack = read(PACKS[PACK_OF[case_id]])
        source = next(case for case in pack['cases'] if case['id'] == case_id)
        reference = pack['references'][case_id]
        corners_working, control_record = control_corners(case_id)
        working = tuple(load_estimator(case_id)['working_size'])
        scale = np.asarray([source['dimensions']['width'], source['dimensions']['height']], dtype=float) / working
        control = {**control_record, 'corners_native_px': (corners_working * scale).tolist()}
        given = {'control_corners_px': control['corners_native_px'], 'given_direction_source': control['control_source']}
        for arm in args.arms:
            for stage in STAGES:
                path = run_dir / arm / stage / f'{case_id}.json.gz'
                if not path.exists():
                    rows.append(matrix.missing_row(case_id, arm, stage, control, 'missing', args.run))
                    missing.append((case_id, arm, stage))
                    continue
                record = read(path)
                diagnosis = diagnose(source, reference, record, given)
                row = matrix.accounting(case_id, arm, stage, record, diagnosis, control)
                rows.append(row)
                records.append({'case_id': case_id, 'arm': arm, 'stage': stage, 'record': str(path.relative_to(HERE)),
                                'control': control, 'diagnosis': diagnosis, 'accounting': row})
    write(run_dir / 'diagnosis.json.gz', {'schema': 'line-identity-diagnosis/1', 'run': args.run, 'records': records})
    with gzip.open(run_dir / 'accounting.csv.gz', 'wt', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=matrix.COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # Comparison at the generation stage and the final all-camera stage, baseline beside each arm.
    lines = io.StringIO()
    by_key = {(row['case_id'], row['arm'], row['stage']): row for row in rows}
    for stage in ('results', 'all_camera'):
        lines.write(f'\n## Stage {stage}\n\n')
        header = ['View', 'Control'] + [f'{arm}: nearest pooled / line / paint' for arm in ('baseline', *args.arms)]
        lines.write('| ' + ' | '.join(header) + ' |\n|' + ' --- |' * len(header) + '\n')
        for case_id in args.cases:
            base = baseline[(case_id, stage)]
            cells = [LABELS[case_id], 'approved' if base['visually_approved'] == 'True' else 'manual',
                     f"{number(base['nearest_pre_global_px'])} / {number(base['line_control_working_px'])} / {number(base['paint_control_working_px'])}"]
            for arm in args.arms:
                row = by_key.get((case_id, arm, stage))
                if row is None or row['status'] != 'complete':
                    cells.append('missing' if row is None else row['status'])
                    continue
                cells.append(f"{number(row['nearest_pre_global_px'])} / {number(row['line_control_working_px'])} / {number(row['paint_control_working_px'])}")
            lines.write('| ' + ' | '.join(cells) + ' |\n')
    comparison = ('# Matcher outcomes per arm against the baseline\n\nWorking pixels, maximum corner distance to the '
                  'frozen control. Each cell: nearest pooled court before the global cap / line winner / paint winner. '
                  'Baseline values are the direction experiment\'s accounting of the saved baseline records.\n' + lines.getvalue())
    (run_dir / 'comparison.md').write_text(comparison)
    print(comparison)
    if missing:
        print('missing case-arm stages:', missing)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
