"""E3: control-fit diagnostics for B/M/R/MR and their fixed-support SVD refits.

This is the first stage that reads control geometry. It uses the frozen E2 arm records
and never changes them. Fits are least-squares diagnostics against a control, not
generated courts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from common import (
    ARMS,
    CASE_IDS,
    SAVED_ESTIMATORS,
    SVD_RECORDS,
    code_md5,
    frozen_control,
    load_sources,
    md5,
    native_size,
    read,
    run_dir,
    write,
)
from run_svd_fixed import fit_groups, fit_pairs, pair_separations

REPLAY_ATOL = 1e-8
SVD_LINE_NORMALISATION = 'unit_2d_normal_after_coordinate_transform'
SET_NAMES = tuple(name for arm in ARMS for name in (arm, f'{arm}_svd'))


def movement_degrees(original: np.ndarray, refit: np.ndarray) -> list[float]:
    """Signed-invariant angle between each original direction and its SVD refit."""
    unit_original = original / np.linalg.norm(original, axis=1)[:, None]
    unit_refit = refit / np.linalg.norm(refit, axis=1)[:, None]
    cosine = np.clip(np.abs(np.einsum('gi,gi->g', unit_original, unit_refit)), 0., 1.)
    return np.degrees(np.arccos(cosine)).tolist()


def brief(record: dict | None) -> dict | None:
    if record is None:
        return None
    return {'pair_id': record['pair_id'], 'groups': record['groups'],
            'max_corner_working_px': record['max_corner_working_px'], 'converged': record['converged']}


def summarise(fits: dict) -> dict:
    return {'attempted': fits['attempted'], 'failed': fits['failed'], 'converged': fits['converged'],
            'best_finite': brief(fits['best_finite']), 'best_converged': brief(fits['best_converged']),
            'elapsed_s': fits['elapsed_s']}


def replay_against_saved(sets: dict, saved: dict, corners: np.ndarray, control_source: str) -> dict:
    """Require the B and B+SVD fits to reproduce the earlier svd_fixed record within 1e-8 px."""
    np.testing.assert_allclose(saved['control_corners_working_px'], corners, rtol=0, atol=REPLAY_ATOL)
    assert saved['control_source'] == control_source, (saved['control_source'], control_source)
    differences = {}
    for set_name, saved_key in (('B', 'baseline'), ('B_svd', 'fixed_membership_svd')):
        current_records = sets[set_name]['fits']['records']
        saved_records = saved[saved_key]['records']
        assert [record['pair_id'] for record in current_records] == [record['pair_id'] for record in saved_records]
        largest = 0.
        for current, previous in zip(current_records, saved_records, strict=True):
            current_error = current.get('max_corner_working_px')
            previous_error = previous.get('max_corner_working_px')
            assert (current_error is None) == (previous_error is None), (current, previous)
            if current_error is not None:
                largest = max(largest, abs(current_error - previous_error))
        assert largest <= REPLAY_ATOL, (set_name, largest)
        differences[set_name] = largest
    np.testing.assert_allclose(sets['B_svd']['points_working'], saved['refit_points_working'], rtol=0, atol=REPLAY_ATOL)
    return {'atol': REPLAY_ATOL, 'max_abs_difference_px': differences, 'passed': True}


def fit_case(case_id: str, saved: dict, arms_record: dict, control: dict, replay_record: dict | None) -> dict:
    """Fit every ordered pair of every set against the frozen control."""
    estimator = saved['estimator']
    lines = np.asarray(estimator['direction_lines'], dtype=float)
    transform = np.asarray(estimator['normalised_to_working'], dtype=float)
    normalised_lines = lines @ transform
    corners = np.asarray(control['corners_working_px'], dtype=float)
    sets: dict[str, dict] = {}
    timing: dict[str, dict] = {}
    for arm in ARMS:
        arm_record = arms_record['arms'][arm]
        if not arm_record['matcher_eligible']:
            skipped = {'status': 'skipped', 'reason': arm_record['court_result']['reason']}
            sets[arm] = skipped
            sets[f'{arm}_svd'] = dict(skipped)
            continue
        points_working = np.asarray(arm_record['points_working'], dtype=float)
        points_normalised = np.asarray(arm_record['points_normalised'], dtype=float)
        masks = np.asarray(arm_record['support_masks'], dtype=bool)
        svd_started = perf_counter()
        refit_normalised, groups = fit_groups(normalised_lines, masks)
        svd_elapsed = perf_counter() - svd_started
        refit_working = refit_normalised @ transform.T
        for group, moved in zip(groups, movement_degrees(points_normalised, refit_normalised), strict=True):
            group['movement_deg'] = moved
        fits = fit_pairs(points_working, corners)
        fits_svd = fit_pairs(refit_working, corners)
        sets[arm] = {
            'status': 'fitted', 'representative_candidate_ids': arm_record['representative_candidate_ids'],
            'points_working': points_working.tolist(), 'support_counts': arm_record['support_counts'],
            'pair_separations': pair_separations(points_normalised), 'fits': fits, 'summary': summarise(fits),
        }
        sets[f'{arm}_svd'] = {
            'status': 'fitted', 'source_arm': arm, 'svd_line_normalisation': SVD_LINE_NORMALISATION,
            'refit_membership_changed': False, 'refit_uses_control': False,
            'points_working': refit_working.tolist(), 'groups': groups, 'svd_elapsed_s': svd_elapsed,
            'pair_separations': pair_separations(refit_normalised), 'fits': fits_svd, 'summary': summarise(fits_svd),
        }
        timing[arm] = {'svd_s': svd_elapsed, 'fit_s': fits['elapsed_s'], 'fit_svd_s': fits_svd['elapsed_s']}
    baseline_best = sets['B']['summary']['best_finite'] if sets['B']['status'] == 'fitted' else None
    deltas = {}
    for name in SET_NAMES:
        best = sets[name].get('summary', {}).get('best_finite')
        if baseline_best is None or best is None:
            deltas[name] = None
        else:
            deltas[name] = best['max_corner_working_px'] - baseline_best['max_corner_working_px']
    replay = None if replay_record is None else replay_against_saved(sets, replay_record, corners, control['control_source'])
    return {
        'schema': 'direction-agreement-e3/1', 'case_id': case_id, 'label_guided_evaluation': True,
        'automatic_detection': False, 'control': control, 'working_size': saved['working_size'],
        'membership_angle_deg': saved['settings']['angle_deg'], 'sets': sets,
        'best_finite_delta_vs_B_px': deltas, 'replay_against_svd_fixed': replay, 'timing': timing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--run', required=True)
    parser.add_argument('--ids', nargs='+', default=list(CASE_IDS))
    args = parser.parse_args()
    cv2.setNumThreads(1)
    sources, _ = load_sources(args.root)
    output = run_dir(args.root, args.run)
    code = code_md5(args.root)
    summary = []
    for case_id in args.ids:
        started = perf_counter()
        saved_path = args.root / SAVED_ESTIMATORS / f'{case_id}.json.gz'
        saved = read(saved_path)
        arms_record = read(output / 'e2' / f'{case_id}.json.gz')
        assert arms_record['case_id'] == case_id and arms_record['controls_loaded'] is False
        control = frozen_control(case_id, args.root, native_size(sources[case_id]), tuple(saved['working_size']))
        replay_path = args.root / SVD_RECORDS / f'{case_id}.json.gz'
        replay_record = read(replay_path) if replay_path.exists() else None
        result = fit_case(case_id, saved, arms_record, control, replay_record)
        result.update({'run': args.run, 'code_md5': code, 'saved_estimator_md5': md5(saved_path),
                       'e2_record_md5': md5(output / 'e2' / f'{case_id}.json.gz'),
                       'replay_record_md5': None if replay_record is None else md5(replay_path),
                       'elapsed_s': perf_counter() - started})
        write(output / 'e3' / f'{case_id}.json.gz', result)
        best = {name: (None if result['sets'][name].get('summary') is None else
                       result['sets'][name]['summary']['best_finite']['max_corner_working_px'])
                for name in SET_NAMES}
        summary.append({'case_id': case_id, 'control_source': control['control_source'],
                        'visually_approved': control['visually_approved'], 'best_finite_px': best,
                        'delta_vs_B_px': result['best_finite_delta_vs_B_px'],
                        'failed': {name: result['sets'][name].get('summary', {}).get('failed') for name in SET_NAMES},
                        'replay_passed': None if result['replay_against_svd_fixed'] is None else True,
                        'elapsed_s': result['elapsed_s']})
        print(case_id, 'best finite px', {name: None if value is None else round(value, 3) for name, value in best.items()},
              'replay', summary[-1]['replay_passed'], 'seconds', round(result['elapsed_s'], 1), flush=True)
    write(output / 'e3' / 'summary.json.gz', {'schema': 'direction-agreement-e3-summary/1', 'run': args.run,
                                               'code_md5': code, 'cases': summary})


if __name__ == '__main__':
    main()
