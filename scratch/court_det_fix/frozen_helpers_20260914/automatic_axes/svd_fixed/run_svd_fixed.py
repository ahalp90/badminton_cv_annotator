"""Compare fixed-support SVD directions with frozen directions in saved records."""

from __future__ import annotations

import argparse
import hashlib
from itertools import combinations, permutations
from pathlib import Path
from time import perf_counter

import numpy as np
from diagnose_directions import control_fit
from run_diagnosis import read, write
from vp_pruning import angular_residuals


def svd_direction(lines: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fit a unit homogeneous point to the supplied normalised line rows.

    :param lines: One homogeneous line per row, with three coefficients per line.
    :return: Direction and singular-value diagnostics; membership stays fixed.
    """
    assert lines.ndim == 2 and lines.shape[1] == 3 and len(lines) >= 2
    # Unit 2D normals remove arbitrary line scale without weighting by the offset.
    lines = lines / np.linalg.norm(lines[:, :2], axis=1)[:, None]
    _, values, right = np.linalg.svd(lines, full_matrices=True)
    singular_values = np.pad(values, (0, 3 - len(values)))
    point = right[-1]
    return point, {
        'line_count': len(lines),
        'singular_values': singular_values.tolist(),
        'normalised_nullspace_gap': float((singular_values[1] - singular_values[2]) / singular_values[0]),
        'algebraic_rms': float(np.sqrt(np.mean(np.square(lines @ point)))),
    }


def fit_groups(lines: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Fit each original membership mask independently without changing its rows."""
    points, records = [], []
    for index, mask in enumerate(masks):
        point, record = svd_direction(lines[mask])
        points.append(point)
        records.append({'group_index': index, 'support_line_ids': np.flatnonzero(mask).tolist(), **record})
    return np.asarray(points), records


def pair_separations(points: np.ndarray) -> list[dict]:
    """Record all signed-invariant angular separations without a merge threshold."""
    unit = points / np.linalg.norm(points, axis=1)[:, None]
    records = []
    for first, second in combinations(range(len(points)), 2):
        cosine = np.clip(abs(unit[first] @ unit[second]), 0., 1.)
        records.append({'groups': [first, second], 'angle_deg': float(np.degrees(np.arccos(cosine)))})
    return sorted(records, key=lambda row: (row['angle_deg'], row['groups']))


def fit_pairs(points: np.ndarray, corners: np.ndarray) -> dict:
    """Evaluate every ordered pair; retain failed solver attempts explicitly."""
    started = perf_counter()
    records = []
    for pair_id, indices in enumerate(permutations(range(len(points)), 2)):
        record = {'pair_id': pair_id, 'groups': list(indices)}
        try:
            fit = control_fit(points[list(indices)], corners)
            if not np.isfinite(fit['max_corner_working_px']):
                raise ValueError('Non-finite corner error')
            record.update(fit)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            record['failure'] = f'{type(error).__name__}: {error}'
        records.append(record)
    finite = [record for record in records if 'failure' not in record]
    converged = [record for record in finite if record['converged']]
    return {
        'records': records,
        'attempted': len(records),
        'failed': len(records) - len(finite),
        'converged': len(converged),
        'best_finite': min(finite, key=lambda row: row['max_corner_working_px'], default=None),
        'best_converged': min(converged, key=lambda row: row['max_corner_working_px'], default=None),
        'elapsed_s': perf_counter() - started,
    }


def run_case(saved: dict, bank: dict) -> dict:
    """Compare one case against the exact target used in its saved bank diagnosis."""
    assert saved['case_id'] == bank['case_id']
    assert saved['working_size'] == bank['working_size']
    estimator = saved['estimator']
    assert estimator['retained_candidate_ids'] == bank['selected_estimator_candidate_ids']
    lines = np.asarray(estimator['direction_lines'])
    transform = np.asarray(estimator['normalised_to_working'])
    normalised_lines = lines @ transform
    masks = np.asarray(estimator['retained_support_masks'], dtype=bool)
    original_working = np.asarray(estimator['points_working'])
    original_normalised = np.linalg.solve(transform, original_working.T).T
    assert len(original_working) == len(masks) == 16
    np.testing.assert_array_equal(
        angular_residuals(normalised_lines, original_normalised) <= saved['settings']['angle_deg'], masks)

    started = perf_counter()
    refit_normalised, refit_records = fit_groups(normalised_lines, masks)
    refit_elapsed = perf_counter() - started
    refit_working = refit_normalised @ transform.T
    original_unit = original_normalised / np.linalg.norm(original_normalised, axis=1)[:, None]
    for record, original, fitted in zip(refit_records, original_unit, refit_normalised, strict=True):
        cosine = np.clip(abs(original @ fitted), 0., 1.)
        record['movement_deg'] = float(np.degrees(np.arccos(cosine)))
    corners = np.asarray(bank['control_corners_working_px'])
    baseline = fit_pairs(original_working, corners)
    refitted = fit_pairs(refit_working, corners)

    control_points = np.asarray(bank['control_points_normalised'])
    control_masks = angular_residuals(normalised_lines, control_points) <= saved['settings']['angle_deg']
    control_refit, control_records = fit_groups(normalised_lines, control_masks)
    control_comparison = fit_pairs(control_refit @ transform.T, corners)
    return {
        'schema': 'fixed-membership-svd-diagnostic/1',
        'case_id': saved['case_id'], 'working_size': saved['working_size'],
        'label_guided_evaluation': True, 'automatic_detection': False,
        'control_source': bank['control_source'], 'control_corners_working_px': corners.tolist(),
        'original_candidate_ids': estimator['retained_candidate_ids'],
        'normalised_to_working': transform.tolist(), 'membership_angle_deg': saved['settings']['angle_deg'],
        'svd_line_normalisation': 'unit_2d_normal_after_coordinate_transform',
        'refit_membership_changed': False, 'refit_uses_control': False,
        'original_points_working': original_working.tolist(), 'refit_points_working': refit_working.tolist(),
        'refit_groups': refit_records, 'refit_elapsed_s': refit_elapsed,
        'baseline': baseline, 'fixed_membership_svd': refitted,
        'original_pair_separations': pair_separations(original_normalised),
        'refit_pair_separations': pair_separations(refit_normalised),
        'control_selected_svd': {
            'label_guided_directions': True, 'known_correct_membership': False,
            'groups': control_records, 'points_working': (control_refit @ transform.T).tolist(),
            'fits': control_comparison,
        },
        'previous_best_bank_fit': bank['best_measured_fit'],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = [
        ('gxBQ_window_00_frame_0', 'gx0_control/bank_diagnosis.json.gz'),
        ('gxBQ_window_00_frame_5', 'gx5_bank_diagnosis.json.gz'),
        ('am2_window_01_frame_28019', 'am2_far_bank_diagnosis.json.gz'),
    ]
    for case_id, bank_file in cases:
        started = perf_counter()
        saved_path = args.root / 'vp_pruning_20260914/coverage/results' / f'{case_id}.json.gz'
        bank_path = args.root / 'automatic_axes_20260914' / bank_file
        saved, bank = read(saved_path), read(bank_path)
        result = run_case(saved, bank)
        result['source_record_md5'] = {
            'estimator': hashlib.md5(saved_path.read_bytes()).hexdigest(),
            'bank_diagnosis': hashlib.md5(bank_path.read_bytes()).hexdigest(),
        }
        if case_id == 'gxBQ_window_00_frame_0':
            previous = read(args.root / 'automatic_axes_20260914/gx0_control/diagnosis.json.gz')
            previous_errors = [row['max_corner_working_px'] for row in previous['frozen_direction_fits']]
            current_errors = [row['max_corner_working_px'] for row in result['baseline']['records']]
            np.testing.assert_allclose(current_errors, previous_errors, rtol=0., atol=1e-8)
            result['gx0_baseline_replay_atol'] = 1e-8
        result['elapsed_s'] = perf_counter() - started
        write(args.output / f'{case_id}.json.gz', result)
        summary = {'refit_s': result['refit_elapsed_s'], 'total_s': result['elapsed_s']}
        arms = {'baseline': result['baseline'], 'fixed_svd': result['fixed_membership_svd'],
                'control_selected_svd': result['control_selected_svd']['fits']}
        for arm, fits in arms.items():
            best = fits['best_finite']
            summary[arm] = None if best is None else best['max_corner_working_px']
            summary[f'{arm}_failures'] = fits['failed']
        print(case_id, summary, flush=True)


if __name__ == '__main__':
    main()
