"""Fit inspected controls to frozen direction pairs for after-generation diagnosis only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from run_diagnosis import read, write
from scipy.optimize import least_squares

from experiments.annotator.independent_court import detector


def control_fit(points: np.ndarray, corners: np.ndarray) -> dict:
    """Find a control fit with fixed direction columns; this does not use observed lines."""
    directions = points.T / np.linalg.norm(points, axis=1)
    court = detector.CORNER_COURT_M.astype(float)
    coefficients = np.zeros((4, 2, 4))
    for coordinate in range(2):
        coefficients[:, coordinate, :2] = court * (
            directions[coordinate] - corners[:, coordinate, None] * directions[2])
        coefficients[:, coordinate, coordinate + 2] = 1.
    initial = np.linalg.lstsq(coefficients.reshape(8, 4), corners.ravel(), rcond=None)[0]

    def transform(parameters: np.ndarray) -> np.ndarray:
        return np.column_stack((directions * parameters[:2], [parameters[2], parameters[3], 1.]))

    def residual(parameters: np.ndarray) -> np.ndarray:
        projected, _ = detector.project(transform(parameters)[None], court)
        return (projected[0] - corners).ravel()

    fit = least_squares(residual, initial, x_scale='jac', max_nfev=200)
    errors = residual(fit.x).reshape(4, 2)
    return {'max_corner_working_px': float(np.linalg.norm(errors, axis=1).max()),
            'rms_coordinate_working_px': float(np.sqrt(np.square(errors).mean())),
            'converged': bool(fit.success), 'evaluations': fit.nfev,
            'homography_working': transform(fit.x).tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--automatic', type=Path, required=True)
    parser.add_argument('--given', type=Path, required=True)
    parser.add_argument('--native-size', nargs=2, type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    automatic, given = read(args.automatic), read(args.given)
    scale = np.asarray(args.native_size) / automatic['working_size']
    control = np.asarray(given['control_corners_px']) / scale
    points = np.asarray(automatic['estimator']['points_working'])
    records = []
    for pair in automatic['pairs']:
        record = control_fit(points[pair['pencils']], control)
        records.append({'pair_id': pair['pair_id'], 'pencils': pair['pencils'],
                        'generation_status': pair['status'], **record})
    ranked = sorted(records, key=lambda record: record['max_corner_working_px'])
    write(args.output, {'label_guided_diagnostics': True, 'case_id': automatic['case_id'],
                        'control_source': given['given_direction_source'],
                        'note': 'Least-squares control fits, not generated candidates or certified global optima.',
                        'records': records})
    print(automatic['case_id'], [(row['pair_id'], round(row['max_corner_working_px'], 4), row['converged'],
                                 row['generation_status']) for row in ranked[:10]], flush=True)


if __name__ == '__main__':
    main()
