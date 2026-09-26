"""Diagnose the full frozen VP bank against an inspected control after generation."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from diagnose_directions import control_fit
from run_diagnosis import read, write

from experiments.annotator.independent_court import detector

TOP_PER_AXIS = 4


def reconstruct_bank(estimator: dict) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Rebuild estimator candidates in their saved normalised coordinates."""
    lines = np.asarray(estimator['direction_lines'], dtype=float)
    transform = np.asarray(estimator['normalised_to_working'], dtype=float)
    normalised_lines = lines @ transform
    pairs = np.asarray(list(combinations(range(len(lines)), 2)), dtype=int).reshape(-1, 2)
    intersections = np.cross(normalised_lines[pairs[:, 0]], normalised_lines[pairs[:, 1]])
    infinity = np.column_stack((normalised_lines[:, 1], -normalised_lines[:, 0], np.zeros(len(lines))))
    raw_candidates = np.concatenate((intersections, infinity))
    norms = np.linalg.norm(raw_candidates, axis=1)
    nondegenerate = norms > 1e-12
    candidate_ids = np.flatnonzero(nondegenerate)
    np.testing.assert_array_equal(candidate_ids, np.asarray(estimator['candidate_ids']))
    np.testing.assert_equal(len(candidate_ids), len(estimator['support_counts']))
    np.testing.assert_equal(len(candidate_ids), len(estimator['candidate_status']))
    candidates = raw_candidates[nondegenerate] / norms[nondegenerate, None]
    generators = [pair.tolist() for pair in pairs]
    generators.extend([[line_id] for line_id in range(len(lines))])
    return candidates, transform, [generators[index] for index in candidate_ids]


def control_homography(automatic: dict, given: dict, native_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Recover the inspected control homography and its working-frame corners."""
    scale = np.asarray(native_size, dtype=float) / np.asarray(automatic['working_size'], dtype=float)
    corners = np.asarray(given['control_corners_px'], dtype=float) / scale
    homography = cv2.getPerspectiveTransform(detector.CORNER_COURT_M, corners.astype(np.float32))
    return homography, corners


def ranked_axis(
    candidates: np.ndarray,
    working_points: np.ndarray,
    candidate_ids: np.ndarray,
    statuses: np.ndarray,
    supports: np.ndarray,
    generators: list[list[int]],
    target: np.ndarray,
) -> dict:
    """Rank the bank by signed-invariant homogeneous angular distance."""
    target /= np.linalg.norm(target)
    absolute_dot = np.abs(candidates @ target)
    distance = np.arccos(np.clip(absolute_dot, 0., 1.))
    order = np.lexsort((candidate_ids, distance))[:TOP_PER_AXIS]
    entries = []
    for rank, index in enumerate(order):
        entries.append({
            'rank': rank,
            'candidate_id': int(candidate_ids[index]),
            'candidate_status': str(statuses[index]),
            'support_count': int(supports[index]),
            'absolute_dot': float(absolute_dot[index]),
            'angular_distance_rad': float(distance[index]),
            'angular_distance_deg': float(np.degrees(distance[index])),
            'point_normalised': candidates[index].tolist(),
            'point_working': working_points[index].tolist(),
            'direction_line_ids': generators[index],
        })
    return {'target_point_normalised': target.tolist(), 'candidates': entries}


def fit_shortlisted(
    ranked_axes: tuple[dict, dict], working_points: np.ndarray, candidate_ids: np.ndarray,
    corners: np.ndarray,
) -> tuple[list[dict], dict | None]:
    """Fit every distinct cross-product of the two four-candidate shortlists."""
    index_by_id = {int(candidate_id): index for index, candidate_id in enumerate(candidate_ids)}
    fits = []
    for x_rank, x_entry in enumerate(ranked_axes[0]['candidates']):
        for y_rank, y_entry in enumerate(ranked_axes[1]['candidates']):
            if x_entry['candidate_id'] == y_entry['candidate_id']:
                continue
            x_index = index_by_id[x_entry['candidate_id']]
            y_index = index_by_id[y_entry['candidate_id']]
            result = control_fit(working_points[[x_index, y_index]], corners)
            fits.append({'x_rank': x_rank, 'y_rank': y_rank,
                         'x_candidate_id': x_entry['candidate_id'],
                         'y_candidate_id': y_entry['candidate_id'], **result})
    best = min(fits, key=lambda entry: (entry['max_corner_working_px'], entry['x_candidate_id'],
                                        entry['y_candidate_id']), default=None)
    return fits, best


def diagnose(automatic: dict, given: dict, native_size: tuple[int, int]) -> dict:
    """Compare every bank direction with the inspected control after generation."""
    estimator = automatic['estimator']
    candidates, transform, generators = reconstruct_bank(estimator)
    candidate_ids = np.asarray(estimator['candidate_ids'], dtype=int)
    statuses = np.asarray(estimator['candidate_status'])
    supports = np.asarray(estimator['support_counts'], dtype=int)
    working_points = candidates @ transform.T
    selected_rows = np.searchsorted(candidate_ids, estimator['retained_candidate_ids'])
    np.testing.assert_allclose(working_points[selected_rows], estimator['points_working'], rtol=1e-12, atol=1e-12)
    homography, corners = control_homography(automatic, given, native_size)
    control_working_points = homography[:, :2]
    control_normalised_points = np.linalg.solve(transform, control_working_points)
    ranked_axes = tuple(
        ranked_axis(candidates, working_points, candidate_ids, statuses, supports, generators,
                    control_normalised_points[:, axis])
        for axis in range(2)
    )
    fits, best = fit_shortlisted(ranked_axes, working_points, candidate_ids, corners)
    return {
        'schema': 'automatic-direction-bank-diagnostic/1',
        'case_id': automatic['case_id'],
        'label_guided_diagnostic_only': True,
        'selection_unchanged': True,
        'note': ('The control ranks the saved VP bank after generation. Angular distance is a signed-invariant '
                 'homogeneous direction metric, not pixel error. Fits are measured least-squares diagnostics, '
                 'not generated hypotheses or certified global optima.'),
        'working_size': automatic['working_size'],
        'native_size': list(native_size),
        'control_source': given.get('given_direction_source'),
        'control_corners_working_px': corners.tolist(),
        'control_points_normalised': control_normalised_points.T.tolist(),
        'estimator_candidate_count': len(candidates),
        'selected_estimator_candidate_ids': estimator['retained_candidate_ids'],
        'axes': list(ranked_axes),
        'shortlisted_cross_product_fits': fits,
        'best_measured_fit': best,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--automatic', type=Path, required=True)
    parser.add_argument('--given', type=Path, required=True)
    parser.add_argument('--native-size', nargs=2, type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    automatic, given = read(args.automatic), read(args.given)
    result = diagnose(automatic, given, tuple(args.native_size))
    write(args.output, result)
    best = result['best_measured_fit']
    print(result['case_id'], 'bank', result['estimator_candidate_count'],
          'fits', len(result['shortlisted_cross_product_fits']),
          'best_max_corner_working_px', None if best is None else best['max_corner_working_px'], flush=True)


if __name__ == '__main__':
    main()
